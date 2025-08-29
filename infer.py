# infer.py — with HitRate@10 & NDCG@10 logging
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import HSTUModel


# ========== 基础工具 ==========

def safe_str(x):
    """把 id/键 统一成字符串，避免 int/str 混用带来的对不上。"""
    try:
        # 兼容 numpy 标量
        if isinstance(x, (np.integer,)):
            return str(int(x))
        # 数字字符串或数字
        if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().lstrip('-').isdigit()):
            return str(int(float(x)))
        return str(x)
    except Exception:
        return str(x)


def get_ckpt_path():
    """递归查找 MODEL_OUTPUT_PATH 下最新的 .pt 权重文件。"""
    root = os.environ.get("MODEL_OUTPUT_PATH")
    if not root:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not os.path.exists(root):
        raise ValueError(f"MODEL_OUTPUT_PATH does not exist: {root}")

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pt"):
                full = os.path.join(dirpath, fn)
                try:
                    mtime = os.path.getmtime(full)
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, full))

    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under {root}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def get_args():
    parser = argparse.ArgumentParser()

    # Eval params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxlen', default=101, type=int)

    # Model
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MM emb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    return parser.parse_args()


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        num_points_query = struct.unpack('I', f.read(4))[0]
        query_ann_top_k = struct.unpack('I', f.read(4))[0]
        num_result_ids = num_points_query * query_ann_top_k
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)
        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """冷启动特征把字符串置 0。"""
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if isinstance(feat_value, list):
            processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
        elif isinstance(feat_value, str):
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, dataset=None):
    """
    生产候选库 item 的 id 和 embedding，并保存到 EVAL_RESULT_PATH。
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0

            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]

            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    Path(os.environ.get('EVAL_RESULT_PATH')).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model.save_item_emb(
            item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'),
            batch_size=1024, dataset=dataset
        )

    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)

    return retrieve_id2creative_id


# ========== 新增：真值加载 & 指标计算 ==========

def try_load_ground_truth_from_dataset(dataset):
    """
    兼容多种常见命名：从 MyTestDataset 里找 user -> 真值 creative_id(或列表) 的映射。
    """
    candidate_attr = [
        "user_label_dict", "label_dict", "gt_dict", "user2label", "labels_by_user",
        "user2creative", "uid2positive"
    ]
    for name in candidate_attr:
        if hasattr(dataset, name):
            d = getattr(dataset, name)
            if isinstance(d, dict) and len(d) > 0:
                gt = {}
                for k, v in d.items():
                    uk = safe_str(k)
                    if isinstance(v, (list, tuple, set)):
                        gt[uk] = {safe_str(x) for x in v}
                    else:
                        gt[uk] = {safe_str(v)}
                return gt
    return None


def try_load_ground_truth_from_files():
    """
    1) 优先读 EVAL_LABEL_PATH 指定的 .jsonl/.json
    2) 兜底尝试 EVAL_DATA_PATH 下常见文件
    返回：dict user_id(str) -> set(creative_id str)
    """
    def load_jsonl(path):
        gt = {}
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                # 常见字段名尝试
                uid = obj.get("user_id", obj.get("uid", obj.get("user")))
                if uid is None:
                    continue
                uid = safe_str(uid)
                if "labels" in obj and isinstance(obj["labels"], (list, tuple, set)):
                    gt[uid] = {safe_str(x) for x in obj["labels"]}
                elif "creative_id" in obj:
                    gt.setdefault(uid, set()).add(safe_str(obj["creative_id"]))
                elif "positive_creative_id" in obj:
                    gt.setdefault(uid, set()).add(safe_str(obj["positive_creative_id"]))
        return gt

    def load_json(path):
        obj = json.load(open(path, "r"))
        gt = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                uk = safe_str(k)
                if isinstance(v, (list, tuple, set)):
                    gt[uk] = {safe_str(x) for x in v}
                else:
                    gt[uk] = {safe_str(v)}
        elif isinstance(obj, list):
            # list of records
            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                uid = rec.get("user_id", rec.get("uid", rec.get("user")))
                if uid is None:
                    continue
                uid = safe_str(uid)
                if "labels" in rec and isinstance(rec["labels"], (list, tuple, set)):
                    gt[uid] = {safe_str(x) for x in rec["labels"]}
                elif "creative_id" in rec:
                    gt.setdefault(uid, set()).add(safe_str(rec["creative_id"]))
                elif "positive_creative_id" in rec:
                    gt.setdefault(uid, set()).add(safe_str(rec["positive_creative_id"]))
        return gt

    # 1) 环境变量指定
    label_path = os.environ.get("EVAL_LABEL_PATH")
    if label_path and os.path.exists(label_path):
        if label_path.endswith(".jsonl"):
            gt = load_jsonl(label_path)
            if gt:
                return gt
        if label_path.endswith(".json"):
            gt = load_json(label_path)
            if gt:
                return gt

    # 2) 兜底在 EVAL_DATA_PATH 下尝试
    data_root = os.environ.get("EVAL_DATA_PATH")
    if data_root and os.path.isdir(data_root):
        candidates = [
            "test_labels.jsonl", "eval_labels.jsonl", "ground_truth.jsonl",
            "labels.jsonl", "truth.jsonl", "test_set_with_label.jsonl",
            "test_labels.json", "eval_labels.json", "ground_truth.json",
            "labels.json", "truth.json"
        ]
        for name in candidates:
            p = os.path.join(data_root, name)
            if os.path.exists(p):
                if p.endswith(".jsonl"):
                    gt = load_jsonl(p)
                else:
                    gt = load_json(p)
                if gt:
                    return gt
    return None


def dcg_at_k(pred_list, gt_set, k=10):
    dcg = 0.0
    for rank, cid in enumerate(pred_list[:k], start=1):
        rel = 1.0 if safe_str(cid) in gt_set else 0.0
        if rel > 0:
            dcg += rel / np.log2(rank + 1.0)
    return dcg


def ndcg_at_k(pred_list, gt_set, k=10):
    dcg = dcg_at_k(pred_list, gt_set, k)
    # IDCG：有多少个 relevant，就有多少个 1/log2(i+1) 的理想值；但最多 k 个
    ideal_rels = min(k, len(gt_set))
    if ideal_rels == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_rels))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_and_log_metrics(top10s, user_list, gt_map, k=10):
    """
    top10s: List[List[creative_id]]
    user_list: List[user_id]
    gt_map: dict user_id(str) -> set(creative_id str)
    """
    if not gt_map:
        print("[Metrics] Ground truth NOT found — skip metric computation.")
        return None

    hits, ndcg_sum, cnt, miss = 0, 0.0, 0, 0
    for preds, uid in zip(top10s, user_list):
        uid_str = safe_str(uid)
        gts = gt_map.get(uid_str)
        if not gts:
            miss += 1
            continue
        cnt += 1
        # HitRate@10
        if any(safe_str(cid) in gts for cid in preds[:k]):
            hits += 1
        # NDCG@10
        ndcg_sum += ndcg_at_k(preds, gts, k=k)

    if cnt == 0:
        print("[Metrics] No valid samples with ground truth. miss=", miss)
        return None

    hitrate = hits / float(cnt)
    ndcg = ndcg_sum / float(cnt)

    # 打印到日志（stdout）
    print(f"[Metrics] evaluated={cnt}, missing_gt={miss}, total_queries={len(user_list)}")
    print(f"[Metrics] HitRate@{k}={hitrate:.6f}, NDCG@{k}={ndcg:.6f}")

    # 写文件
    out_dir = Path(os.environ.get('EVAL_RESULT_PATH', './results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "evaluated": cnt,
                "missing_gt": miss,
                "total_queries": len(user_list),
                f"hitrate@{k}": hitrate,
                f"ndcg@{k}": ndcg,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"hitrate": hitrate, "ndcg": ndcg, "evaluated": cnt, "missing": miss}


# ========== 主流程 ==========

def infer():
    args = get_args()

    # 数据
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
        pin_memory=(args.device.startswith("cuda")),
    )

    # 模型
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    # 加载 checkpoint（strict=False 兼容新增的 LayerNorm）
    ckpt_path = get_ckpt_path()
    state = torch.load(ckpt_path, map_location=torch.device(args.device))
    incompatible = model.load_state_dict(state, strict=False)
    miss = getattr(incompatible, "missing_keys", [])
    unexp = getattr(incompatible, "unexpected_keys", [])
    if miss or unexp:
        print(f"[load_state_dict(strict=False)] missing={miss}, unexpected={unexp}")

    # 生成 query（用户）向量
    all_embs = []
    user_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            seq, token_type, seq_feat, user_id = batch
            seq = seq.to(args.device)
            logits = model.predict(seq, seq_feat, token_type)  # [B, H] 已做 L2 normalize
            all_embs.append(logits.detach().cpu().numpy().astype(np.float32))
            user_list += user_id

    all_embs = np.concatenate(all_embs, axis=0)

    # 生成候选库向量
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        dataset=test_dataset,
    )

    # 保存 query 向量
    Path(os.environ.get('EVAL_RESULT_PATH')).mkdir(parents=True, exist_ok=True)
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    # ANN 检索（使用平台提供的 faiss_demo）
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path=" + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
        + " --dataset_id_file_path="    + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
        + " --query_vector_file_path="  + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
        + " --result_id_file_path="     + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    # 取出 top-k（转 creative_id）
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))
    top10s = [top10s_untrimmed[i:i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    # ===== 新增：加载真值并计算指标 =====
    gt_map = try_load_ground_truth_from_files()
    if gt_map is None:
        gt_map = try_load_ground_truth_from_dataset(test_dataset)

    compute_and_log_metrics(top10s, user_list, gt_map, k=10)

    return top10s, user_list


if __name__ == "__main__":
    infer()

