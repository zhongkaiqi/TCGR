# dataset.py
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集
    """
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 路径 & 偏移量（懒打开文件句柄）
        self._load_data_and_offsets()

        # 侧信息
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), "r"))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)

        with open(self.data_dir / "indexer.pkl", "rb") as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer["i"])
            self.usernum = len(indexer["u"])
        self.indexer = indexer
        self.indexer_i_rev = {v: k for k, v in indexer["i"].items()}
        self.indexer_u_rev = {v: k for k, v in indexer["u"].items()}

        # 特征信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    # ---------- I/O 与多进程安全 ----------
    def _load_data_and_offsets(self):
        self._data_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, "seq_offsets.pkl"), "rb") as f:
            self.seq_offsets = pickle.load(f)
        self.data_file = None  # 惰性打开

    def _ensure_data_file(self):
        if not hasattr(self, "_data_path"):
            raise RuntimeError("Dataset _data_path not set. Did _load_data_and_offsets() run?")
        if getattr(self, "data_file", None) is None or getattr(self.data_file, "closed", True):
            if not Path(self._data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self._data_path}")
            self.data_file = open(self._data_path, "rb", buffering=1024 * 1024)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data_file = None

    def __del__(self):
        f = getattr(self, "data_file", None)
        try:
            if f is not None and not f.closed:
                f.close()
        except Exception:
            pass

    # ---------- JSON 读取与修复 ----------
    @staticmethod
    def _safe_json_loads(line_bytes):
        import re
        if line_bytes is None:
            return None
        if isinstance(line_bytes, bytes):
            s = line_bytes.decode("utf-8", errors="ignore")
        else:
            s = str(line_bytes)
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            s2 = re.sub(r",\s*([}\]])", r"\1", s)
            return json.loads(s2)
        except Exception:
            pass
        for ch in ("}", "]", '"'):
            p = s.rfind(ch)
            if p != -1:
                try:
                    return json.loads(s[: p + 1])
                except Exception:
                    continue
        return None

    def _load_user_data(self, uid):
        self._ensure_data_file()
        f = self.data_file
        try:
            start_off = self.seq_offsets[uid]
        except Exception:
            return []
        try:
            f.seek(start_off)
        except Exception:
            self.data_file = None
            self._ensure_data_file()
            f = self.data_file
            f.seek(start_off)

        line = f.readline()
        rec = self._safe_json_loads(line)

        if rec is None:
            try:
                if isinstance(self.seq_offsets, (list, tuple)):
                    end_off = self.seq_offsets[uid + 1] if uid + 1 < len(self.seq_offsets) else None
                else:
                    keys = sorted(self.seq_offsets.keys())
                    pos = keys.index(uid) if uid in keys else -1
                    end_off = self.seq_offsets[keys[pos + 1]] if (pos != -1 and pos + 1 < len(keys)) else None
            except Exception:
                end_off = None

            if end_off is not None and end_off > start_off:
                f.seek(start_off)
                chunk = f.read(end_off - start_off)
                rec = self._safe_json_loads(chunk)

        if rec is None:
            print(f"[WARN] Skip bad json for uid={uid} at pos={start_off}")
            return []
        return rec

    # ---------- 采样与 __getitem__（训练/验证） ----------
    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        目标：把 user token 永远固定在第 0 位；物品从右侧对齐。
        形成：user | PAD ... PAD | item ... item
        """
        user_sequence = self._load_user_data(uid)

        # 拆分出唯一的 user 以及全体 item（保留动作）
        user_tuple = None          # (u, user_feat, 2, action_type)
        items = []                 # [(i, item_feat, 1, action_type), ...]
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                user_tuple = (u, user_feat, 2, action_type)
            if i and item_feat:
                items.append((i, item_feat, 1, action_type))

        # 预分配
        L = self.maxlen + 1
        seq = np.zeros([L], dtype=np.int32)
        pos = np.zeros([L], dtype=np.int32)
        neg = np.zeros([L], dtype=np.int32)
        token_type = np.zeros([L], dtype=np.int32)
        next_token_type = np.zeros([L], dtype=np.int32)
        next_action_type = np.zeros([L], dtype=np.int32)

        seq_feat = np.empty([L], dtype=object)
        pos_feat = np.empty([L], dtype=object)
        neg_feat = np.empty([L], dtype=object)

        # 正集合，避免采到正样本
        ts = set()
        for i, feat, typ, act in items:
            if i:
                ts.add(i)

        # ------- 固定 user 在 index=0 -------
        if user_tuple is not None:
            u_id, u_feat, _, _ua = user_tuple
            u_feat = self.fill_missing_feat(u_feat, u_id)
            seq[0] = u_id
            token_type[0] = 2
            seq_feat[0] = u_feat

            # user 的下一个若是首个 item，则构造正负样本/动作
            if len(items) >= 1:
                nxt_i, nxt_feat, nxt_type, nxt_act = items[0]
                nxt_feat = self.fill_missing_feat(nxt_feat, nxt_i)
                next_token_type[0] = nxt_type  # 一定为 1
                if nxt_act is not None:
                    next_action_type[0] = nxt_act
                pos[0] = nxt_i
                pos_feat[0] = nxt_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[0] = neg_id
                neg_feat[0] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
        else:
            # 若未找到 user（理论不应发生），填默认特征，保持 0 值即可
            seq_feat[0] = self.feature_default_value

        # ------- 物品从右侧对齐（训练丢最后一个）-------
        idx = self.maxlen  # 最右端
        if len(items) >= 1:
            nxt = items[-1]  # (i, feat, 1, act)
        else:
            nxt = (0, {}, 1, None)

        for record_tuple in reversed(items[:-1]):  # 丢弃最后一个
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt

            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)

            # 写入右端；保留 index=0 给 user
            if idx <= 0:
                break

            seq[idx] = i
            token_type[idx] = type_  # 1
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)

            nxt = record_tuple
            idx -= 1

        # None → 默认值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_feat,
            pos_feat,
            neg_feat,
        )

    def __len__(self):
        try:
            return len(self.seq_offsets)
        except Exception:
            return 0

    # ---------- 特征信息 ----------
    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {
            "user_sparse": ["103", "104", "105", "109"],
            "item_sparse": [
                "100", "117", "111", "118", "101", "102",
                "119", "120", "114", "112", "121", "115", "122", "116",
            ],
            "item_array": [],
            "user_array": ["106", "107", "108", "110"],
            "item_emb": self.mm_emb_ids,
            "user_continual": [],
            "item_continual": [],
        }

        for fid in feat_types["user_sparse"]:
            feat_default_value[fid] = 0
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["item_sparse"]:
            feat_default_value[fid] = 0
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["item_array"]:
            feat_default_value[fid] = [0]
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["user_array"]:
            feat_default_value[fid] = [0]
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["user_continual"]:
            feat_default_value[fid] = 0
        for fid in feat_types["item_continual"]:
            feat_default_value[fid] = 0

        # item_emb 维度来自已加载的 mm_emb（若为空，用 0 维兜底）
        for fid in feat_types["item_emb"]:
            emb_dim = 0
            emb_map = self.mm_emb_dict.get(fid, {})
            if emb_map:
                any_vec = next(iter(emb_map.values()))
                emb_dim = np.asarray(any_vec, dtype=np.float32).shape[0]
            feat_default_value[fid] = np.zeros(emb_dim, dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat is None:
            feat = {}
        filled_feat = dict(feat)

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(filled_feat.keys())
        for fid in missing_fields:
            filled_feat[fid] = self.feature_default_value[fid]

        for fid in self.feature_types["item_emb"]:
            if item_id != 0:
                raw_item_id = self.indexer_i_rev.get(item_id, None)
                if raw_item_id is not None and raw_item_id in self.mm_emb_dict.get(fid, {}):
                    v = self.mm_emb_dict[fid][raw_item_id]
                    if isinstance(v, np.ndarray):
                        filled_feat[fid] = v
        return filled_feat

    # ---------- 批量拼接与预 tensor 化（训练/验证） ----------
    @staticmethod
    def legacy_collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def feat2tensor_prebatch(self, feature_array):
        B = len(feature_array)
        L = len(feature_array[0]) if B > 0 else 0
        out = {
            "user_sparse": {}, "item_sparse": {},
            "user_array": {}, "item_array": {},
            "user_continual": {}, "item_continual": {},
            "item_emb": {},
        }

        def collect(fid):
            return [[feature_array[b][t].get(fid, 0) for t in range(L)] for b in range(B)]

        # sparse
        for fid in self.feature_types.get("user_sparse", []):
            out["user_sparse"][fid] = torch.as_tensor(collect(fid), dtype=torch.long)
        for fid in self.feature_types.get("item_sparse", []):
            out["item_sparse"][fid] = torch.as_tensor(collect(fid), dtype=torch.long)

        # array -> pad [B,L,M]
        def pad3(fid):
            max_len = 0
            for b in range(B):
                for t in range(L):
                    v = feature_array[b][t].get(fid, [])
                    if isinstance(v, (list, tuple)):
                        if len(v) > max_len:
                            max_len = len(v)
                    else:
                        max_len = max(max_len, 1)
            arr = np.zeros((B, L, max_len), dtype=np.int64)
            for b in range(B):
                for t in range(L):
                    v = feature_array[b][t].get(fid, [])
                    if isinstance(v, (list, tuple)):
                        if v:
                            arr[b, t, : len(v)] = np.asarray(v, dtype=np.int64)
                    else:
                        arr[b, t, 0] = int(v)
            return torch.from_numpy(arr)

        for fid in self.feature_types.get("user_array", []):
            out["user_array"][fid] = pad3(fid)
        for fid in self.feature_types.get("item_array", []):
            out["item_array"][fid] = pad3(fid)

        # continual
        for fid in self.feature_types.get("user_continual", []):
            out["user_continual"][fid] = torch.as_tensor(collect(fid), dtype=torch.float32)
        for fid in self.feature_types.get("item_continual", []):
            out["item_continual"][fid] = torch.as_tensor(collect(fid), dtype=torch.float32)

        # item_emb (vector)
        for fid in self.feature_types.get("item_emb", []):
            emb_dim = None
            for b in range(B):
                for t in range(L):
                    v = feature_array[b][t].get(fid, None)
                    if v is not None:
                        vv = np.asarray(v, dtype=np.float32)
                        emb_dim = vv.shape[-1]
                        break
                if emb_dim is not None:
                    break
            if emb_dim is None:
                out["item_emb"][fid] = torch.zeros((B, L, 0), dtype=torch.float32)
                continue
            arr = np.zeros((B, L, emb_dim), dtype=np.float32)
            for b in range(B):
                for t in range(L):
                    v = feature_array[b][t].get(fid, None)
                    if v is not None:
                        vv = np.asarray(v, dtype=np.float32)
                        if vv.ndim == 1 and vv.shape[0] == emb_dim:
                            arr[b, t, :] = vv
                        elif vv.ndim == 2 and vv.shape[-1] == emb_dim:
                            arr[b, t, :] = vv.mean(axis=0)
            out["item_emb"][fid] = torch.from_numpy(arr)
        return out

    def collate_fn(self, batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = self.legacy_collate_fn(batch)
        return (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            self.feat2tensor_prebatch(seq_feat),
            self.feat2tensor_prebatch(pos_feat),
            self.feat2tensor_prebatch(neg_feat),
        )


class MyTestDataset(MyDataset):
    """
    测试数据集（推理阶段：使用完整序列，不丢最后一个位置）
    user 固定在 index=0，物品从右侧对齐
    """
    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self._data_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, "predict_seq_offsets.pkl"), "rb") as f:
            self.seq_offsets = pickle.load(f)
        self.data_file = None  # 惰性打开

    def _process_cold_start_feat(self, feat):
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        # 拆分 user + items（推理不需要动作，但保留以兼容处理）
        user_tuple = None          # (u_reid, user_feat, 2)
        items = []                 # [(i_reid, item_feat, 1), ...]
        user_id = None

        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple

            if u:
                if isinstance(u, str):
                    user_id = u
                else:
                    user_id = self.indexer_u_rev.get(u, str(u))

            if u and user_feat:
                u_reid = 0 if isinstance(u, str) else u
                user_feat = self._process_cold_start_feat(user_feat) if user_feat else {}
                user_tuple = (u_reid, user_feat, 2)

            if i and item_feat:
                i_reid = 0 if (i > self.itemnum) else i
                item_feat = self._process_cold_start_feat(item_feat) if item_feat else {}
                items.append((i_reid, item_feat, 1))

        # 预分配（完整序列，不丢最后一个）
        L = self.maxlen + 1
        seq = np.zeros([L], dtype=np.int32)
        token_type = np.zeros([L], dtype=np.int32)
        seq_feat = np.empty([L], dtype=object)

        # user 固定在 0
        if user_tuple is not None:
            u_id, u_feat, _ = user_tuple
            u_feat = self.fill_missing_feat(u_feat, u_id)
            seq[0] = u_id
            token_type[0] = 2
            seq_feat[0] = u_feat
        else:
            seq_feat[0] = self.feature_default_value

        # 物品从右对齐（不丢最后一个）
        idx = self.maxlen
        for i_reid, feat, _ in reversed(items):
            if idx <= 0:
                break
            feat = self.fill_missing_feat(feat, i_reid)
            seq[idx] = i_reid
            token_type[idx] = 1
            seq_feat[idx] = feat
            idx -= 1

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        if user_id is None:
            user_id = str(self.indexer_u_rev.get(uid, uid))

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        return len(self.seq_offsets)

    @staticmethod
    def legacy_collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)
        return seq, token_type, seq_feat, user_id

    # 测试集自己的 collate_fn（返回 4 个值，且把特征张量化）
    def collate_fn(self, batch):
        seq, token_type, seq_feat, user_id = self.legacy_collate_fn(batch)
        seq_feat_tensors = self.feat2tensor_prebatch(seq_feat)
        return seq, token_type, seq_feat_tensors, user_id


# ---------- 通用工具 ----------
def save_emb(emb, save_path):
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f"saving {save_path}")
    with open(Path(save_path), "wb") as f:
        f.write(struct.pack("II", num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc="Loading mm_emb"):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != "81":
            try:
                base_path = Path(mm_path, f"emb_{feat_id}_{shape}")
                for json_file in base_path.glob("*.json"):
                    with open(json_file, "r", encoding="utf-8") as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin["emb"]
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin["anonymous_cid"]: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == "81":
            with open(Path(mm_path, f"emb_{feat_id}_{shape}.pkl"), "rb") as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f"Loaded #{feat_id} mm_emb")
    return mm_emb_dict