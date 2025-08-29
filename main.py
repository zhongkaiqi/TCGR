import argparse
import json
import os
import random
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # 供 compute_infonce_loss 使用
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='推荐系统训练脚本')
    
    # 训练参数
    parser.add_argument('--batch_size', default=128, type=int, help='训练批次大小')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader workers for parallel preprocessing')
    parser.add_argument('--lr', default=0.005, type=float, help='初始学习率')
    parser.add_argument('--maxlen', default=101, type=int, help='序列最大长度')
    
    # 模型结构参数
    parser.add_argument('--hidden_units', default=128, type=int, help='隐藏层维度')
    parser.add_argument('--num_blocks', default=8, type=int, help='Transformer块数量')
    parser.add_argument('--num_epochs', default=6, type=int, help='训练轮数')
    parser.add_argument('--num_heads', default=4, type=int, help='注意力头数量')
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--loss', choices=['bce', 'infonce'], default='infonce', help='Training loss type')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature')
    parser.add_argument('--sample_k', type=int, default=0, help='Use up to K in-batch negatives (0 = all)')
                        
    # 正则与优化
    parser.add_argument('--l2_emb', default=0.0, type=float, help='嵌入层的L2正则化系数（独立附加在item_emb上）')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='AdamW 的 weight decay 系数')
    parser.add_argument('--clip_norm', default=1.0, type=float, help='梯度裁剪的最大范数（0或负数表示不裁剪）')

    # 学习率调度
    parser.add_argument('--lr_scheduler', choices=['none', 'cosine', 'cosine_warmup', 'step', 'plateau'],
                        default='cosine_warmup', help='学习率调度策略')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='warmup 步数（cosine_warmup 生效）')
    parser.add_argument('--min_lr', type=float, default=1e-20, help='最小学习率（cosine/plateau 生效）')
    parser.add_argument('--lr_step_size', type=int, default=1, help='StepLR 的步长（单位：epoch）')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR/Plateau 的衰减因子')
    parser.add_argument('--lr_patience', type=int, default=2, help='Plateau 的耐心轮数（单位：epoch）')

    # 设备与运行
    parser.add_argument('--device', default='cuda', type=str, help='计算设备(cpu/cuda)')
    parser.add_argument('--inference_only', action='store_true', help='是否仅运行推理(不训练)')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型路径')
    parser.add_argument('--norm_first', action='store_true', help='是否在Transformer中使用Pre-LayerNorm')
    
    # 多模态嵌入特征ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str,
                        choices=[str(s) for s in range(81, 87)],
                        help='使用的多模态特征ID列表')
    
    # —— 新增：点击动作的编码（只在验证指标里使用）——
    parser.add_argument('--click_action_id', type=int, default=1, help='点击动作在 next_action_type 中的取值（默认1）')

    # 随机种子
    parser.add_argument('--seed', default=2023, type=int, help='随机种子，确保实验可重复性')
    return parser.parse_args()


# ======== 保持原样：不要改 compute_infonce_loss 定义 ========
def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
    """
    基于“正样本余弦 + in-batch negatives”的 InfoNCE：
      - 正样本：cos(seq_emb, pos_emb)
      - 负样本：与 batch 内所有 neg_emb 的点积（均做 L2 归一化）
      - mask：仅对有效位置参与损失
      - 温度：self.temp（缺省 0.07）

    形状约定（最后一维为隐藏维）：
      seq_embs: [B, L, D]
      pos_embs: [B, L, D]
      neg_embs: [*, D] 或 [B, L, N, D]（任意前缀维，最终会被展平为 M×D 参与 in-batch negative）
      loss_mask: [B, L]，bool/0-1
    """
    device = seq_embs.device
    B, L, D = seq_embs.shape
    temp = float(getattr(self, "temp", 0.07))

    # -------- Normalize (cosine) --------
    seq = F.normalize(seq_embs, dim=-1)
    pos = F.normalize(pos_embs, dim=-1)
    neg = F.normalize(neg_embs, dim=-1)

    # -------- Positive logits: cosine(seq, pos) --------
    pos_logits = F.cosine_similarity(seq, pos, dim=-1, eps=1e-8).unsqueeze(-1)  # [B, L, 1]

    # -------- In-batch negatives: seq vs ALL neg vectors --------
    # 展平所有负样本到 [M, D]
    neg_all = neg.reshape(-1, D)                       # [M, D]
    # 将 (B,L,D) 与 (M,D) 做两两点积 → [B, L, M]
    seq_flat = seq.reshape(-1, D)                      # [B*L, D]
    neg_logits_flat = torch.matmul(seq_flat, neg_all.t())  # [B*L, M]
    neg_logits = neg_logits_flat.view(B, L, -1)        # [B, L, M]

    # -------- 拼接 & 取有效样本 --------
    logits = torch.cat([pos_logits, neg_logits], dim=-1)        # [B, L, 1+M]
    if loss_mask.dtype != torch.bool:
        loss_mask = loss_mask.to(torch.bool)
    if loss_mask.any():
        logits_v = logits[loss_mask] / temp                      # [N_valid, 1+M]
        labels = torch.zeros(logits_v.size(0), device=device, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits_v, labels, reduction="mean")
    else:
        # 无有效样本：返回零且不参与梯度
        return (logits.sum() * 0.0)

    # -------- TensorBoard 打点（正/负相似度趋势）--------
    phase = "train" if torch.is_grad_enabled() else "valid"
    writer = getattr(self, "writer", None)
    if writer is None:
        writer = globals().get("writer", None)
    global_step = int(getattr(self, "global_step", 0))

    if writer is not None:
        # 只对有效位置统计
        pos_v = pos_logits[loss_mask].squeeze(-1)               # [N_valid]
        neg_v = neg_logits[loss_mask]                            # [N_valid, M]
        max_neg_v = neg_v.max(dim=1).values                      # [N_valid]
        mean_neg_v = neg_v.mean(dim=1)                           # [N_valid]

        # 软概率与 top1
        prob_v = torch.nn.functional.softmax(logits_v, dim=1)    # [N_valid, 1+M]
        pos_prob = prob_v[:, 0]
        top1_acc = (logits_v.argmax(dim=1) == 0).float().mean()

        # 统计量
        writer.add_scalar(f"Sim/{phase}/pos_mean",      pos_v.mean().item(),      global_step)
        writer.add_scalar(f"Sim/{phase}/neg_mean",      neg_v.mean().item(),      global_step)
        writer.add_scalar(f"Sim/{phase}/maxneg_mean",   max_neg_v.mean().item(),  global_step)
        writer.add_scalar(f"Sim/{phase}/margin_mean",   (pos_v - mean_neg_v).mean().item(), global_step)
        writer.add_scalar(f"Sim/{phase}/margin_vs_max", (pos_v - max_neg_v).mean().item(),  global_step)
        writer.add_scalar(f"Acc/{phase}/top1",          top1_acc.item(),          global_step)
        writer.add_scalar(f"Prob/{phase}/pos_mean",     pos_prob.mean().item(),   global_step)

        # 原始示例中的两项（保持一致的命名）
        writer.add_scalar("Model/nce_pos_logits", pos_v.mean().item(), global_step)
        writer.add_scalar("Model/nce_neg_logits", neg_v.mean().item(), global_step)

        # 直方图（抽样以避免过大）
        try:
            max_bins = 2048
            if pos_v.numel() > max_bins:
                idx = torch.randint(0, pos_v.numel(), (max_bins,), device=device)
                writer.add_histogram(f"Hist/{phase}/pos",     pos_v[idx].detach().cpu(),    global_step)
                writer.add_histogram(f"Hist/{phase}/max_neg", max_neg_v[idx].detach().cpu(),global_step)
            else:
                writer.add_histogram(f"Hist/{phase}/pos",     pos_v.detach().cpu(),         global_step)
                writer.add_histogram(f"Hist/{phase}/max_neg", max_neg_v.detach().cpu(),     global_step)
        except Exception:
            pass

    return loss
# ======== 保持原样：不要改 compute_infonce_loss 定义（到此结束） ========


def set_seed(seed):
    """设置所有随机种子确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ======== 新增：验证阶段点击命中率的工具函数 ========
def _click_mask(next_token_type: torch.Tensor, next_action_type: torch.Tensor, click_action_id: int) -> torch.Tensor:
    """
    仅统计【点击位置】且为 item 的位置：
    mask = (next_token_type == 1) & (next_action_type == click_action_id)
    返回 bool 张量，形状 [B, L]
    """
    m = (next_token_type == 1)
    m = m & (next_action_type == click_action_id)
    return m.to(torch.bool)


def _hit_rate_infonce(seq_embs: torch.Tensor,
                      pos_embs: torch.Tensor,
                      neg_embs: torch.Tensor,
                      mask: torch.Tensor) -> (float, int):
    """
    用与 InfoNCE 一致的余弦相似度，计算点击位置 top-1 命中率（pos > 所有 neg）。
    返回 (hits, total)
    """
    device = seq_embs.device
    B, L, D = seq_embs.shape
    # 归一化
    seq = F.normalize(seq_embs, dim=-1)
    pos = F.normalize(pos_embs, dim=-1)
    neg = F.normalize(neg_embs, dim=-1)

    pos_logits = (seq * pos).sum(dim=-1)  # [B, L]
    neg_all = neg.reshape(-1, D)          # [M, D]
    seq_flat = seq.reshape(-1, D)         # [B*L, D]
    neg_logits = (seq_flat @ neg_all.t()).view(B, L, -1)  # [B, L, M]
    max_neg = neg_logits.max(dim=-1).values                # [B, L]

    mask = mask.to(device)
    if mask.any():
        hits = (pos_logits > max_neg).to(torch.float32)[mask].sum().item()
        total = int(mask.sum().item())
        return hits, total
    else:
        return 0.0, 0


def _hit_rate_bce(pos_logits: torch.Tensor,
                  neg_logits: torch.Tensor,
                  mask: torch.Tensor) -> (float, int):
    """
    兼容 BCE 分支：若 neg_logits 为 [B,L,N]，使用 N 维的 max；若为 [B,L] 直接比较。
    返回 (hits, total)
    """
    # squeeze 掉可能的最后一维=1
    if pos_logits.dim() == 3 and pos_logits.size(-1) == 1:
        pos_logits = pos_logits.squeeze(-1)
    if neg_logits.dim() == 3:
        neg_max = neg_logits.max(dim=-1).values
    else:
        neg_max = neg_logits  # [B,L]

    mask = mask.to(pos_logits.device)
    if mask.any():
        hits = (pos_logits > neg_max).to(torch.float32)[mask].sum().item()
        total = int(mask.sum().item())
        return hits, total
    else:
        return 0.0, 0


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)

    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')

    # 让 writer / global_step 成为模块级变量，供 compute_infonce_loss 写入 TB
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    global_step = 0

    # 数据
    data_path = os.environ.get('TRAIN_DATA_PATH')
    dataset = MyDataset(data_path, args)

    generator = torch.Generator().manual_seed(args.seed)

    # 划分训练/验证
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [0.9, 0.1], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn, pin_memory=(args.device.startswith("cuda")), persistent_workers=(args.num_workers>0),
        generator=generator
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn, pin_memory=(args.device.startswith("cuda")), persistent_workers=(args.num_workers>0)
    )

    # 模型
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 把温度/写日志句柄挂到 model 上，供 compute_infonce_loss 读取（不改函数体）
    model.temp = args.temperature
    model.writer = writer

    # 参数初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # padding embedding 置 0
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    # 载入预训练
    if args.state_dict_path is not None:
        try:
            state = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            # 允许老权重缺少新增的 LayerNorm 参数
            incompatible = model.load_state_dict(state, strict=False)
            # 打印信息，便于核对
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing or unexpected:
                print(f"[load_state_dict(strict=False)] missing={missing}, unexpected={unexpected}")
    
            # 若文件名里带 epoch=xx，按你原来的逻辑恢复起始轮
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1 if 'epoch=' in args.state_dict_path else 1
        except Exception as e:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError(f'failed loading state_dicts: {e}')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # ===== 优化器：AdamW + weight decay（参数组：bias/Norm不做WD）=====
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or n.endswith('.bias') or 'norm' in n.lower() or 'layernorm' in n.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98)
    )

    # ===== 学习率调度 =====
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.num_epochs * steps_per_epoch)

    scheduler = None
    scheduler_step_on_batch = False  # True=每步更新；False=每epoch或基于指标更新

    if args.lr_scheduler == 'cosine_warmup':
        warmup_steps = max(1, min(args.warmup_steps, total_steps - 1))
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # 余弦退火到 0；优化器的 base_lr = args.lr
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_step_on_batch = True

    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr
        )
        scheduler_step_on_batch = True

    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.lr_step_size), gamma=args.lr_gamma
        )
        scheduler_step_on_batch = False

    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=max(1, args.lr_patience),
            min_lr=args.min_lr, verbose=False
        )
        scheduler_step_on_batch = False

    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch

            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            loss_mask = (next_token_type == 1).to(args.device)  # [B, L] 仅 item 位置参与损失

            optimizer.zero_grad()

            if args.loss == 'infonce':
                # 1) 序列表征（含用户特征）
                seq_embs = model.log2feats(seq, token_type, seq_feat)  # [B, L, D]
                # 2) 正样本表征（仅 item 端）
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)  # [B, L, D]
                # 3) 负样本池：使用 **batch 内所有 neg_embs**（不混入他人的 pos）
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)   # [B, L, D] 或 [B, L, N, D]
                # 4) TB 步数
                model.global_step = global_step

                # 正确调用：self + 四个张量（保持 compute_infonce_loss 不变）
                loss = compute_infonce_loss(model, seq_embs, pos_embs, neg_embs, loss_mask)
            else:
                # 原 BCE 分支保持不变
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(next_token_type == 1)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # L2 正则（embedding）
            if args.l2_emb > 0:
                reg = 0.0
                for p in model.item_emb.parameters():
                    reg = reg + torch.norm(p)
                loss = loss + args.l2_emb * reg

            # 反向 + 梯度裁剪 + 更新
            loss.backward()
            if args.clip_norm is not None and args.clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                # 可选记录梯度范数
                writer.add_scalar('Grad/global_norm', float(total_norm), global_step)

            optimizer.step()

            # 学习率调度（按 step）
            if scheduler is not None and scheduler_step_on_batch:
                scheduler.step()

            # 训练日志
            log_data = {
                'global_step': global_step,
                'loss': float(loss.item()),
                'epoch': epoch,
                'time': time.time(),
                'seed': args.seed
            }
            log_file.write(json.dumps(log_data) + '\n')
            log_file.flush()
            print(json.dumps(log_data))

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', float(loss.item()), global_step)
            writer.add_scalar('LR', current_lr, global_step)

            global_step += 1  # 放在 step 之后，下一次统计会用到新的 global_step

        # ===== 验证 =====
        model.eval()
        valid_loss_sum = 0.0
        click_hits_total = 0.0
        click_count_total = 0

        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch

            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            loss_mask = (next_token_type == 1).to(args.device)

            # —— 点击位置 mask（仅统计点击 & item 位置）——
            click_mask = _click_mask(next_token_type, next_action_type, args.click_action_id)

            with torch.no_grad():
                if args.loss == 'infonce':
                    # 与训练同样的调用方式：neg_embs 来自 batch 内 neg
                    seq_embs = model.log2feats(seq, token_type, seq_feat)
                    pos_embs = model.feat2emb(pos, pos_feat, include_user=False)
                    neg_embs = model.feat2emb(neg, neg_feat, include_user=False)
                    model.global_step = global_step  # 便于 TB 记录 valid 曲线
                    loss = compute_infonce_loss(model, seq_embs, pos_embs, neg_embs, loss_mask)

                    # —— 新增：点击位置命中率（InfoNCE 版本）——
                    h, t = _hit_rate_infonce(seq_embs, pos_embs, neg_embs, click_mask)
                    click_hits_total += h
                    click_count_total += t

                else:
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    pos_labels = torch.ones(pos_logits.shape, device=args.device)
                    neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                    indices = np.where(next_token_type == 1)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                    # —— 新增：点击位置命中率（BCE 版本）——
                    h, t = _hit_rate_bce(pos_logits, neg_logits, click_mask.to(args.device))
                    click_hits_total += h
                    click_count_total += t

                valid_loss_sum += float(loss.item())

        valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
        writer.add_scalar('Loss/valid', valid_loss_avg, global_step)

        # 点击命中率（防止除零）
        if click_count_total > 0:
            click_hit_rate = float(click_hits_total) / float(click_count_total)
        else:
            click_hit_rate = 0.0  # 若验证集中没有点击位置，记为0并给出计数

        # 打点与日志输出
        writer.add_scalar('Metric/valid_click_hit_rate', click_hit_rate, global_step)
        writer.add_scalar('Metric/valid_click_count', click_count_total, global_step)
        print(f"[valid] epoch={epoch} loss={valid_loss_avg:.6f} click_hit_rate={click_hit_rate:.6f} "
              f"({int(click_hits_total)}/{int(click_count_total)})")

        # 记录到日志文件（按 epoch 的汇总）
        epoch_log = {
            'epoch': epoch,
            'valid_loss': valid_loss_avg,
            'valid_click_hit_rate': click_hit_rate,
            'click_hits': int(click_hits_total),
            'click_total': int(click_count_total),
            'time': time.time(),
            'seed': args.seed
        }
        log_file.write(json.dumps(epoch_log) + '\n')
        log_file.flush()

        # 学习率调度（按 epoch / 指标）
        if scheduler is not None and not scheduler_step_on_batch:
            if args.lr_scheduler == 'plateau':
                scheduler.step(valid_loss_avg)
            else:
                scheduler.step()

        # 保存检查点
        save_dir = Path(
            os.environ.get('TRAIN_CKPT_PATH'),
            f"global_step{global_step}.seed={args.seed}_valid_loss={valid_loss_avg:.4f}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
