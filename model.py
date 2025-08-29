"""HSTU model implementation for time-aware sequential recommendation.

This module implements the High-performance Sequential Transduction Unit (HSTU)
described in the paper *Action Speaks Louder than Words: Trillion-Parameter
Sequential Transducers for Generative Recommendations*.

The interface is kept compatible with the previous ``BaselineModel`` so that it
works seamlessly with ``dataset.py``, ``main.py`` and ``infer.py``.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    """Multi-head self-attention with optional FlashAttention path."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        bsz, seq_len, _ = query.size()

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
            )
        else:  # pragma: no cover - compatibility path
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)
        return output


class PointWiseFeedForward(torch.nn.Module):
    """Simple point-wise feed-forward network used inside HSTU blocks."""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs.transpose(-1, -2))
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        return out.transpose(-1, -2)


class HSTULayer(torch.nn.Module):
    """One HSTU block with gated residual connections."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attn_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = FlashMultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.attn_gate = torch.nn.Linear(hidden_units, hidden_units)

        self.ffn_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_units, dropout_rate)
        self.ffn_gate = torch.nn.Linear(hidden_units, hidden_units)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with gated residual
        h = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), attn_mask)
        gate = torch.sigmoid(self.attn_gate(x))
        x = x + self.dropout(gate * h)

        # Feed-forward with gated residual
        h = self.ffn(self.ffn_norm(x))
        gate = torch.sigmoid(self.ffn_gate(x))
        x = x + self.dropout(gate * h)
        return x


class HSTUModel(torch.nn.Module):
    """Time-aware sequential model built from stacked HSTU blocks."""

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.temp = getattr(args, "temp", 0.07)

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        self.userdnn_ln = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.itemdnn_ln = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.input_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Embedding tables for sparse features
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        self.hstu_layers = torch.nn.ModuleList(
            [HSTULayer(args.hidden_units, args.num_heads, args.dropout_rate) for _ in range(args.num_blocks)]
        )

        with torch.no_grad():
            self.user_emb.weight.zero_()

    # ------------------------------------------------------------------
    # feature preparation helpers (copied from previous BaselineModel)
    # ------------------------------------------------------------------
    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types["user_sparse"]}
        self.USER_CONTINUAL_FEAT = feat_types["user_continual"]
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types["item_sparse"]}
        self.ITEM_CONTINUAL_FEAT = feat_types["item_continual"]
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types["user_array"]}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types["item_array"]}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types["item_emb"]}

    # --- The following methods (feat2tensor, feat2emb, log2feats, forward,
    #     forward_inbatch_infonce, predict, save_item_emb) are adapted from the
    #     previous BaselineModel with minimal changes. ---

    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, features, mask=None, include_user=False):
        import numpy as np

        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list, user_feat_list = [item_embedding], [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list, user_feat_list = [item_embedding], []

        def get(group, fid):
            d = features.get(group, {})
            t = d.get(fid, None)
            return None if t is None else t.to(self.dev)

        for fid in self.ITEM_SPARSE_FEAT.keys():
            t = get("item_sparse", fid)
            if t is not None:
                item_feat_list.append(self.sparse_emb[fid](t))
        if include_user:
            for fid in self.USER_SPARSE_FEAT.keys():
                t = get("user_sparse", fid)
                if t is not None:
                    user_feat_list.append(self.sparse_emb[fid](t))

        for fid in self.ITEM_ARRAY_FEAT.keys():
            t = get("item_array", fid)
            if t is not None:
                item_feat_list.append(self.sparse_emb[fid](t).sum(2))
        if include_user:
            for fid in self.USER_ARRAY_FEAT.keys():
                t = get("user_array", fid)
                if t is not None:
                    user_feat_list.append(self.sparse_emb[fid](t).sum(2))

        for fid in self.ITEM_CONTINUAL_FEAT:
            t = get("item_continual", fid)
            if t is not None:
                item_feat_list.append(t.unsqueeze(2))
        if include_user:
            for fid in self.USER_CONTINUAL_FEAT:
                t = get("user_continual", fid)
                if t is not None:
                    user_feat_list.append(t.unsqueeze(2))

        for fid in self.ITEM_EMB_FEAT.keys():
            t = get("item_emb", fid)
            if t is not None and t.shape[-1] == self.ITEM_EMB_FEAT[fid]:
                item_feat_list.append(self.emb_transform[fid](t))

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        all_item_emb = self.itemdnn_ln(all_item_emb)

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            all_user_emb = self.userdnn_ln(all_user_emb)
            return all_user_emb + all_item_emb
        else:
            return all_item_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        seqs = self.input_layernorm(seqs)

        ones = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for layer in self.hstu_layers:
            seqs = layer(seqs, attention_mask)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    # -----------------------------
    # Training (BCE path with cosine similarity)
    # -----------------------------
    def forward(
        self,
        user_item,
        pos_seqs,
        neg_seqs,
        mask,
        next_mask,
        next_action_type,
        seq_feature,
        pos_feature,
        neg_feature,
    ):
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        log_feats = F.normalize(log_feats, dim=-1)
        pos_embs = F.normalize(pos_embs, dim=-1)
        neg_embs = F.normalize(neg_embs, dim=-1)

        pos_logits = (log_feats * pos_embs).sum(dim=-1) * loss_mask
        neg_logits = (log_feats * neg_embs).sum(dim=-1) * loss_mask
        return pos_logits, neg_logits

    # ------------------------------------
    # Training (in-batch InfoNCE path)
    # ------------------------------------
    def forward_inbatch_infonce(
        self,
        user_item,
        pos_seqs,
        neg_seqs,
        mask,
        next_mask,
        seq_feature,
        pos_feature,
        neg_feature,
        temperature: float = 0.1,
        sample_k: Optional[int] = None,
    ):
        log_feats = self.log2feats(user_item, mask, seq_feature)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        log_feats = F.normalize(log_feats, dim=-1)
        pos_embs = F.normalize(pos_embs, dim=-1)
        neg_embs = F.normalize(neg_embs, dim=-1)

        B, L, D = log_feats.shape
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_all = neg_embs.reshape(-1, D)
        q_flat = log_feats.reshape(-1, D)
        neg_logits_flat = torch.matmul(q_flat, neg_all.t())
        neg_logits = neg_logits_flat.view(B, L, -1)

        if sample_k is not None and sample_k > 0:
            M = neg_logits.shape[-1]
            k = min(sample_k, M)
            idx = torch.randint(0, M, (B, L, k), device=neg_logits.device)
            neg_logits = torch.gather(neg_logits, dim=2, index=idx)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        gold = torch.zeros((B, L), dtype=torch.long, device=logits.device)
        valid_mask = (next_mask == 1)
        return logits, gold, valid_mask

    # -----------------------------
    # Inference
    # -----------------------------
    def predict(self, log_seqs, seq_feature, mask):
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, dim=-1)
        return final_feat

    def save_item_emb(
        self,
        item_ids,
        retrieval_ids,
        feat_dict,
        save_path,
        batch_size: int = 1024,
        dataset=None,
    ):
        all_embs = []
        for start in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end = min(start + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start:end], device=self.dev).unsqueeze(1)

            batch_feat = []
            for i in range(start, end):
                batch_feat.append([feat_dict[i]])

            if dataset is not None and hasattr(dataset, "feat2tensor_prebatch"):
                prepared = dataset.feat2tensor_prebatch(batch_feat)
                batch_emb = self.feat2emb(item_seq, prepared, include_user=False).squeeze(1)
            else:
                batch_feat_np = np.array(batch_feat, dtype=object)
                batch_emb = self.feat2emb(item_seq, batch_feat_np, include_user=False).squeeze(1)

            batch_emb = F.normalize(batch_emb, dim=-1)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, "embedding.fbin"))
        save_emb(final_ids, Path(save_path, "id.u64bin"))


__all__ = ["HSTUModel"]

