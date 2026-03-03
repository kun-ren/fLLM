import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.dataset import OHLCDataset
from models.loss import ProfitLoss


class AttentionPooling(nn.Module):
    def __init__(self, d_model, time_decay='linear'):
        super().__init__()
        self.time_decay = time_decay
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, C, T, d]

        B, C, T, d = x.shape

        var_score = self.proj(x)  # to [B, C, T, 1]

        # weighted pooling
        var_weights = torch.softmax(var_score, dim=1)
        x_var = torch.sum(x * var_weights, dim=1)  # [B, T, d]

        # temporal sequence weight
        time_weights = self.get_time_weights(T, x.device)  # [T]
        x_time = (x_var * time_weights.view(1, T, 1)).sum(dim=1)  # [B, d]
        return x_time

    def get_time_weights(self, T, device):
        """
        生成时间权重，最近时间权重大
        输出 shape: [T]
        """
        idx = torch.arange(T, device=device).float()

        if self.time_decay == "linear":
            # 越靠后权重越大
            weights = (idx + 1) / T

        elif self.time_decay == "exp":
            # 指数衰减
            weights = torch.exp((idx - T + 1) / T)

        else:
            raise ValueError("Unsupported time_decay type")

        weights = weights / weights.sum()
        return weights  # [T]


# -----------------------------
# Embedding Head: MLP + LayerNorm + tanh
# -----------------------------

class EmbeddingHead(nn.Module):
    def __init__(self, d_model, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.pooling = AttentionPooling(d_model, time_decay='linear')
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),  # [D, d]
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding):
        # embedding: [B, C, L, d_model]
        x = self.pooling(embedding)  # x = [B, d]
        out = self.mlp(x)  # [B, C, L, output_dim]
        out = torch.tanh(out)
        return out


# -----------------------------
# Crossformer Encoder single time step version
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, d = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, d)
        return self.out(out)


class CrossformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.temporal_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.channel_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L, d = x.shape

        # Temporal Attention
        xt = x.reshape(B * C, L, d)
        xt2 = self.temporal_attn(xt)
        xt = self.norm1(xt + self.dropout1(xt2))
        xt = xt.view(B, C, L, d)

        # Cross-Channel Attention
        xc = xt.permute(0, 2, 1, 3).reshape(B * L, C, d)
        xc2 = self.channel_attn(xc)
        xc = self.norm2(xc + self.dropout2(xc2))
        xc = xc.view(B, L, C, d).permute(0, 2, 1, 3)

        # Feedforward
        x_ffn = self.ffn(xc)
        x = self.norm3(xc + self.dropout3(x_ffn))
        return x


class CrossformerEncoderTimeStep(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=3, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            CrossformerEncoderLayer(d_model, n_heads, dim_feedforward)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B, C, L]
        x = x.unsqueeze(-1)  # [B, C, L, 1]
        x = self.embedding(x)  # [B, C, L, d_model]
        for layer in self.layers:
            x = layer(x)
        return x  # [B, C, L, d_model]


# -----------------------------
# training
# -----------------------------
def train_crossformer(csv_path, seq_len=36, use_last_n=None,
                      batch_size=16, epochs=5, lr=1e-3, device='cuda'):
    # Dataset & DataLoader
    dataset, y_indices = OHLCDataset(csv_path, seq_len=seq_len,
                                     use_last_n=use_last_n,
                                     sliding_window=True,
                                     normalize=True,
                                     )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    C = dataset[0][0].shape[0]
    logging.log(f"Check number of channels: {C}")

    d_model = 32

    encoder = CrossformerEncoderTimeStep(input_dim=C, d_model=d_model, n_heads=6, n_layers=3).to(device)
    head = EmbeddingHead(d_model=d_model, hidden_dim=d_model * 2, output_dim=1).to(device)

    optimizer = torch.optim.AdamW(
        [{'params': encoder.parameters(), 'lr': 5e-5},
         {'params': head.parameters(), 'lr': 1e-4}],
        betas=(0.9, 0.999),  # 动量系数
        eps=1e-8,  # 数值稳定性分母
        weight_decay=0.05,  # 权重衰减 (L2 正则化)
        amsgrad=False  # 是否使用 AMSGrad 变体
    )
    loss_fn = ProfitLoss(dataset)

    encoder.train()
    head.train()

    for epoch in range(epochs):
        for index, (x, indices) in enumerate(loader):
            x = x.to(device)  # [B, C, L]

            # Encoder
            embedding = encoder(x)  # [B, C, L, d_model]

            # MLP Head + tanh
            out = head(embedding)  # [B, C, L],

            loss = loss_fn(out, indices)  # loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


# -----------------------------
# 使用示例
# -----------------------------
train_crossformer('data/ohlc_data.csv', use_last_n=500)
