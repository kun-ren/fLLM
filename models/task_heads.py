"""
Task-specific prediction heads for trading signal generation.
All heads take encoder output [B, C, L, d_model] and produce task-specific predictions.
"""
import torch
import torch.nn as nn

from models.pooling import build_pooling
from models.crossformer_lib.decoder import Decoder as CrossformerDecoder


class ReversalHead(nn.Module):
    """
    Predicts directional reversal confidence.

    Output: [B, 1] - reversal confidence in range [-1, 1]
        -1: strong downward reversal (bearish)
         0: no reversal / continuation
        +1: strong upward reversal (bullish)
    """
    def __init__(self, d_model, hidden_dim=128, dropout=0.2, pooling="AttentionPooling"):
        super().__init__()
        self.pooling = build_pooling(pooling, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output directional confidence [-1, 1]
        )

    def forward(self, embedding):
        """
        Args:
            embedding: [B, C, L, d_model]
        Returns:
            reversal_confidence: [B, 1] in [-1, 1]
        """
        #pooled = self.pooling(embedding)  # [B, d_model]
        return self.mlp(embedding)  # [B, 1]


class LongTermTrendHead(nn.Module):
    """
    Decoder that predicts future price sequence using Crossformer decoder.

    Output: [B, pred_len] - predicted price values for future time steps
    """
    def __init__(self, d_model, pred_len=24, hidden_dim=128, dropout=0.2, n_heads=4, d_layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # Use Crossformer decoder
        self.decoder = CrossformerDecoder(
            seg_len=pred_len,
            d_layers=d_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=hidden_dim,
            dropout=dropout,
            out_seg_num=pred_len,
            factor=10,
            router=False
        )

        # Project decoder output to price prediction
        self.price_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding):
        """
        Args:
            embedding: [B, C, L, d_model] - encoder output
        Returns:
            prices: [B, pred_len] - predicted future prices
        """
        B, C, L, d = embedding.shape

        # Reshape for decoder: [B, pred_len, d_model]
        # Use mean pooling over channels as initial query
        query = embedding.mean(dim=1)  # [B, L, d_model]

        # Take last pred_len steps or repeat if needed
        if L >= self.pred_len:
            query = query[:, -self.pred_len:, :]  # [B, pred_len, d_model]
        else:
            # Repeat to match pred_len
            repeat_factor = (self.pred_len + L - 1) // L
            query = query.repeat(1, repeat_factor, 1)[:, :self.pred_len, :]

        # Decode future representations
        decoded = self.decoder(query)  # [B, pred_len, d_model]

        # Project to price predictions
        prices = self.price_proj(decoded).squeeze(-1)  # [B, pred_len]

        return prices


class ResistanceHead(nn.Module):
    """
    Predicts potential resistance level for bull runs.

    Output: [B, 2]
        [:, 0]: resistance distance (% above current price)
        [:, 1]: confidence score [0, 1]
    """
    def __init__(self, d_model, hidden_dim=128, dropout=0.2, pooling="AttentionPooling"):
        super().__init__()
        self.pooling = build_pooling(pooling, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Resistance level (positive percentage)
        self.level_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive values
        )

        # # Confidence
        # self.confidence_head = nn.Sequential(
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, embedding):
        """
        Args:
            embedding: [B, C, L, d_model]
        Returns:
            resistance: [B, 2] - (level_%, confidence)
        """
        pooled = self.pooling(embedding)  # [B, d_model]
        features = self.mlp(pooled)  # [B, hidden_dim // 2]

        level = self.level_head(features)  # [B, 1]
        #confidence = self.confidence_head(features)  # [B, 1]

        return level  # [B, 1]


class SupportHead(nn.Module):
    """
    Predicts potential support level for bear runs.

    Output: [B, 2]
        [:, 0]: support distance (% below current price, positive value)
        [:, 1]: confidence score [0, 1]
    """
    def __init__(self, d_model, hidden_dim=128, dropout=0.2, pooling="AttentionPooling"):
        super().__init__()
        self.pooling = build_pooling(pooling, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Support level (positive percentage representing downward distance)
        self.level_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive values
        )

        # # Confidence
        # self.confidence_head = nn.Sequential(
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, embedding):
        """
        Args:
            embedding: [B, C, L, d_model]
        Returns:
            support: [B, 2] - (level_%, confidence)
        """
        pooled = self.pooling(embedding)  # [B, d_model]
        features = self.mlp(pooled)  # [B, hidden_dim // 2]

        level = self.level_head(features)  # [B, 1]
        #confidence = self.confidence_head(features)  # [B, 1]

        return level  # [B, 1]


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head for comprehensive trading signal generation.

    Outputs:
        - reversal_prob: [B, 1]
        - trend: [B, pred_len] - predicted future prices
        - resistance: [B, 2] (level, confidence)
        - support: [B, 2] (level, confidence)
    """
    def __init__(self, heads, d_model, pred_len=24, hidden_dim=128, dropout=0.2, pooling="AttentionPooling", n_heads=4, d_layers=2):
        super().__init__()
        self.heads = []
        for head in heads:
            self.heads.append({'type': head, 'object': build_task_head(head, d_model, pred_len, hidden_dim, dropout, pooling, n_heads, d_layers)})



    def forward(self, embedding):
        """
        Args:
            embedding: [B, C, L, d_model]
        Returns:
            dict with keys: reversal, trend, resistance, support
        """
        output = {}
        for head in self.heads:
            output[head['type']] = head['object'](embedding)
        return output
        # {
        #     "reversal": self.reversal_head(embedding),  # [B, 1]
        #     "trend": self.trend_head(embedding),  # [B, pred_len]
        #     "resistance": self.resistance_head(embedding),  # [B, 2]
        #     "support": self.support_head(embedding),  # [B, 2]
        # }


def build_task_head(task_type, d_model, pred_len=24, hidden_dim=128, dropout=0.2, pooling="AttentionPooling", n_heads=4, d_layers=2):
    """
    Factory function to build task-specific heads.

    Args:
        task_type: One of "reversal", "trend", "resistance", "support", "multitask"
        d_model: Model dimension
        pred_len: Prediction length for trend head
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        pooling: Pooling strategy name
        n_heads: Number of attention heads for decoder
        d_layers: Number of decoder layers

    Returns:
        Task head module
    """
    heads = {
        "reversal": lambda: ReversalHead(d_model, hidden_dim, dropout, pooling),
        "trend": lambda: LongTermTrendHead(d_model, pred_len, hidden_dim, dropout, n_heads, d_layers),
        "resistance": lambda: ResistanceHead(d_model, hidden_dim, dropout, pooling),
        "support": lambda: SupportHead(d_model, hidden_dim, dropout, pooling),
        "multitask": lambda: MultiTaskHead(d_model, pred_len, hidden_dim, dropout, pooling, n_heads, d_layers),
    }

    if task_type not in heads:
        available = ", ".join(heads.keys())
        raise ValueError(f"Unknown task type '{task_type}'. Available: {available}")

    return heads[task_type]()
