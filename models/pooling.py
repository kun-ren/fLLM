"""
Pooling strategies for temporal sequence embeddings.
All pooling classes take input [B, C, T, d_model] and return [B, d_model].
"""
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Attention-based pooling with learnable channel and temporal weights.
    """
    def __init__(self, d_model, time_decay='linear'):
        super().__init__()
        self.time_decay = time_decay
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # [[B, C, T, d]]
        x_out = []
        for x_layer in x:
            B, C, T, d = x_layer.shape

            var_score = self.proj(x_layer)  # to [B, C, T, 1]

            # weighted pooling
            var_weights = torch.softmax(var_score, dim=1)
            x_var = torch.sum(x_layer * var_weights, dim=1)  # [B, T, d]

            # temporal sequence weight
            time_weights = self.get_time_weights(T, x_layer.device)  # [T]
            x_out.append((x_var * time_weights.view(1, T, 1)).sum(dim=1))  # [B, d]

        return torch.cat(x_out, dim=0).to('cuda')

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
            raise ValueError(f"Unsupported time_decay type: {self.time_decay}")

        weights = weights / weights.sum()
        return weights  # [T]


class LastStepPooling(nn.Module):
    """
    Pool by taking the last time step with learnable channel weights.
    """
    def __init__(self, d_model):
        super().__init__()
        self.channel_weight = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x):  # [B, C, T, d]
        _, C, _, _ = x.shape
        last_step = x[:, :, -1, :]  # [B, C, d]
        weights = self.channel_weight.expand(1, C, 1)
        weights = torch.sigmoid(weights)
        weighted = last_step * weights
        return weighted.sum(dim=1)  # [B, d]


class MeanPooling(nn.Module):
    """
    Simple mean pooling across channels and time.
    """
    def __init__(self, d_model):
        super().__init__()

    def forward(self, x):  # [B, C, T, d]
        return x.mean(dim=(1, 2))  # [B, d]


class MaxPooling(nn.Module):
    """
    Max pooling across channels and time.
    """
    def __init__(self, d_model):
        super().__init__()

    def forward(self, x):  # [B, C, T, d]
        return x.amax(dim=(1, 2))  # [B, d]


# Registry mapping pooling names to classes
POOLING_REGISTRY = {
    "AttentionPooling": AttentionPooling,
    "LastStepPooling": LastStepPooling,
    "MeanPooling": MeanPooling,
    "MaxPooling": MaxPooling,
}


def get_pooling_class(name):
    """
    Get pooling class by name.

    Args:
        name: Pooling type name (case-sensitive)

    Returns:
        Pooling class

    Raises:
        ValueError: If pooling name is not recognized
    """
    try:
        return POOLING_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(POOLING_REGISTRY.keys())
        raise ValueError(f"Unsupported pooling type '{name}'. Available types: {available}") from exc


def build_pooling(name, d_model, **kwargs):
    """
    Build a pooling module by name.

    Args:
        name: Pooling type name
        d_model: Model dimension
        **kwargs: Additional arguments passed to pooling constructor

    Returns:
        Instantiated pooling module
    """
    pooling_class = get_pooling_class(name)
    return pooling_class(d_model=d_model, **kwargs)
