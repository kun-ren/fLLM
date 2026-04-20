import torch


def power_distance(x):
    """
    Distance function: y = x^(3/4) + (x/50)^(3/2)

    Args:
        x: input value, can be a float, int, or numpy array / torch tensor
    Returns:
        y: computed distance
    """
    return x ** (3 / 4) + (x / 50) ** (3 / 2)


def amplified_atanh(x, target_range=5.0, eps=1e-4):
    """
    将 [-1, 1] 的输入放大到 [-target_range, target_range]
    """
    # 1. 安全缩放，防止输入严格等于 1 或 -1 导致 Inf
    x_safe = x * (1 - eps)

    # 2. 计算原始 atanh 值
    # atanh(0.9999) 约等于 4.95
    raw_atanh = torch.atanh(x_safe)

    # 3. 动态计算缩放因子 k
    # 使得当输入为 1 时，输出刚好是 target_range
    max_val = torch.atanh(torch.tensor(1.0 - eps))
    k = target_range / max_val

    return k * raw_atanh
