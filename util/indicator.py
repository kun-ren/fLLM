import pandas as pd


def calculate_btc_atr(df: pd.DataFrame, n: int = 14, high_col='high', low_col='low') -> pd.DataFrame:
    """
    专为 7x24 小时无跳空市场设计的 ATR 计算

    ATRt = ( ATRt-1 * (N-1) + TRt) / N
    """
    # 1. 极简 TR：直接用最高价减去最低价
    df['TR'] = df[high_col] - df[low_col]

    # 2. 计算 ATR：使用 Wilder's Smoothing (等价于 alpha = 1/n 的 EMA)
    # adjust=False 确保严格按照 Wilder 的递推公式进行计算
    df['ATR'] = df['TR'].ewm(alpha=1 / n, adjust=False).mean()

    return df

