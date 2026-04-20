import ast
import os
import logging
import numpy as np

# from config.googleCloud import download_blob, get_latest_blob_path
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import Dataset

from controller.config_manager import get_config_manager
from models.functions import power_distance


def _extract_time_features(dates):
    """ convert timestamp to periodic time feature  (scale to [-0.5, 0.5])"""
    df_stamp = pd.DataFrame()
    df_stamp['month'] = dates.dt.month / 12 - 0.5
    df_stamp['day'] = dates.dt.day / 31 - 0.5
    df_stamp['weekday'] = dates.dt.weekday / 7 - 0.5
    df_stamp['hour'] = dates.dt.hour / 24 - 0.5
    return df_stamp.values


def preprocess_dataframe(device=None):
    config = get_config_manager()
    dataset_filetype = config.get("dataset_filetype").value
    dataset_file_path = config.get("train_dataset_path").value
    if device is None:
        device = config.get("device").value or "cuda"

    df = None
    filetype = dataset_filetype.lstrip(".")
    if filetype == "csv":
        df = pd.read_csv(dataset_file_path)
    elif filetype == "parquet":
        df = pd.read_parquet(dataset_file_path)
    elif filetype == "feather":
        df = pd.read_feather(dataset_file_path)
    else:
        raise ValueError(f"Unsupported file type: {dataset_filetype}. Use csv, parquet, or feather.")

    if get_config_manager().get("use_last_n_num").value:
        df = df.tail(get_config_manager().get("use_last_n_num").value).reset_index(drop=True)

    time_features = np.zeros((len(df), 4))
    if "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_features = _extract_time_features(df['timestamp'])
        df = df.drop(columns=["timestamp"])

    # order book data
    for col in ["bid_volume", "ask_volume"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            prefix = col.split('_')[0] + "_"
            # expend data
            expanded = pd.DataFrame(df[col].tolist(), index=df.index).add_prefix(prefix)
            # sum bid/ask volume
            df[f'{prefix}sum'] = df[col].apply(sum)
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    # 4. amplify variance
    # calculate bps
    # fluctuation to open price
    df['open_ret'] = df['open'].pct_change().fillna(0) * 1000
    df['high_gap'] = (df['high'] / df['open'] - 1) * 1000
    df['low_gap'] = (df['low'] / df['open'] - 1) * 1000
    df['close_gap'] = (df['close'] / df['open'] - 1) * 1000

    # bid/ask imbalance  (important，range [-1, 1])
    df['imbalance'] = (df['bid_sum'] - df['ask_sum']) / (df['bid_sum'] + df['ask_sum'] + 1e-8)

    # 5.Z-score
    norm_cols = df.filter(regex='volume|sum|delta|bid_|ask_').columns
    if get_config_manager().get("normalize").value:
        df[norm_cols] = (df[norm_cols] - df[norm_cols].mean()) / (df[norm_cols].std() + 1e-8)

    # 6. integrate features

    feature_cols = ['open_ret', 'high_gap', 'low_gap', 'close_gap', 'imbalance', 'volume', 'delta']
    # 加上展开的 10 档盘口
    feature_cols += [c for c in df.columns if 'bid_' in c or 'ask_' in c]

    # 转换为 Numpy 再合并时间特征
    data_values = df[feature_cols].astype("float32").values
    final_data = np.column_stack((data_values, time_features))

    # 保存为 Tensor
    data = torch.tensor(final_data, dtype=torch.float32, device=device)

    # 这里的 close 价格需要单独保存一份，用于计算未来的 PP (利润)
    # 注意：这里的 close 使用的是原始价格或对数价格，不能是 Z-score 后的

    # look ahead

    close_col = torch.tensor(df['close_gap'].values, dtype=torch.float32)
    # padding
    close_col = F.pad(close_col, (0, get_config_manager().get("num_look_ahead").value))

    return data, close_col  # close_col not normalized


class OHLCDataset(Dataset):
    def __init__(self, data, close_col, device='cuda'):
        # Load from config manager
        config = get_config_manager()
        seq_len = config.get("seq_len").value
        sliding_step = config.get("sliding_step").value
        pred_len_param = config.get("num_look_ahead")
        pred_len = pred_len_param.value if pred_len_param else 1
        use_last_n = config.get("use_last_n").value
        normalize = config.get("normalize").value
        num_look_ahead = config.get("num_look_ahead").value
        sliding_window = config.get("sliding_window").value

        self.data = data
        self.close_col = close_col
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.pred_len = pred_len
        self.use_last_n = use_last_n
        self.normalize = normalize
        self.sliding_window = sliding_window
        self.num_look_ahead = num_look_ahead
        self.device = device

        returns = self.close_col.diff()
        self.volatility_50 = returns.rolling(50, min_periods=1).std()

        look_ahead_k = self.close_col.unfold(0, self.num_look_ahead + 1, 1)
        look_ahead_k = look_ahead_k[:, 1:]  # remove itself, [N, K]
        # generate dataset [L, C]
        dataset_temp = []
        reference_k_temp = []
        if sliding_window:
            # 计算可用样本量
            self.num_samples = (len(self.data) - self.seq_len - self.num_look_ahead) // self.sliding_step + 1
            for i in range(self.num_samples):
                start = i * self.sliding_step
                end = start + self.seq_len
                dataset_temp.append(self.data[start:end])
                reference_k_temp.append(look_ahead_k[start:end])

        else:
            # 计算可用样本量
            self.num_samples = len(self.data) - self.seq_len - self.num_look_ahead + 1

            for i in range(self.num_samples):
                start = i * self.seq_len
                end = start + self.seq_len
                dataset_temp.append(self.data[start: end])
                reference_k_temp.append(look_ahead_k[start: end])

        self.dataset = torch.stack(dataset_temp)
        self.reference_k = torch.stack(reference_k_temp)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.dataset[idx], self.reference_k[idx], self.volatility_50[idx]  # [L, C], [K], 1


