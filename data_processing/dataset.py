import ast
import os
import logging

import numpy as np

from config.googleCloud import download_blob, get_latest_blob_path

import pandas as pd
import torch
from torch.utils.data import Dataset
from models.functions import power_distance


class OHLCDataset(Dataset):
    def __init__(self,
                 csv_path,
                 seq_len=64,
                 sliding_step=1,
                 pred_len=1,
                 use_last_n=None,  # use the last n items only
                 normalize=False,
                 sliding_window=False):

        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.pred_len = pred_len
        self.use_last_n = use_last_n
        self.normalize = normalize
        self.sliding_window = sliding_window

        df = pd.read_csv(csv_path)

        # drop timestamp column

        time_feature = None
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_feature = self._extract_time_features(df['timestamp'])
            df = df.drop(columns=["timestamp"])

        # replace timestamp to distance
        # df.insert(0, 'distance', power_distance(np.arange(len(df))))

        if use_last_n is not None:
            df = df.tail(use_last_n).reset_index(drop=True)

        # convert str to list
        for col in ["bid_volume", "ask_volume"]:
            if col in df.columns and df[col].dtype == str:
                df[col] = df[col].apply(ast.literal_eval)
                col_names = pd.DataFrame(df[col].tolist()).add_prefix('bid_')
                df = pd.concat([df.drop(col), col_names], axis=1)

        df.filter(like="ask_")[:] = -df.filter(like="ask_")

        # to float32
        data = df.astype("float32").values

        # normalization
        if normalize:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-8
            data = (data - mean) / std
        data = np.column_stack((data, time_feature))

        self.num_samples = (data.shape[0] - self.seq_len - self.pred_len) // self.sliding_step + 1
        self.data = torch.tensor(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self.indices = torch.arange(self.data.size(0), device=self.data.size(0))
        if self.sliding_window:
            start = idx * self.sliding_step
            end = start + self.seq_len
            return self.data[start:end], self.indices[start:end]
        else:
            return self.data[idx: idx + self.seq_len], self.indices[idx: idx + self.seq_len + self]

    def _extract_time_features(self, dates):
        """ convert timestamp to periodic time feature  (scale to [-0.5, 0.5])"""
        df_stamp = pd.DataFrame()
        df_stamp['month'] = dates.dt.month / 12 - 0.5
        df_stamp['day'] = dates.dt.day / 31 - 0.5
        df_stamp['weekday'] = dates.dt.weekday / 7 - 0.5
        df_stamp['hour'] = dates.dt.hour / 24 - 0.5
        return df_stamp.values


# download file
aggregated_trades_file_path = '../data/Binance_BTC_USDT_USDT_3m.csv'

aggregated_trades_bucket_path = 'aggTrades/BTCUSDT'
aggregated_trades_bucket_name = 'binance-histrial-files'

if not os.path.exists(aggregated_trades_file_path):
    latest_path = get_latest_blob_path(aggregated_trades_bucket_name,
                                       f"{aggregated_trades_bucket_path}/BTCUSDT-aggTrades-")
    download_blob(aggregated_trades_bucket_name, latest_path, aggregated_trades_file_path)

aggregated_trades = pd.read_parquet(aggregated_trades_file_path)

logging.info(f"read aggregated trades")

logging.info(f"Columns: {aggregated_trades.columns.to_list()}")

logging.info(f"Shape: {aggregated_trades.shape}")

logging.info(aggregated_trades.sample(2))

# aggregated_trades = aggregated_trades.to_numpy()
