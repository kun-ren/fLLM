import torch
import pandas as pd
import numpy as np
import ast

from torch.utils.data import DataLoader

from config.env_loader import get_config
from data_processing.dataset import OHLCDataset, preprocess_dataframe
from models.crossformer import CrossformerEncoderTimeStep, EmbeddingHead
from strategies.delta_reverse import backtesting


class TrendPredictor:
    """
    Predict trend reversal using trained Crossformer model.
    Output: confidence score in [-1, 1]
        -1: downward trend
        +1: upward trend
    """

    def __init__(self, encoder_path, head_path, device='cuda', pooling='AttentionPooling'):
        """
        Args:
            encoder_path: path to saved encoder weights
            head_path: path to saved head weights
            device: 'cuda' or 'cpu'
            pooling: pooling type to use (default: 'AttentionPooling')
        """
        # Load config from environment
        self.seq_len = get_config().seq_len
        self.device = device

        # Load model architecture (must match training config)
        self.encoder = CrossformerEncoderTimeStep(
            input_dim=31,  # 7 base features + 20 order book + 4 time features
            d_model=64,
            n_heads=8,
            n_layers=5
        ).to(device)

        self.head = EmbeddingHead(
            d_model=64,
            hidden_dim=128,
            output_dim=1,
            pooling=pooling,
        ).to(device)

        # Load trained weights
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.head.load_state_dict(torch.load(head_path, map_location=device))

        self.encoder.eval()
        self.head.eval()

    def _preprocess_dataframe(self, df, normalize=True):
        """
        Preprocess raw OHLC DataFrame into model input format.
        Matches the preprocessing in OHLCDataset.

        Args:
            df: DataFrame with columns: open, high, low, close, volume, delta,
                bid_volume, ask_volume, timestamp (optional)
            normalize: whether to apply Z-score normalization

        Returns:
            torch.Tensor of shape [L, C] where L=seq_len, C=num_features
        """
        df = df.copy()

        # Extract time features if timestamp exists
        time_features = np.zeros((len(df), 4))
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_features = self._extract_time_features(df['timestamp'])
            df = df.drop(columns=["timestamp"])

        # Process order book data
        for col in ["bid_volume", "ask_volume"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                prefix = col.split('_')[0] + "_"
                expanded = pd.DataFrame(df[col].tolist(), index=df.index).add_prefix(prefix)
                df[f'{prefix}sum'] = df[col].apply(sum)
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

        # Calculate bps features
        df['open_ret'] = df['open'].pct_change().fillna(0) * 1000
        df['high_gap'] = (df['high'] / df['open'] - 1) * 1000
        df['low_gap'] = (df['low'] / df['open'] - 1) * 1000
        df['close_gap'] = (df['close'] / df['open'] - 1) * 1000

        # Bid/ask imbalance
        df['imbalance'] = (df['bid_sum'] - df['ask_sum']) / (df['bid_sum'] + df['ask_sum'] + 1e-8)

        # Z-score normalization
        norm_cols = df.filter(regex='volume|sum|delta|bid_|ask_').columns
        if normalize:
            df[norm_cols] = (df[norm_cols] - df[norm_cols].mean()) / (df[norm_cols].std() + 1e-8)

        # Select feature columns
        feature_cols = ['open_ret', 'high_gap', 'low_gap', 'close_gap', 'imbalance', 'volume', 'delta']
        feature_cols += [c for c in df.columns if 'bid_' in c or 'ask_' in c]

        # Combine with time features
        data_values = df[feature_cols].astype("float32").values
        final_data = np.column_stack((data_values, time_features))

        return torch.tensor(final_data, dtype=torch.float32, device=self.device)

    def _extract_time_features(self, dates):
        """Convert timestamp to periodic time features (scale to [-0.5, 0.5])"""
        df_stamp = pd.DataFrame()
        df_stamp['month'] = dates.dt.month / 12 - 0.5
        df_stamp['day'] = dates.dt.day / 31 - 0.5
        df_stamp['weekday'] = dates.dt.weekday / 7 - 0.5
        df_stamp['hour'] = dates.dt.hour / 24 - 0.5
        return df_stamp.values

    def predict_from_array(self, ohlc_array):
        """
        Predict trend reversal from plain numpy array or list.

        Args:
            ohlc_array: array-like of shape [seq_len, num_raw_features]
                Expected columns: open, high, low, close, volume, delta,
                                 bid_volume (list), ask_volume (list), timestamp (optional)

        Returns:
            confidence: float in [-1, 1]
        """
        normalize = get_config().normalize
        # Convert to DataFrame for preprocessing
        columns = ['open', 'high', 'low', 'close', 'volume', 'delta', 'bid_volume', 'ask_volume']

        if isinstance(ohlc_array, np.ndarray):
            if ohlc_array.shape[1] == 9:
                columns.append('timestamp')
            df = pd.DataFrame(ohlc_array, columns=columns[:ohlc_array.shape[1]])
        else:
            df = pd.DataFrame(ohlc_array, columns=columns)

        # Take last seq_len rows
        df = df.tail(self.seq_len).reset_index(drop=True)

        # Preprocess
        data = self._preprocess_dataframe(df, normalize=normalize)  # [L, C]
        data = data.unsqueeze(0)  # [1, L, C]

        return self._predict(data)

    def predict_from_dataframe(self, df):
        """
        Predict trend reversal from pandas DataFrame.
        Uses the last seq_len rows as input.

        Args:
            df: DataFrame with columns matching OHLC format

        Returns:
            confidence: float in [-1, 1]
        """
        normalize = get_config().normalize
        df = df.tail(self.seq_len).reset_index(drop=True)

        data = self._preprocess_dataframe(df, normalize=normalize)  # [L, C]
        data = data.unsqueeze(0)  # [1, L, C]

        return self._predict(data)

    @torch.no_grad()
    def _predict(self, data):
        """
        Internal prediction method.

        Args:
            data: torch.Tensor of shape [NUM, L, C]

        Returns:
            confidence: [NUM, 1]
        """
        # Forward pass
        embedding = self.encoder(data)
        confidence = self.head(embedding)

        print(f"Confidence shape: {confidence.shape}")

        return confidence

    def interpret_prediction(self, confidence, threshold=0.3):
        """
        Interpret the confidence score into human-readable trend signal.

        Args:
            confidence: float in [-1, 1]
            threshold: minimum absolute value to consider as strong signal

        Returns:
            dict with 'signal', 'strength', and 'confidence'
        """
        abs_conf = abs(confidence)

        if abs_conf < threshold:
            signal = 'neutral'
            strength = 'weak'
        elif abs_conf < 0.6:
            signal = 'upward' if confidence > 0 else 'downward'
            strength = 'moderate'
        else:
            signal = 'upward' if confidence > 0 else 'downward'
            strength = 'strong'

        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'abs_confidence': abs_conf
        }

    def backtest(self, backtesting_csv_path=None, sliding_step=None):
        """
        Backtest model performance on historical data.
        Metric: profit = confidence * sum(future_price_changes[:num_look_ahead])

        Args:
            csv_path: path to CSV with OHLC data (defaults to env config)
            sliding_step: step size for sliding window (defaults to env config)

        Returns:
            dict with backtest results:
                - total_profit: sum of (confidence * actual_profit)
                - avg_profit_per_trade: average profit per prediction
                - num_predictions: total number of predictions made
                - predictions: list of individual prediction results
        """
        if backtesting_csv_path is None:
            backtesting_csv_path = get_config().backtesting_csv_path
        if sliding_step is None:
            sliding_step = get_config().sliding_step

        num_look_ahead = get_config().num_of_look_ahead
        normalize = get_config().normalize

        data, close_col = preprocess_dataframe(backtesting_csv_path)

        dataset = OHLCDataset(data, close_col)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = []
        references = []

        for index, (batch_data, reference_k) in enumerate(loader):
            confidences = self._predict(batch_data)
            predictions.extend(confidences.detach().cpu().numpy())
            references.extend(reference_k.detach().cpu().numpy())

        profits = backtesting(predictions, references)


# Example usage
if __name__ == '__main__':
    # Initialize predictor with trained model weights
    predictor = TrendPredictor(
        encoder_path='checkpoints/encoder.pth',
        head_path='checkpoints/head.pth',
        device='cuda'
    )

    # ===== Single Prediction =====
    confidence = predictor.predict_from_csv()
    result = predictor.interpret_prediction(confidence)

    print(f"Trend Prediction:")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Signal: {result['signal']}")
    print(f"  Strength: {result['strength']}")

    # ===== Backtesting =====
    print("\n" + "="*50)
    print("Running Backtest (Simple)...")
    print("="*50)

    backtest_results = predictor.backtest(sliding_step=10)  # Test every 10 candles for speed

    print(f"\nBacktest Results:")
    print(f"  Total Predictions: {backtest_results['num_predictions']}")
    print(f"  Total Profit: {backtest_results['total_profit_bps']:.2f} bps")
    print(f"  Avg Profit per Trade: {backtest_results['avg_profit_per_trade_bps']:.4f} bps")

    # ===== Backtesting with Stop-Loss =====
    print("\n" + "="*50)
    print("Running Backtest (With Stop-Loss)...")
    print("="*50)

    backtest_sl_results = predictor.backtest_with_stop_loss(
        stop_loss_bps=10.0,
        sliding_step=10
    )

    print(f"\nBacktest Results (Stop-Loss):")
    print(f"  Total Predictions: {backtest_sl_results['num_predictions']}")
    print(f"  Total //: {backtest_sl_results['total_profit_bps']:.2f} bps")
    print(f"  Avg Profit per Trade: {backtest_sl_results['avg_profit_per_trade_bps']:.4f} bps")
    print(f"  Total Stop-Loss Hits: {backtest_sl_results['total_stop_loss_hits']}")

    # Show sample predictions
    print(f"\nSample Predictions (first 5):")
    for pred in backtest_sl_results['predictions'][:5]:
        print(f"  Index {pred['index']}: confidence={pred['confidence']:.3f}, "
              f"net_profit={pred['net_profit_bps']:.2f} bps, "
              f"weighted_profit={pred['weighted_profit_bps']:.2f} bps")
