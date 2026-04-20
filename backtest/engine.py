"""
Backtesting engine for trading strategy evaluation.

Loads trained model, runs inference on test data, executes trading strategy
with TP/SL logic, and calculates comprehensive performance metrics.
"""
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from data_processing.dataset import OHLCDataset, preprocess_dataframe
from models.inference import ModelInference
from controller.config_manager import get_config_manager

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    pnl_bps: float
    pnl_pct: float
    exit_reason: str  # 'tp', 'sl', 'timeout'
    confidence: float
    hold_periods: int


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # PnL metrics
    total_pnl_bps: float
    total_pnl_pct: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float

    # Risk metrics
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade details
    avg_hold_periods: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Exit reasons
    tp_exits: int
    sl_exits: int
    timeout_exits: int

    # Time series data
    equity_curve: List[float]
    trades: List[Dict]

    # Strategy parameters
    confidence_threshold: float
    take_profit_bps: float
    stop_loss_bps: float
    commission_rate: float


class BacktestEngine:
    """
    Backtesting engine for reversal trading strategy.

    Strategy:
    1. Model predicts reversal confidence [-1, 1]
    2. If |confidence| > threshold, enter trade in predicted direction
    3. Exit on TP/SL or timeout (max_hold_periods)
    4. Track all trades and compute performance metrics
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.6,
        take_profit_bps: float = 5.0,
        stop_loss_bps: float = 10.0,
        max_hold_periods: int = 20,
        commission_rate: float = 0.0004,
        device: str = 'cuda'
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps
        self.max_hold_periods = max_hold_periods
        self.commission_rate = commission_rate
        self.device = device

        self.model = None

    def load_model(self):
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.model = ModelInference(str(self.model_path), device=self.device)
        logger.info(f"Model loaded from {self.model_path}")

    def run_inference(self, dataset: OHLCDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference on dataset.

        Returns:
            Tuple of (reversal_signals, support_levels, resistance_levels, future_prices)
        """
        return self.model.predict_dataset(dataset)

    def execute_strategy(
        self,
        reversal_signals: np.ndarray,
        support_levels: np.ndarray,
        resistance_levels: np.ndarray,
        future_prices: np.ndarray
    ) -> List[Trade]:
        """
        Execute trading strategy based on predictions.

        Args:
            reversal_signals: [N] reversal confidence scores
            support_levels: [N] predicted support levels
            resistance_levels: [N] predicted resistance levels
            future_prices: [N, L, K] future price changes in bps

        Returns:
            List of Trade objects
        """
        trades = []
        N = len(reversal_signals)

        for i in range(N):
            confidence = reversal_signals[i]

            # Check if signal is strong enough
            if abs(confidence) < self.confidence_threshold:
                continue

            # Determine direction
            direction = 'long' if confidence > 0 else 'short'

            # Get future price trajectory for this sample
            future_prices_sample = future_prices[i, -1, :]  # [K] - last timestep's look-ahead
            cumsum = np.cumsum(future_prices_sample)

            # Simulate trade execution
            entry_price = 0.0  # Entry at current price (relative)
            exit_idx = None
            exit_price = None
            exit_reason = None

            for t in range(min(len(cumsum), self.max_hold_periods)):
                price_change = cumsum[t]

                if direction == 'long':
                    # Long: profit when price goes up
                    if price_change >= self.take_profit_bps:
                        exit_idx = t
                        exit_price = self.take_profit_bps
                        exit_reason = 'tp'
                        break
                    elif price_change <= -self.stop_loss_bps:
                        exit_idx = t
                        exit_price = -self.stop_loss_bps
                        exit_reason = 'sl'
                        break
                else:  # short
                    # Short: profit when price goes down
                    if price_change <= -self.take_profit_bps:
                        exit_idx = t
                        exit_price = -self.take_profit_bps
                        exit_reason = 'tp'
                        break
                    elif price_change >= self.stop_loss_bps:
                        exit_idx = t
                        exit_price = self.stop_loss_bps
                        exit_reason = 'sl'
                        break

            # Timeout exit
            if exit_idx is None:
                exit_idx = min(len(cumsum) - 1, self.max_hold_periods - 1)
                exit_price = cumsum[exit_idx]
                exit_reason = 'timeout'

            # Calculate PnL
            if direction == 'long':
                pnl_bps = exit_price - entry_price
            else:  # short
                pnl_bps = entry_price - exit_price

            # Apply commission
            pnl_bps -= 2 * self.commission_rate * 10000  # Entry + exit commission
            pnl_pct = pnl_bps / 10000  # Convert bps to percentage

            trade = Trade(
                entry_idx=i,
                exit_idx=i + exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                pnl_bps=pnl_bps,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                confidence=abs(confidence),
                hold_periods=exit_idx + 1
            )

            trades.append(trade)

        return trades

    def calculate_metrics(self, trades: List[Trade]) -> BacktestResult:
        """Calculate comprehensive performance metrics from trades."""
        if not trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                total_pnl_bps=0.0, total_pnl_pct=0.0, avg_win_bps=0.0, avg_loss_bps=0.0,
                profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
                sortino_ratio=0.0, calmar_ratio=0.0, avg_hold_periods=0.0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                tp_exits=0, sl_exits=0, timeout_exits=0,
                equity_curve=[], trades=[],
                confidence_threshold=self.confidence_threshold,
                take_profit_bps=self.take_profit_bps,
                stop_loss_bps=self.stop_loss_bps,
                commission_rate=self.commission_rate
            )

        # Basic statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl_bps > 0)
        losing_trades = sum(1 for t in trades if t.pnl_bps < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # PnL metrics
        pnls_bps = [t.pnl_bps for t in trades]
        total_pnl_bps = sum(pnls_bps)
        total_pnl_pct = sum(t.pnl_pct for t in trades)

        wins = [t.pnl_bps for t in trades if t.pnl_bps > 0]
        losses = [t.pnl_bps for t in trades if t.pnl_bps < 0]

        avg_win_bps = np.mean(wins) if wins else 0.0
        avg_loss_bps = np.mean(losses) if losses else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity curve
        equity_curve = [0.0]
        for pnl in pnls_bps:
            equity_curve.append(equity_curve[-1] + pnl)

        # Drawdown
        peak = equity_curve[0]
        max_drawdown = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / 10000  # Convert to percentage
            max_drawdown = max(max_drawdown, drawdown)

        # Risk-adjusted returns
        returns = np.array(pnls_bps) / 10000  # Convert to percentage
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1e-6

        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-6
        sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

        # Calmar ratio
        calmar_ratio = (total_pnl_pct / max_drawdown) if max_drawdown > 0 else 0.0

        # Trade details
        avg_hold_periods = np.mean([t.hold_periods for t in trades])

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl_bps > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        # Exit reasons
        tp_exits = sum(1 for t in trades if t.exit_reason == 'tp')
        sl_exits = sum(1 for t in trades if t.exit_reason == 'sl')
        timeout_exits = sum(1 for t in trades if t.exit_reason == 'timeout')

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_bps=total_pnl_bps,
            total_pnl_pct=total_pnl_pct,
            avg_win_bps=avg_win_bps,
            avg_loss_bps=avg_loss_bps,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_hold_periods=avg_hold_periods,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            tp_exits=tp_exits,
            sl_exits=sl_exits,
            timeout_exits=timeout_exits,
            equity_curve=equity_curve,
            trades=[asdict(t) for t in trades],
            confidence_threshold=self.confidence_threshold,
            take_profit_bps=self.take_profit_bps,
            stop_loss_bps=self.stop_loss_bps,
            commission_rate=self.commission_rate
        )

    def run_backtest(self, test_data_path: Optional[str] = None) -> BacktestResult:
        """
        Run complete backtest pipeline.

        Args:
            test_data_path: Path to test dataset (uses config if None)

        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info("Starting backtest...")

        # Load model
        if self.model is None:
            self.load_model()

        # Load test data
        config = get_config_manager()
        if test_data_path:
            config.set("test_dataset_path", value=test_data_path)

        data, close_col = preprocess_dataframe()
        dataset = OHLCDataset(data, close_col, device=self.device)

        logger.info(f"Test dataset: {len(dataset)} samples")

        # Run inference
        logger.info("Running inference...")
        reversal_signals, support_levels, resistance_levels, future_prices = self.run_inference(dataset)

        # Execute strategy
        logger.info("Executing strategy...")
        trades = self.execute_strategy(reversal_signals, support_levels, resistance_levels, future_prices)

        logger.info(f"Executed {len(trades)} trades")

        # Calculate metrics
        result = self.calculate_metrics(trades)

        logger.info(f"Backtest complete. Win rate: {result.win_rate:.2%}, Total PnL: {result.total_pnl_bps:.2f} bps")

        return result


def run_backtest_from_config(model_path: str) -> Dict:
    """
    Run backtest using parameters from config_manager.

    Args:
        model_path: Path to trained model checkpoint

    Returns:
        Dictionary with backtest results
    """
    config = get_config_manager()

    engine = BacktestEngine(
        model_path=model_path,
        confidence_threshold=config.get("confidence_threshold").value,
        take_profit_bps=config.get("take_profit").value,
        stop_loss_bps=abs(config.get("take_loss").value),
        commission_rate=config.get("commission_rate").value,
        device=config.get("device").value
    )

    result = engine.run_backtest()
    return asdict(result)
