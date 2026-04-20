"""
Backtesting strategy: delta-based reversal trading with TP/SL,
commission fees, and multi-threshold sweep analysis.
"""
import numpy as np

from controller.config_manager import get_config_manager


def backtesting(confidence, reference_k,
                tp=None, sl=None, confidence_threshold=None,
                margin=None, commission_rate=None):
    """
    Backtest reversal signals against future price movements.

    Args:
        confidence: [N] or [N,1] - model confidence scores
        reference_k: [N, K] - future price changes in bps per step
        tp: take-profit threshold (bps, positive). None = read from config.
        sl: stop-loss threshold (bps, negative). None = read from config.
        confidence_threshold: min |confidence| to trade. None = read from config.
        margin: position margin (bps). None = read from config.
        commission_rate: per-trade commission rate. None = read from config.

    Returns:
        list of tuples: (sample_index, exit_step, gross_profit, net_profit)
    """
    cm = get_config_manager()
    if tp is None:
        tp = float(cm.get("take_profit").value)
    if sl is None:
        sl = float(cm.get("take_loss").value)
    if confidence_threshold is None:
        confidence_threshold = float(cm.get("confidence_threshold").value)
    if margin is None:
        margin = float(cm.get("margin").value)
    if commission_rate is None:
        cr_param = cm.get("commission_rate")
        commission_rate = float(cr_param.value) if cr_param else 0.0

    confidence = np.asarray(confidence).flatten()
    reference_k = np.asarray(reference_k)

    # Cumulative price trajectory per sample
    cum_prices = np.cumsum(reference_k, axis=1)  # [N, K]
    N, K = cum_prices.shape

    # Commission cost per round-trip trade
    commission_cost = 2 * commission_rate * margin

    profits = []
    for i in range(N):
        if abs(confidence[i]) < confidence_threshold:
            continue

        path = cum_prices[i]  # [K]

        # Find first step hitting TP or SL
        tp_steps = np.where(path >= tp)[0]
        sl_steps = np.where(path <= sl)[0]

        first_tp = tp_steps[0] if len(tp_steps) > 0 else K
        first_sl = sl_steps[0] if len(sl_steps) > 0 else K

        if first_tp <= first_sl:
            exit_step = first_tp
            gross = float(path[exit_step]) if exit_step < K else float(path[-1])
        elif first_sl < first_tp:
            exit_step = first_sl
            gross = float(path[exit_step]) if exit_step < K else float(path[-1])
        else:
            # Neither hit — use last price
            exit_step = K - 1
            gross = float(path[-1])

        net = gross - commission_cost
        profits.append((i, int(exit_step), gross, net))

    return profits


def threshold_sweep(confidence, reference_k,
                    threshold_values=None, commission_rate=None, **kwargs):
    """
    Run backtesting at multiple confidence thresholds.

    Args:
        confidence: [N] or [N,1]
        reference_k: [N, K]
        threshold_values: list of floats. None = read from config.
        commission_rate: per-trade rate. None = read from config.
        **kwargs: passed through to backtesting()

    Returns:
        list of dicts with keys:
            threshold, total_gross_profit, total_net_profit, win_rate, num_trades
    """
    if threshold_values is None:
        cm = get_config_manager()
        sweep_param = cm.get("threshold_sweep_values")
        if sweep_param and sweep_param.value:
            threshold_values = [float(x.strip()) for x in sweep_param.value.split(",")]
        else:
            threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for thresh in sorted(threshold_values):
        trades = backtesting(
            confidence, reference_k,
            confidence_threshold=thresh,
            commission_rate=commission_rate,
            **kwargs,
        )
        if trades:
            gross_vals = [t[2] for t in trades]
            net_vals = [t[3] for t in trades]
            winning = sum(1 for v in net_vals if v > 0)
            results.append({
                "threshold": thresh,
                "total_gross_profit": sum(gross_vals),
                "total_net_profit": sum(net_vals),
                "win_rate": winning / len(trades),
                "num_trades": len(trades),
            })
        else:
            results.append({
                "threshold": thresh,
                "total_gross_profit": 0.0,
                "total_net_profit": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
            })

    return results
