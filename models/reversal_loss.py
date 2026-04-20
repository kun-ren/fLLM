"""
Enhanced Reversal Loss with 3 components:
1. Reversal strength prediction (confidence score)
2. Resistance/Support price prediction (highest/lowest price in look-ahead window)
3. TP/SL penalty (stop-loss hit penalty)

Model outputs: [reversal_strength, highest_price, lowest_price]
- reversal_strength: [-1, 1] via tanh
- highest_price: predicted max price change in bps (unbounded)
- lowest_price: predicted min price change in bps (unbounded)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.functions import power_distance


class ReversalLoss(nn.Module):
    """
    3-component loss for reversal trading:
    1. Reversal strength MSE
    2. Resistance/Support price MSE
    3. TP/SL penalty

    Args:
        L: look-ahead window size
        tp_bps: take-profit cap in bps
        sl_bps: stop-loss cap in bps
        strength_weight: weight for reversal strength loss component
        price_weight: weight for resistance/support price loss component
        penalty_weight: weight for TP/SL penalty component
    """
    def __init__(self, L=20, strength_weight=1.0, price_weight=1.0, penalty_weight=0.5):
        super().__init__()
        self.L = L
        self.strength_weight = strength_weight
        self.price_weight = price_weight
        self.penalty_weight = penalty_weight

    def forward(self, out, close_prices, volatility):
        """
        Args:
            reversal_signal: [B, 1] - [reversal_strength]
            support: [B, 1] - future price support
            resistance: [B, 1] - future price resistance

        Returns:
            loss: scalar total loss
            metrics: dict of logging metrics
        """

        reversal_signal = out['reversal']
        resistance = out['resistance']
        support = out['support']
        B, L = close_prices.shape
        device = reversal_signal.device

        # Extract predictions
        pred_strength = reversal_signal.squeeze(-1)
        pred_highest = resistance.squeeze(-1)
        pred_lowest = support.squeeze(-1)



        # Cumulative price trajectory
        cumsum = torch.cumsum(close_prices, dim=1) / volatility  # [B, L]

        actual_highest = cumsum.max(dim=1)[0]
        actual_lowest = cumsum.min(dim=1)[0]

        price_prediction_loss = F.huber_loss(pred_highest/volatility, actual_highest, delta=10.0) + \
                                F.huber_loss(pred_lowest/volatility, actual_lowest, delta=10.0)

        # Price proximity weights
        index_dist = torch.arange(1, L + 1, device=device, dtype=torch.float32)  # [L]
        raw_weights = 1.0 / (power_distance(index_dist) + 1e-6)  # [L]
        weights = raw_weights / raw_weights.sum()  # [L]
        weights = weights.unsqueeze(0) # [1, L] broadcasting
        weighted_diff = (weights * cumsum).sum(dim=1, keepdim=True)  # [B, 1]
        target_signal = torch.tanh(weighted_diff)  # [B, 1]
        reversal_signal_loss = F.mse_loss(reversal_signal, target_signal)




        # ═══════════════════════════════════════════════════════════════════════
        # Component 1: Reversal Strength Target
        # ═══════════════════════════════════════════════════════════════════════
        # volatility = torch.std(future_price_changes, dim=1) + 1e-6  # [B]
        # max_prices, max_idx = cumsum.max(dim=1)  # [B]
        # min_prices, min_idx = cumsum.min(dim=1)  # [B]
        #
        # is_bullish = min_prices >= 0
        # is_bearish = max_prices <= 0
        #
        # # Time weight: earlier extreme = stronger signal
        # extreme_idx = torch.where(is_bullish, max_idx, min_idx).float()
        # time_weight = 1.0 / (power_distance(extreme_idx + 1.0) + 1e-6)
        #
        # # Price proximity weights
        # price_dist = cumsum.abs()
        # price_weights = 1.0 / (power_distance(price_dist + 1.0) + 1e-6)
        # price_weights = price_weights / (price_weights.sum(dim=1, keepdim=True) + 1e-6)
        #
        # # Directional movement
        # movement = torch.where(
        #     is_bullish.unsqueeze(1),
        #     F.relu(cumsum),
        #     F.relu(-cumsum),
        # )
        # weighted_mag = (movement * price_weights).sum(dim=1)
        #
        # # Reversal strength target
        # strength = time_weight * weighted_mag / volatility
        # target_strength = torch.tanh(strength / 5.0)
        # target_strength = torch.where(is_bearish, -target_strength, target_strength)
        # target_strength = torch.where(~is_bullish & ~is_bearish, torch.zeros_like(target_strength), target_strength)
        #
        # strength_loss = F.mse_loss(pred_strength, target_strength)
        #
        # # ═══════════════════════════════════════════════════════════════════════
        # # Component 2: Resistance/Support Price Targets
        # # ═══════════════════════════════════════════════════════════════════════
        # # Target highest/lowest prices in the look-ahead window (after TP/SL capping)
        # target_highest = max_prices  # [B]
        # target_lowest = min_prices   # [B]
        #
        # highest_loss = F.mse_loss(pred_highest, target_highest)
        # lowest_loss = F.mse_loss(pred_lowest, target_lowest)
        # price_loss = (highest_loss + lowest_loss) / 2.0
        #
        # # ═══════════════════════════════════════════════════════════════════════
        # # Component 3: TP/SL Penalty
        # # ═══════════════════════════════════════════════════════════════════════
        # # Penalize predictions that would hit stop-loss
        # # If SL is hit, the trade is bad regardless of predicted strength
        # sl_penalty = sl_hit_ratio.mean()

        # ═══════════════════════════════════════════════════════════════════════
        # Total Loss
        # ═══════════════════════════════════════════════════════════════════════

        price_scale = reversal_signal_loss.detach() / (price_prediction_loss.detach() + 1e-8)
        total_loss = price_prediction_loss + price_scale * reversal_signal_loss

        # metrics = {
        #     'loss': total_loss.item(),
        #     'strength_loss': strength_loss.item(),
        #     'price_loss': price_loss.item(),
        #     'highest_loss': highest_loss.item(),
        #     'lowest_loss': lowest_loss.item(),
        #     'sl_penalty': sl_penalty.item(),
        #     'avg_target_strength': target_strength.abs().mean().item(),
        #     'avg_pred_strength': pred_strength.abs().mean().item(),
        #     'avg_target_highest': target_highest.mean().item(),
        #     'avg_pred_highest': pred_highest.mean().item(),
        #     'avg_target_lowest': target_lowest.mean().item(),
        #     'avg_pred_lowest': pred_lowest.mean().item(),
        #     'bullish_ratio': is_bullish.float().mean().item(),
        #     'bearish_ratio': is_bearish.float().mean().item(),
        #     'neutral_ratio': (~is_bullish & ~is_bearish).float().mean().item(),
        #     'tp_hit_ratio': tp_hit_ratio.mean().item(),
        #     'sl_hit_ratio': sl_hit_ratio.mean().item(),
        # }

        return total_loss

"""
Simplified Reversal Signal Loss Function

Measures reversal signal strength [-1,1] by:
- Time span to reach extreme price (weighted by power_distance)
- Magnitude of price change from last price of current sequence
- Price proximity weighting (prices closer to dip weighted more)
- Hold check: strength = 0 if price fails to hold the dip
- TP/SL capping: trajectory is truncated at whichever cap is hit first

Formula: strength = weight_func(time) * (highest_price - dip_price) / Volatility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.functions import power_distance


class SimpleReversalLoss(nn.Module):
    """
    Simplified loss for reversal signal strength.

    Computes target reversal strength from future prices and compares
    against model prediction via MSE.

    Args:
        L: look-ahead window size (number of future time steps)
        tp_bps: take-profit cap in bps (None = no cap)
        sl_bps: stop-loss cap in bps (None = no cap)
    """
    def __init__(self, L=20, tp_bps=None, sl_bps=None):
        super().__init__()
        self.L = L
        self.tp_bps = tp_bps
        self.sl_bps = sl_bps

    def forward(self, predictions, future_price_changes):
        """
        Args:
            predictions: [B] or [B, 1] - model reversal signal in [-1, 1]
            future_price_changes: [B, K] - future price changes in bps

        Returns:
            loss: scalar MSE loss
            metrics: dict of logging metrics
        """
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        B = predictions.shape[0]
        device = predictions.device
        L = future_price_changes.shape[1]

        # Adjust to target look-ahead window
        if L > self.L:
            future_price_changes = future_price_changes[:, :self.L]
        elif L < self.L:
            future_price_changes = F.pad(future_price_changes, (0, self.L - L))
        L = self.L

        # Cumulative price from last price of current sequence (entry = 0)
        cumsum = torch.cumsum(future_price_changes, dim=1)  # [B, L]

        # TP/SL capping: truncate trajectory at whichever cap is hit first
        tp_hit_ratio = torch.zeros(B, device=device)
        sl_hit_ratio = torch.zeros(B, device=device)

        if self.tp_bps is not None and self.sl_bps is not None:
            step_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
            big_val = L

            # Bullish direction: TP = price >= tp_bps, SL = price <= -sl_bps
            tp_bull = cumsum >= self.tp_bps
            sl_bull = cumsum <= -self.sl_bps
            tp_bull_first = torch.where(tp_bull, step_idx, big_val).min(dim=1)[0]  # [B]
            sl_bull_first = torch.where(sl_bull, step_idx, big_val).min(dim=1)[0]  # [B]

            # Bearish direction: TP = price <= -tp_bps, SL = price >= sl_bps
            tp_bear = cumsum <= -self.tp_bps
            sl_bear = cumsum >= self.sl_bps
            tp_bear_first = torch.where(tp_bear, step_idx, big_val).min(dim=1)[0]  # [B]
            sl_bear_first = torch.where(sl_bear, step_idx, big_val).min(dim=1)[0]  # [B]

            # Use the earlier exit for each direction
            exit_bull = torch.min(tp_bull_first, sl_bull_first)  # [B]
            exit_bear = torch.min(tp_bear_first, sl_bear_first)  # [B]
            # Pick whichever direction exits first (determines the relevant cap)
            exit_idx = torch.min(exit_bull, exit_bear)  # [B]

            # Zero out cumsum beyond exit point and clamp
            mask = step_idx <= exit_idx.unsqueeze(1)  # [B, L]
            cumsum = cumsum * mask.float()
            cumsum = cumsum.clamp(-self.sl_bps, self.tp_bps)

            # Metrics
            tp_hit_ratio = ((tp_bull_first < sl_bull_first) | (tp_bear_first < sl_bear_first)).float()
            sl_hit_ratio = ((sl_bull_first < tp_bull_first) | (sl_bear_first < tp_bear_first)).float()

        # Volatility for normalization
        volatility = torch.std(future_price_changes, dim=1) + 1e-6  # [B]

        # Find extremes
        max_prices, max_idx = cumsum.max(dim=1)  # [B]
        min_prices, min_idx = cumsum.min(dim=1)  # [B]

        # Determine signal type based on whether dip holds
        is_bullish = min_prices >= 0  # [B]
        is_bearish = max_prices <= 0  # [B]

        # Time weight: earlier extreme = stronger signal
        extreme_idx = torch.where(is_bullish, max_idx, min_idx).float()  # [B]
        time_weight = 1.0 / (power_distance(extreme_idx + 1.0) + 1e-6)  # [B]

        # Price proximity weights: closer to entry (0) = higher weight
        price_dist = cumsum.abs()  # [B, L]
        price_weights = 1.0 / (power_distance(price_dist + 1.0) + 1e-6)  # [B, L]
        price_weights = price_weights / (price_weights.sum(dim=1, keepdim=True) + 1e-6)

        # Directional movement from entry
        movement = torch.where(
            is_bullish.unsqueeze(1),
            F.relu(cumsum),
            F.relu(-cumsum),
        )  # [B, L]

        # Price-weighted magnitude
        weighted_mag = (movement * price_weights).sum(dim=1)  # [B]

        # Reversal strength
        strength = time_weight * weighted_mag / volatility  # [B]

        # Normalize to [-1, 1] and apply direction
        target = torch.tanh(strength / 5.0)  # [B]
        target = torch.where(is_bearish, -target, target)
        target = torch.where(~is_bullish & ~is_bearish, torch.zeros_like(target), target)

        loss = F.mse_loss(predictions, target)

        metrics = {
            'loss': loss.item(),
            'avg_target_strength': target.abs().mean().item(),
            'avg_pred_strength': predictions.abs().mean().item(),
            'bullish_ratio': is_bullish.float().mean().item(),
            'bearish_ratio': is_bearish.float().mean().item(),
            'neutral_ratio': (~is_bullish & ~is_bearish).float().mean().item(),
            'avg_weighted_mag': weighted_mag.mean().item(),
            'avg_volatility': volatility.mean().item(),
            'tp_hit_ratio': tp_hit_ratio.mean().item(),
            'sl_hit_ratio': sl_hit_ratio.mean().item(),
        }

        return loss, metrics
