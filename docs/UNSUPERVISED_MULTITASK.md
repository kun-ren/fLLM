# Fully Unsupervised Multi-Task Trading Signal Learning

## Overview

This framework implements **fully unsupervised multi-task learning** for cryptocurrency trading signals. All components learn from future price movements without any explicit labels.

## Key Innovation: Profit-Based Reinforcement Learning

Unlike traditional supervised learning that requires labeled data, this approach uses **future price movements as the only supervision signal**. The model learns to:

1. **Predict high confidence** when future prices are profitable
2. **Predict resistance/support levels** that align with actual price extremes
3. **Optimize risk-reward ratios** automatically

## Architecture

```
Input [B, C, L]
    ↓
Crossformer Encoder (shared)
    ↓
Embedding [B, C, L, d_model]
    ↓
    ├─→ Reversal Head → [B, 1] confidence (unsupervised via profit)
    ├─→ Trend Head → [B, pred_len] future prices (supervised by actual prices)
    ├─→ Resistance Head → [B, 3] (primary, secondary, confidence) (self-supervised)
    └─→ Support Head → [B, 3] (primary, secondary, confidence) (self-supervised)
```

## Loss Components

### 1. Reversal Loss (Unsupervised - Profit-Based RL)

**Key insight**: The model learns to predict high confidence when future price movements are profitable, without knowing what a "reversal" is.

```python
# Compute actual profit from future prices
price_changes = (future_prices - current_price) / current_price * 100  # [B, pred_len]
cumsum_profit = torch.cumsum(price_changes, dim=1)  # Cumulative profit

# Time-weighted profit (recent steps more important)
weights = 3.0 / (i + 3.0)  # Decay weights
weighted_profit = weights * cumsum_profit
unrealized_profit = weighted_profit.mean(dim=1)

# Stop loss penalty
stop_loss_mask = cumsum_profit < -stop_loss_bps
stop_loss_penalty = (stop_loss_mask * (-weights)).sum(dim=1)

# Net profit
net_profit = unrealized_profit + stop_loss_penalty

# Loss: maximize profit weighted by confidence
shaped_profit = tanh(net_profit / threshold_bps)
loss = -softplus(shaped_profit * amplified_atanh(confidence))
```

**What the model learns:**
- High confidence → High profit correlation
- Automatically discovers patterns that lead to profitable trades
- No need to define what a "reversal" is
- Learns from bid/ask imbalance patterns implicitly (if included in features)

### 2. Trend Loss (Supervised by Future Prices)

```python
# MSE on predicted vs actual future prices
mse_loss = ((pred_prices - actual_prices) ** 2 * temporal_weights).mean()

# Directional accuracy
direction_loss = (pred_direction != actual_direction).float().mean()

total_trend_loss = mse_loss + direction_weight * direction_loss
```

**What the model learns:**
- Predict future price trajectory accurately
- Match directional movements
- Recent predictions weighted more than distant ones

### 3. Resistance/Support Loss (Self-Supervised)

**Key insight**: Resistance and support are learned by aligning with predicted and actual price extremes.

```python
# Consistency with predicted trend
pred_max = pred_prices.max(dim=1)
pred_resistance_pct = ((pred_max - current_price) / current_price) * 100
resistance_loss = MSE(predicted_resistance, pred_resistance_pct)

# Consistency with actual future prices
actual_max = actual_prices.max(dim=1)
actual_resistance_pct = ((actual_max - current_price) / current_price) * 100
resistance_loss += MSE(predicted_resistance, actual_resistance_pct)

# Same for support (using min instead of max)
```

**What the model learns:**
- Resistance = maximum upside potential
- Support = maximum downside risk
- Automatically learns to predict take-profit and stop-loss levels

### 4. Risk-Reward Ratio Loss

```python
risk_reward_ratio = resistance / (support + eps)
loss = relu(1.5 - risk_reward_ratio).mean()
```

**What the model learns:**
- Prefer trade setups with resistance > 1.5 × support
- Automatically filters low-quality signals

## Complete Loss Function

```python
Total Loss = w1·L_reversal + w2·L_trend + w3·L_resistance + w4·L_support + w5·L_risk_reward

where:
- L_reversal: Profit-based RL (unsupervised)
- L_trend: MSE on future prices (supervised by data)
- L_resistance: Consistency with price extremes (self-supervised)
- L_support: Consistency with price extremes (self-supervised)
- L_risk_reward: Risk-reward ratio constraint (self-supervised)
```

## Comparison with Your Previous Solution

### Your Original `loss.py` (Single Task)

```python
class ProfitLoss(nn.Module):
    def forward(self, y_prob, reference_k):
        # reference_k: [B, K] future price changes
        cumsum_profit = torch.cumsum(reference_k, dim=1)
        
        # Time-weighted profit
        weights = 3.0 / (i + 3.0)
        weighted_profit = weights * cumsum_profit
        unrealized_profit = weighted_profit.mean(dim=1)
        
        # Stop loss penalty
        stop_loss_penalty = (stop_loss_mask * (-weights)).sum(dim=1)
        net_profit = unrealized_profit + stop_loss_penalty
        
        # Maximize profit weighted by confidence
        loss = -softplus(shaped_profit * amplified_atanh(y_prob))
```

**Limitations:**
- Single output (confidence only)
- No explicit take-profit/stop-loss prediction
- No future price trajectory prediction

### New Multi-Task Framework

**Improvements:**
1. **Multiple outputs**: confidence, trend, resistance, support
2. **Explicit risk management**: Predicts take-profit and stop-loss levels
3. **Better interpretability**: Can see predicted price trajectory
4. **Coordinated learning**: All heads learn complementary patterns
5. **Same profit-based RL**: Keeps your proven reversal detection approach

## Training Process

### Data Requirements

```python
# Minimal data needed - no labels required!
features: [N, C, L]        # Input features (OHLCV, indicators, bid/ask volumes)
future_prices: [N, pred_len]  # Future prices (ground truth)
current_prices: [N]        # Current price
```

### Training Loop

```python
from models.multitask_trainer import MultiTaskTradingModel, train_multitask_model

model = MultiTaskTradingModel(
    data_dim=10,
    in_len=96,
    out_len=24,
    d_model=256,
    pred_len=24,
    pooling="AttentionPooling",
)

train_multitask_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
)
```

## Inference and Trading Signals

```python
from models.multitask_trainer import inference

signals = inference(model, x, device='cuda')

# Extract signals
confidence = signals['reversal_probability']      # [B] - trade confidence
predicted_prices = signals['predicted_prices']    # [B, pred_len] - price trajectory
take_profit_pct = signals['take_profit_pct']      # [B] - % above current
stop_loss_pct = signals['stop_loss_pct']          # [B] - % below current
resistance_conf = signals['resistance_confidence'] # [B] - resistance confidence
support_conf = signals['support_confidence']      # [B] - support confidence

# Trading logic
for i in range(len(confidence)):
    if confidence[i] > 0.7:  # High confidence signal
        current_price = x[i, 0, -1]
        
        # Calculate levels
        tp_price = current_price * (1 + take_profit_pct[i] / 100)
        sl_price = current_price * (1 - stop_loss_pct[i] / 100)
        risk_reward = take_profit_pct[i] / stop_loss_pct[i]
        
        # Filter by risk-reward and confidence
        if risk_reward > 1.5 and resistance_conf[i] > 0.6:
            print(f"TRADE: Entry={current_price:.2f}, TP={tp_price:.2f}, "
                  f"SL={sl_price:.2f}, R/R={risk_reward:.2f}")
```

## How It Learns Reversal Signals from Bid/Ask Imbalance

**Without explicit labels**, the model learns to detect reversals by:

1. **Feature engineering**: Include bid/ask volume imbalance in input features
   ```python
   # Add these to your feature channels
   bid_volume = bid_levels.sum(dim=-1)  # [B, L]
   ask_volume = ask_levels.sum(dim=-1)  # [B, L]
   imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
   ```

2. **Profit correlation**: The model discovers that certain imbalance patterns correlate with profitable price movements

3. **Automatic pattern discovery**: Through profit-based RL, the model learns:
   - When strong bid volume predicts upward moves
   - When strong ask volume predicts downward moves
   - When imbalance fails to predict direction (reversal signals!)

4. **No manual labeling**: You don't need to define what a "reversal" is - the model discovers it through profit optimization

## Advantages Over Supervised Learning

### Traditional Supervised Approach
- ❌ Requires manual labeling of reversals
- ❌ Label quality depends on human judgment
- ❌ Hard to define "reversal" objectively
- ❌ Doesn't directly optimize for profit

### This Unsupervised Approach
- ✅ No labels needed - learns from price data only
- ✅ Directly optimizes for profit
- ✅ Automatically discovers reversal patterns
- ✅ Learns take-profit and stop-loss levels
- ✅ Adapts to changing market conditions

## Hyperparameters

```python
# Loss weights
reversal_weight = 1.0      # Profit-based RL importance
trend_weight = 1.0         # Trend prediction importance
resistance_weight = 0.5    # Resistance consistency
support_weight = 0.5       # Support consistency
risk_reward_weight = 0.3   # Risk-reward ratio

# Profit thresholds
threshold_bps = 2.0        # Target profit threshold (2%)
stop_loss_bps = 2.0        # Stop loss threshold (2%)

# Model architecture
d_model = 256
hidden_dim = 128
n_heads = 4
e_layers = 3
dropout = 0.2
```

## Files

- `models/multitask_loss.py` - Fully unsupervised loss functions
- `models/multitask_trainer.py` - Training framework
- `models/task_heads.py` - Task-specific prediction heads
- `models/pooling.py` - Pooling strategies
- `models/crossformer.py` - Encoder architecture

## Next Steps

1. **Prepare your data**: Add bid/ask volume features to input
2. **Train the model**: Use the provided training script
3. **Backtest signals**: Evaluate on historical data
4. **Tune hyperparameters**: Adjust loss weights based on performance
5. **Monitor confidence scores**: Filter trades by confidence thresholds
