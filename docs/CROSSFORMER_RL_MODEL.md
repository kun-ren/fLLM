# Crossformer Reinforcement Learning Model

## Overview

The `CrossformerRLModel` in `models/crossformer.py` is a complete reinforcement learning system for cryptocurrency trading that combines:

1. **Crossformer Encoder** - Extracts features from multi-channel time series
2. **Four Task Heads** - Predict different aspects of trading signals
3. **Multi-Task Loss** - Coordinates all predictions with reward-based learning

## Architecture

```
Input [B, C, L]
    ↓
Crossformer Encoder
    ↓
Embedding [B, C, L, d_model]
    ↓
    ├─→ Reversal Head → [B, 1] confidence
    ├─→ Trend Head → [B, pred_len] future prices
    ├─→ Resistance Head → [B, 2] (level_%, confidence)
    └─→ Support Head → [B, 2] (level_%, confidence)
```

## Model Components

### 1. CrossformerRLModel

**Location**: `models/crossformer.py`

**Key Methods**:

```python
# Forward pass - get all predictions
predictions = model(x)
# Returns: {
#   'reversal': [B, 1],
#   'trend': [B, pred_len],
#   'resistance': [B, 2],
#   'support': [B, 2]
# }

# Compute reward/loss
reward, loss_dict = model.compute_reward(predictions, targets, loss_fn)

# Get trading signals
signals = model.get_trading_signals(predictions, current_price)
# Returns: {
#   'confidence': [B],
#   'trend_prices': [B, pred_len],
#   'tp_price': [B],
#   'sl_price': [B],
#   'resistance_conf': [B],
#   'support_conf': [B],
#   'horizon': [B],
#   'resistance_pct': [B],
#   'support_pct': [B]
# }
```

### 2. Task Heads (models/task_heads.py)

All heads simplified to output single level + confidence:

- **ReversalHead**: [B, 1] - reversal confidence
- **TrendHead**: [B, pred_len] - predicted future prices
- **ResistanceHead**: [B, 2] - (level_%, confidence)
- **SupportHead**: [B, 2] - (level_%, confidence)

### 3. Multi-Task Loss (models/multitask_loss.py)

The reward/loss is measured by:

1. **Reversal Confidence** - Profit-based RL
   - Quick reversals get higher reward
   - Adaptive threshold based on resistance/support
   - Temporal weighting: earlier = better

2. **Predicted Future Prices** - MSE + directional accuracy
   - Accurate trend predictions get higher reward
   - Variable-length loss based on dynamic horizon

3. **Resistance Area** - Consistency with trend max
   - Resistance should align with predicted price peak
   - Weighted by confidence

4. **Support Area** - Consistency with trend min
   - Support should align with predicted price bottom
   - Weighted by confidence

5. **Take Profit / Stop Loss** - Derived from resistance/support
   - TP = current_price × (1 + resistance_pct / 100)
   - SL = current_price × (1 - support_pct / 100)
   - Risk-reward ratio constraint (TP/SL > 1.5)

## Training

**Function**: `train_crossformer_rl()` in `models/crossformer.py`

```python
from models.crossformer import train_crossformer_rl

# Train the model
for progress in train_crossformer_rl():
    print(progress['log'])
    # Progress dict contains:
    # - status: loading/initializing/training/completed
    # - log: human-readable message
    # - progress: 0.0 to 1.0
    # - epoch: current epoch
    # - step: global step
    # - loss: current loss value
```

**Loss Components Tracked**:
- Total loss
- Reversal loss (profit-based)
- Trend loss (MSE)
- Resistance loss (consistency)
- Support loss (consistency)
- Risk-reward loss

## Usage Example

```python
import torch
from models.crossformer import CrossformerRLModel
from models.multitask_loss import MultiTaskTradingLoss

# Create model
model = CrossformerRLModel(
    data_dim=10,        # Number of input channels
    in_len=96,          # Input sequence length
    out_len=24,         # Output sequence length
    d_model=256,
    n_heads=4,
    e_layers=3,
    pred_len=24,
    pooling="AttentionPooling",
)

# Create loss function
loss_fn = MultiTaskTradingLoss(
    reversal_weight=1.0,
    trend_weight=1.0,
    resistance_weight=0.5,
    support_weight=0.5,
    risk_reward_weight=0.3,
    threshold_bps=2.0,
    stop_loss_bps=2.0,
)

# Forward pass
x = torch.randn(32, 10, 96)  # [batch, channels, length]
predictions = model(x)

# Prepare targets
current_price = x[:, 0, -1:].unsqueeze(1)  # [B, 1]
future_prices = torch.randn(32, 24)  # [B, pred_len]

targets = {
    'future_prices': future_prices,
    'current_price': current_price,
}

# Compute reward
reward, loss_dict = model.compute_reward(predictions, targets, loss_fn)

# Get trading signals
signals = model.get_trading_signals(predictions, current_price)

print(f"Confidence: {signals['confidence']}")
print(f"Take Profit: {signals['tp_price']}")
print(f"Stop Loss: {signals['sl_price']}")
print(f"Horizon: {signals['horizon']}")
```

## Inference

```python
# Load trained model
model.eval()

with torch.no_grad():
    # Get predictions
    predictions = model(x)
    
    # Extract trading signals
    signals = model.get_trading_signals(predictions, current_price)
    
    # Make trading decision
    for i in range(len(signals['confidence'])):
        if signals['confidence'][i] > 0.7:  # High confidence
            print(f"Trade Signal:")
            print(f"  Entry: {current_price[i].item():.2f}")
            print(f"  Take Profit: {signals['tp_price'][i].item():.2f}")
            print(f"  Stop Loss: {signals['sl_price'][i].item():.2f}")
            print(f"  Expected Horizon: {signals['horizon'][i].item():.0f} candles")
            print(f"  Risk/Reward: {signals['resistance_pct'][i].item() / signals['support_pct'][i].item():.2f}")
```

## Key Features

1. **Fully Unsupervised** - No manual labels needed, learns from price movements
2. **Multi-Task Learning** - All heads learn complementary patterns
3. **Adaptive Thresholds** - Reversal detection uses predicted resistance/support
4. **Dynamic Horizon** - Time to target computed from trend predictions
5. **Coordinated Loss** - All components work together for profitable trades
6. **Temporal Weighting** - Quick reversals rewarded more than slow ones

## Files Modified

1. `models/crossformer.py` - Added `CrossformerRLModel` and `train_crossformer_rl()`
2. `models/task_heads.py` - Simplified ResistanceHead and SupportHead to [B, 2]
3. `models/multitask_loss.py` - Already has the coordinated loss function

## Next Steps

1. **Train the model**: Run `train_crossformer_rl()`
2. **Evaluate signals**: Backtest on historical data
3. **Tune hyperparameters**: Adjust loss weights based on performance
4. **Deploy**: Use `get_trading_signals()` for live trading
