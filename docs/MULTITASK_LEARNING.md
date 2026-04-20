# Multi-Task Trading Signal Learning Framework

## Overview

This framework implements a multi-task learning approach for cryptocurrency trading signal generation. It combines multiple prediction heads that learn complementary aspects of market behavior, specifically focusing on **reversal detection from bid/ask volume imbalance** and **risk management through resistance/support prediction**.

## Architecture

```
Input [B, C, L]
    ↓
Crossformer Encoder
    ↓
Embedding [B, C, L, d_model]
    ↓
    ├─→ Reversal Head → [B, 1] reversal probability
    ├─→ Trend Head (Decoder) → [B, pred_len] future prices
    ├─→ Resistance Head → [B, 3] (primary, secondary, confidence)
    └─→ Support Head → [B, 3] (primary, secondary, confidence)
```

## Key Components

### 1. Task Heads (`models/task_heads.py`)

#### ReversalHead
- **Purpose**: Detect potential trend reversals caused by bid/ask volume imbalance
- **Output**: `[B, 1]` - probability of reversal occurring
- **Architecture**: Pooling + MLP with sigmoid activation

#### LongTermTrendHead
- **Purpose**: Predict future price trajectory after reversal
- **Output**: `[B, pred_len]` - sequence of predicted prices
- **Architecture**: Crossformer Decoder + projection layer
- **Key feature**: Uses decoder architecture for sequence-to-sequence prediction

#### ResistanceHead
- **Purpose**: Predict take-profit levels (upside resistance)
- **Output**: `[B, 3]` - (primary_resistance_%, secondary_resistance_%, confidence)
- **Architecture**: Pooling + MLP with Softplus (ensures positive values)

#### SupportHead
- **Purpose**: Predict stop-loss levels (downside support)
- **Output**: `[B, 3]` - (primary_support_%, secondary_support_%, confidence)
- **Architecture**: Pooling + MLP with Softplus (ensures positive values)

### 2. Multi-Task Loss (`models/multitask_loss.py`)

The loss function coordinates all heads to learn complementary patterns:

```python
Total Loss = w1·L_reversal + w2·L_trend + w3·L_resistance + w4·L_support + w5·L_risk_reward
```

#### Loss Components

1. **Reversal Loss** (Binary Cross-Entropy)
   - Supervised by reversal labels computed from bid/ask imbalance + future price movement
   - Teaches the model to detect when volume imbalance predicts price reversal

2. **Trend Loss** (MSE + Directional Accuracy)
   - Supervised by actual future prices
   - Temporal weighting: recent predictions weighted more than distant ones
   - Directional penalty: ensures predicted trend direction matches reality

3. **Resistance Consistency Loss**
   - **Self-supervised**: Resistance should align with maximum predicted price from trend head
   - Formula: `L_resistance = MSE(primary_resistance, (pred_max - current) / current * 100)`
   - Weighted by confidence score

4. **Support Consistency Loss**
   - **Self-supervised**: Support should align with minimum predicted price from trend head
   - Formula: `L_support = MSE(primary_support, (current - pred_min) / current * 100)`
   - Weighted by confidence score

5. **Risk-Reward Ratio Loss**
   - Encourages profitable trade setups: `resistance / support > 1.5`
   - Penalizes setups with poor risk-reward ratios

### 3. Label Generation (`compute_reversal_labels`)

Reversal labels are computed from historical data:

```python
def compute_reversal_labels(prices, bid_volumes, ask_volumes, threshold_pct=2.0, window=5):
    """
    A reversal is detected when:
    1. Significant bid/ask imbalance occurs (|imbalance| > 0.2)
    2. Price moves > threshold_pct in OPPOSITE direction within window
    
    Example:
    - If bid > ask (imbalance > 0), expect price UP
      → Reversal = price goes DOWN > 2%
    - If ask > bid (imbalance < 0), expect price DOWN
      → Reversal = price goes UP > 2%
    """
```

This creates supervised labels that teach the model to recognize when volume imbalance fails to predict price direction (i.e., a reversal signal).

## Training Process

### Data Requirements

```python
TradingDataset requires:
- prices: [N, L] - historical prices
- features: [N, C, L] - multi-channel features (OHLCV, indicators, etc.)
- bid_volumes: [N, L, 10] - bid order book depth (10 levels)
- ask_volumes: [N, L, 10] - ask order book depth (10 levels)
- future_prices: [N, pred_len] - ground truth future prices
```

### Training Loop

```python
from models.multitask_trainer import MultiTaskTradingModel, train_multitask_model

# Create model
model = MultiTaskTradingModel(
    data_dim=10,        # Number of input channels
    in_len=96,          # Input sequence length
    out_len=24,         # Output sequence length
    d_model=256,        # Model dimension
    pred_len=24,        # Prediction horizon
    pooling="AttentionPooling",
)

# Train
train_multitask_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
)
```

## Inference and Signal Generation

```python
from models.multitask_trainer import inference

# Run inference
signals = inference(model, x, device='cuda')

# Extract trading signals
reversal_prob = signals['reversal_probability']  # [B]
predicted_prices = signals['predicted_prices']   # [B, pred_len]
take_profit_pct = signals['take_profit_pct']     # [B] - % above current
stop_loss_pct = signals['stop_loss_pct']         # [B] - % below current
resistance_conf = signals['resistance_confidence'] # [B]
support_conf = signals['support_confidence']      # [B]

# Trading logic example
for i in range(len(reversal_prob)):
    if reversal_prob[i] > 0.7:  # High reversal probability
        current_price = x[i, 0, -1]  # Assuming first channel is price
        
        # Set take profit
        tp_price = current_price * (1 + take_profit_pct[i] / 100)
        
        # Set stop loss
        sl_price = current_price * (1 - stop_loss_pct[i] / 100)
        
        # Check risk-reward ratio
        risk_reward = take_profit_pct[i] / stop_loss_pct[i]
        
        if risk_reward > 1.5 and resistance_conf[i] > 0.6:
            print(f"Trade signal: Entry={current_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}, R/R={risk_reward:.2f}")
```

## How It Addresses Your Objectives

### 1. Identify Reversal Signals from Bid/Ask Imbalance

- **ReversalHead** learns to detect when volume imbalance predicts reversals
- **Training labels** are generated by finding cases where:
  - Strong bid/ask imbalance exists
  - Price moves opposite to the imbalance direction
- Model learns patterns in the features that precede these reversals

### 2. Predict Reversal Magnitude

- **TrendHead** predicts the full future price trajectory
- **ResistanceHead** extracts the maximum upside (take profit target)
- **SupportHead** extracts the maximum downside (stop loss level)
- These are learned in a **self-supervised** way by aligning with the trend prediction

### 3. Set Take Profit and Stop Loss

- **Take Profit**: Use `primary_resistance` from ResistanceHead
  - `TP_price = current_price * (1 + primary_resistance / 100)`
- **Stop Loss**: Use `primary_support` from SupportHead
  - `SL_price = current_price * (1 - primary_support / 100)`
- **Confidence scores** help filter low-quality signals

### 4. Multi-Task Learning Benefits

- **Shared encoder**: All heads benefit from learning rich representations
- **Consistency constraints**: Resistance/support must align with trend predictions
- **Risk-reward optimization**: Model learns to predict profitable setups
- **Complementary signals**: Reversal timing + magnitude + risk levels

## Loss Function Optimization

The key insight is that **resistance and support are NOT directly supervised**, but instead are constrained to be consistent with the trend prediction:

```
Trend Head (supervised by future prices)
    ↓
Predicts future price sequence
    ↓
    ├─→ Max predicted price → Resistance target
    └─→ Min predicted price → Support target
```

This creates a **self-supervised learning signal** where:
- Trend head learns from actual future prices (supervised)
- Resistance/support heads learn to extract extremes from the trend (self-supervised)
- Risk-reward loss encourages profitable trade setups

## Hyperparameters

### Loss Weights
```python
reversal_weight = 1.0      # Reversal detection importance
trend_weight = 1.0         # Trend prediction importance
resistance_weight = 0.5    # Resistance consistency importance
support_weight = 0.5       # Support consistency importance
risk_reward_weight = 0.3   # Risk-reward ratio importance
```

### Model Architecture
```python
d_model = 256              # Model dimension
hidden_dim = 128           # Task head hidden dimension
n_heads = 4                # Attention heads
e_layers = 3               # Encoder layers
d_layers = 2               # Decoder layers (trend head)
dropout = 0.2              # Dropout rate
```

### Training
```python
learning_rate = 1e-4       # AdamW learning rate
weight_decay = 1e-5        # L2 regularization
batch_size = 32            # Batch size
num_epochs = 100           # Training epochs
```

## Files

- `models/task_heads.py` - Task-specific prediction heads
- `models/multitask_loss.py` - Multi-task loss functions and label generation
- `models/multitask_trainer.py` - Training framework and inference
- `models/pooling.py` - Pooling strategies for heads
- `models/crossformer.py` - Encoder architecture
- `models/crossformer/decoder.py` - Decoder for trend prediction

## Next Steps

1. **Integrate with your data pipeline**: Adapt `TradingDataset` to your data format
2. **Tune hyperparameters**: Adjust loss weights based on validation performance
3. **Add more features**: Include order book features, technical indicators, etc.
4. **Backtest signals**: Evaluate trading performance on historical data
5. **Monitor confidence scores**: Filter trades based on resistance/support confidence
