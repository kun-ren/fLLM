# Backtest and Inference Implementation

Complete implementation of backtesting and inference capabilities for the Crossformer model, fully integrated with the web frontend.

## 🎯 Components Implemented

### 1. **Model Inference Module** (`models/inference.py`)
Standalone inference wrapper for trained models:
- **ModelInference Class**: Loads checkpoint and reconstructs model architecture
- **Automatic Architecture Recovery**: Reads hyperparameters from checkpoint
- **Batch Prediction Support**: Efficient inference on datasets
- **Multi-task Output**: Returns reversal confidence, support, and resistance predictions

**Key Features:**
- Loads model weights from `.pt` checkpoint files
- Extracts hyperparameters automatically (no manual config needed)
- Sets model to eval mode with `torch.no_grad()`
- Returns predictions as numpy arrays or dictionaries

### 2. **Inference Service** (`models/inference_service.py`)
Production-ready inference management:
- **Model Registry**: Load/unload multiple models in memory
- **Active Model Selection**: Switch between loaded models
- **Single & Batch Predictions**: Flexible prediction API
- **Model Information**: Query model parameters and hyperparameters

**API:**
```python
service = get_inference_service()
service.load_model("checkpoints/model.pt", "my_model")
service.set_active_model("my_model")
predictions = service.predict(input_tensor)
```

### 3. **Enhanced Backtest Engine** (`backtest/engine.py`)
Updated to use the new inference module:
- **Simplified Model Loading**: Uses `ModelInference` instead of manual reconstruction
- **Multi-task Predictions**: Leverages reversal, support, and resistance signals
- **Strategy Execution**: TP/SL logic with commission calculation
- **Comprehensive Metrics**: 15+ performance indicators

**Changes:**
- Replaced manual encoder/taskheads loading with `ModelInference`
- Updated `run_inference()` to return all prediction types
- Modified `execute_strategy()` to accept multi-task predictions

### 4. **Backend API Extensions** (`web/backend/app.py`)
Added 8 new inference endpoints:

**Model Management:**
- `POST /api/inference/load` - Load model into inference service
- `POST /api/inference/unload` - Unload model from memory
- `GET /api/inference/models` - List loaded models
- `POST /api/inference/set-active` - Set active model

**Prediction:**
- `POST /api/inference/predict` - Single sample prediction
- `POST /api/inference/predict-batch` - Batch prediction
- `GET /api/inference/model-info` - Get model information

### 5. **Inference Panel** (`web/frontend/src/components/InferencePanel.js`)
Complete UI for model inference:

**Features:**
- **Model Loading**: Select and load checkpoints into memory
- **Model Management**: View loaded models, set active, unload
- **Model Info Display**: Shows parameter counts, device, hyperparameters
- **Interactive Inference**: 
  - JSON input field for test data
  - Run predictions on active model
  - Display results (reversal confidence, support, resistance)
- **Visual Feedback**: Color-coded predictions (green=bullish, red=bearish)

### 6. **Updated Frontend** (`web/frontend/src/App.js`)
Extended to 5-tab navigation:
1. Training
2. Backtest
3. **Inference** (NEW)
4. History
5. Configuration

## 🔄 Complete Workflow

### Training → Inference → Backtest

**1. Train Model:**
```
Configuration Tab → Set hyperparameters
Training Tab → Start training
→ Model saved to checkpoints/run_XXX_eN.pt
```

**2. Load for Inference:**
```
Inference Tab → Select checkpoint → Load Model
→ Model loaded into inference service
→ View model info (parameters, hyperparams)
```

**3. Run Predictions:**
```
Inference Tab → Enter input data (JSON) → Run Inference
→ Get reversal confidence, support, resistance
```

**4. Backtest Strategy:**
```
Backtest Tab → Select same checkpoint
→ Configure strategy (threshold, TP/SL)
→ Run backtest on test data
→ View comprehensive metrics and equity curve
```

## 📊 Prediction Output Format

**Single Prediction:**
```json
{
  "reversal_confidence": 0.734,  // [-1, 1] bullish/bearish
  "support_level": -8.5,          // bps below current price
  "resistance_level": 12.3        // bps above current price
}
```

**Batch Prediction:**
```json
{
  "reversal_confidence": [0.734, -0.421, 0.156, ...],
  "support_level": [-8.5, -6.2, -9.1, ...],
  "resistance_level": [12.3, 8.7, 15.4, ...]
}
```

## 🔧 Technical Details

### Model Loading Process
1. Read checkpoint file (`.pt`)
2. Extract `hyperparams` dict from checkpoint
3. Reconstruct `CrossformerEncoder` with saved architecture
4. Reconstruct `MultiTaskHead` with saved configuration
5. Load state dicts for encoder and task heads
6. Set to eval mode

### Inference Pipeline
1. Input: `[B, L, C]` tensor (batch, sequence length, channels)
2. Encoder forward pass → `[B, C, L, d_model]` embeddings
3. Task heads forward pass → dict of predictions
4. Output: reversal, support, resistance tensors

### Backtest Integration
- Uses same `ModelInference` class as inference service
- Runs predictions on entire test dataset
- Executes trading strategy with TP/SL logic
- Calculates performance metrics

## 🚀 Usage Examples

### Python API
```python
# Load model for inference
from models.inference import ModelInference

model = ModelInference("checkpoints/model.pt", device='cuda')
predictions = model.predict(input_tensor)

# Run backtest
from backtest.engine import BacktestEngine

engine = BacktestEngine(
    model_path="checkpoints/model.pt",
    confidence_threshold=0.6,
    take_profit_bps=5.0,
    stop_loss_bps=10.0
)
result = engine.run_backtest()
print(f"Win rate: {result.win_rate:.2%}")
```

### REST API
```bash
# Load model
curl -X POST http://localhost:5000/api/inference/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "checkpoints/model.pt"}'

# Run prediction
curl -X POST http://localhost:5000/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": [[0.1, 0.2, ...]]}'
```

## 📁 File Structure

```
models/
├── inference.py           # NEW: Model inference wrapper
├── inference_service.py   # NEW: Inference service manager
├── crossformer.py         # Training code
├── task_heads.py          # Multi-task prediction heads
└── reversal_loss.py       # Loss function

backtest/
└── engine.py              # UPDATED: Uses ModelInference

web/
├── backend/
│   └── app.py             # UPDATED: +8 inference endpoints
└── frontend/
    └── src/
        ├── components/
        │   └── InferencePanel.js  # NEW: Inference UI
        ├── App.js         # UPDATED: 5-tab navigation
        └── api.js         # UPDATED: Inference API calls
```

## ✅ Integration Complete

All components are now fully integrated:
- ✅ Model training saves checkpoints with hyperparameters
- ✅ Inference module loads models from checkpoints
- ✅ Backtest engine uses inference module
- ✅ Backend API exposes inference endpoints
- ✅ Frontend UI provides complete inference interface
- ✅ End-to-end workflow: Train → Inference → Backtest
