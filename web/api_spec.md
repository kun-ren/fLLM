# REST API Specification

## Base URL
`http://localhost:5000/api`

## Endpoints

### Configuration Management

#### GET /config
Get all hyperparameters with their current values.

**Response:**
```json
{
  "batch_size": {"name": "batch_size", "value": 64, "min_val": 4, "max_val": 256, "step": 4, "mode": "single", "group": "Training", "description": "Training batch size"},
  "d_model": {"name": "d_model", "value": 64, "min_val": 32, "max_val": 256, "step": 32, "mode": "single", "group": "Model", "description": "Model embedding dimension"},
  ...
}
```

#### GET /config/:name
Get a specific hyperparameter.

**Response:**
```json
{
  "name": "batch_size",
  "value": 64,
  "min_val": 4,
  "max_val": 256,
  "step": 4,
  "mode": "single",
  "group": "Training",
  "description": "Training batch size"
}
```

#### PUT /config/:name
Update a specific hyperparameter.

**Request:**
```json
{
  "value": 128,
  "mode": "single"
}
```

**Response:**
```json
{
  "success": true,
  "param": {"name": "batch_size", "value": 128, ...}
}
```

#### POST /config/save
Save current configuration to a named preset.

**Request:**
```json
{
  "name": "experiment_1"
}
```

**Response:**
```json
{
  "success": true,
  "path": "controller/configs/experiment_1.json"
}
```

#### POST /config/load
Load a saved configuration preset.

**Request:**
```json
{
  "name": "experiment_1"
}
```

**Response:**
```json
{
  "success": true,
  "config": {...}
}
```

#### GET /config/presets
List all saved configuration presets.

**Response:**
```json
{
  "presets": ["default", "experiment_1", "experiment_2"]
}
```

#### DELETE /config/presets/:name
Delete a saved configuration preset.

**Response:**
```json
{
  "success": true
}
```

### Training Control

#### POST /training/start
Start training with current configuration.

**Response:**
```json
{
  "success": true,
  "session_id": "train_20260415_143022"
}
```

#### POST /training/stop
Stop the current training session.

**Response:**
```json
{
  "success": true
}
```

#### GET /training/status
Get current training status.

**Response:**
```json
{
  "status": "training",
  "epoch": 5,
  "total_epochs": 10,
  "step": 1250,
  "total_steps": 2500,
  "loss": 0.0234,
  "progress": 0.5,
  "log": "Epoch 5/10 Step 1250/2500 Loss: 0.0234"
}
```

#### GET /training/stream (SSE)
Server-Sent Events stream for real-time training updates.

**Event Stream:**
```
event: training
data: {"status": "training", "epoch": 1, "step": 100, "loss": 0.045, "progress": 0.04, "log": "..."}

event: training
data: {"status": "training", "epoch": 1, "step": 200, "loss": 0.038, "progress": 0.08, "log": "..."}

event: completed
data: {"status": "completed", "epoch": 10, "step": 2500, "loss": 0.012, "progress": 1.0, "log": "Training completed!"}
```

### Metrics & Visualization

#### GET /metrics/history
Get training history for visualization.

**Query Parameters:**
- `session_id` (optional): specific training session
- `limit` (optional): number of recent points

**Response:**
```json
{
  "session_id": "train_20260415_143022",
  "history": [
    {"epoch": 1, "step": 100, "loss": 0.045, "timestamp": "2026-04-15T14:30:25"},
    {"epoch": 1, "step": 200, "loss": 0.038, "timestamp": "2026-04-15T14:30:30"},
    ...
  ]
}
```

#### GET /metrics/summary
Get training summary statistics.

**Response:**
```json
{
  "total_sessions": 5,
  "current_session": "train_20260415_143022",
  "best_loss": 0.0089,
  "avg_epoch_time": 120.5,
  "total_training_time": 3600
}
```

### System Info

#### GET /system/info
Get system and model information.

**Response:**
```json
{
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA RTX 3090",
  "memory_allocated": "2.5 GB",
  "memory_reserved": "4.0 GB"
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Error message description"
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (config/preset not found)
- 500: Internal Server Error
