# fLLM Web Dashboard

Modern web interface for training, backtesting, and managing the fLLM cryptocurrency prediction model.

## Features

### 1. Training Dashboard
- Real-time training progress with live loss curves
- Start/stop training control
- GPU monitoring and system info
- Server-Sent Events for instant updates

### 2. Backtest Panel
- Load trained model checkpoints
- Configure strategy parameters (confidence threshold, TP/SL, commission)
- Run backtests on test data
- Comprehensive performance metrics:
  - Win rate, PnL, Sharpe/Sortino/Calmar ratios
  - Max drawdown, profit factor
  - Equity curve and exit reason analysis

### 3. Training History
- View all past training sessions
- Compare multiple sessions side-by-side
- Session detail view with full metrics
- Delete old sessions

### 4. Configuration Panel
- Interactive parameter controls for all hyperparameters
- Save/load configuration presets
- Grouped by category (Data, Model, Optimizer, Training, Loss, Backtest)

## Architecture

### Frontend (React + Material-UI + Recharts)
- **Material-UI**: Professional UI components with dark theme
- **Recharts**: Real-time loss curve and equity curve visualization
- **Server-Sent Events**: Live training progress streaming

### Backend (Flask)
- RESTful API for configuration management
- Real-time training control and monitoring
- Backtest execution engine
- Training session persistence
- Integration with `config_manager` for parameter persistence

## Installation

### Backend Setup

```bash
cd web/backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd web/frontend
npm install
```

## Running the Application

### Start Backend Server

```bash
cd web/backend
python app.py
```

Backend runs on `http://localhost:5000`

### Start Frontend Development Server

```bash
cd web/frontend
npm start
```

Frontend runs on `http://localhost:3000`

## Project Structure

```
web/
├── backend/
│   ├── app.py              # Flask API server (40+ endpoints)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── ConfigPanel.js        # Parameter configuration UI
│   │   │   ├── TrainingDashboard.js  # Training control & visualization
│   │   │   ├── BacktestPanel.js      # Backtest execution & results
│   │   │   ├── TrainHistoryPanel.js  # Session management & comparison
│   │   │   └── SystemInfo.js         # GPU/system status
│   │   ├── api.js          # API client
│   │   ├── App.js          # Main application (4-tab navigation)
│   │   └── index.js        # Entry point
│   └── package.json        # Node dependencies
├── api_spec.md             # API documentation
├── FEATURES.md             # Detailed feature documentation
└── README.md               # This file

backtest/
└── engine.py               # Backtesting engine
```

## API Endpoints

See `web/api_spec.md` for complete API documentation.

### Key Endpoint Groups
- **Configuration**: CRUD operations on hyperparameters and presets
- **Training**: Start/stop training, real-time status streaming
- **History**: List/view/delete/compare training sessions
- **Backtest**: Execute backtests, retrieve results
- **Models**: List/delete model checkpoints
- **System**: GPU and device information

## Technology Stack

**Frontend:**
- React 18.2
- Material-UI 5.15
- Recharts 2.12
- Axios 1.6

**Backend:**
- Flask 3.0
- Flask-CORS 4.0
- PyTorch (for model loading)

## Usage Workflow

### Training
1. Configure hyperparameters in **Configuration** tab
2. Start training in **Training** tab
3. Monitor real-time loss curves and progress
4. View completed session in **History** tab

### Backtesting
1. Select trained model checkpoint in **Backtest** tab
2. Configure strategy parameters (threshold, TP/SL, commission)
3. Run backtest on test data
4. Analyze comprehensive performance metrics and equity curve

### Session Management
1. View all training sessions in **History** tab
2. Select multiple sessions for comparison
3. View detailed metrics and configuration for each session
4. Delete old or failed sessions

## Development Notes

- Frontend proxies API requests to `http://localhost:5000` in development
- Backend uses threading for non-blocking training and backtest execution
- SSE (Server-Sent Events) provides real-time training updates without polling
- All parameters from `controller/schema.py` are automatically rendered in the UI
- Training sessions auto-saved to `training_history/` directory
- Model checkpoints stored in `checkpoints/` directory

## Performance Metrics

The backtest engine calculates 15+ metrics:
- **Returns**: Total PnL (bps and %), average win/loss
- **Risk**: Max drawdown, Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Stats**: Win rate, profit factor, consecutive wins/losses
- **Execution**: Average hold periods, exit reasons (TP/SL/timeout)

