# fLLM Web Dashboard - Extended Features

## New Features Added

### 1. Backtest Panel
Complete backtesting system for evaluating trained models on test data.

**Features:**
- Model checkpoint selection from saved models
- Interactive parameter configuration (confidence threshold, TP/SL, commission)
- Real-time backtest execution with progress tracking
- Comprehensive performance metrics:
  - Win rate, total PnL, Sharpe ratio
  - Max drawdown, profit factor
  - Average win/loss, hold periods
- Visual analytics:
  - Equity curve chart
  - Exit reasons distribution (TP/SL/Timeout)
  - Trade statistics table

**Backend (`backtest/engine.py`):**
- `BacktestEngine` class for strategy execution
- Loads trained model and runs inference on test data
- Simulates trades with TP/SL logic
- Calculates 15+ performance metrics
- Supports custom trading parameters

### 2. Training History Panel
Session management and comparison system for tracking training experiments.

**Features:**
- Table view of all training sessions with:
  - Session ID, timestamp, epochs, final loss
  - Multi-select for comparison
  - View details and delete actions
- Session detail dialog:
  - Full loss curve visualization
  - Configuration parameters used
  - Training metrics timeline
- Session comparison:
  - Side-by-side loss curve comparison
  - Parameter comparison table
  - Support for comparing 2+ sessions

**Backend Storage:**
- Training sessions auto-saved to `training_history/` as JSON
- Includes full history, config snapshot, and final metrics
- Persistent across application restarts

### 3. Enhanced Backend API

**New Endpoints:**

**Training History:**
- `GET /api/history/sessions` - List all training sessions
- `GET /api/history/sessions/:id` - Get session details
- `DELETE /api/history/sessions/:id` - Delete session
- `POST /api/history/compare` - Compare multiple sessions

**Backtest:**
- `POST /api/backtest/start` - Start backtest with parameters
- `GET /api/backtest/status` - Check backtest progress
- `GET /api/backtest/result` - Get backtest results

**Model Management:**
- `GET /api/models/list` - List saved model checkpoints
- `DELETE /api/models/:name` - Delete checkpoint

## Updated Architecture

```
web/
├── backend/
│   ├── app.py                    # Extended with 20+ new endpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── BacktestPanel.js       # NEW: Backtest UI
│   │   │   ├── TrainHistoryPanel.js   # NEW: History management
│   │   │   ├── ConfigPanel.js
│   │   │   ├── TrainingDashboard.js
│   │   │   └── SystemInfo.js
│   │   ├── api.js                # Extended with new endpoints
│   │   └── App.js                # 4-tab navigation
│   └── package.json
└── api_spec.md

backtest/
└── engine.py                     # NEW: Backtesting engine
```

## Usage Flow

### Training Workflow
1. **Configure** → Set hyperparameters in Configuration tab
2. **Train** → Start training in Training tab, monitor real-time progress
3. **History** → View completed session in History tab
4. **Compare** → Select multiple sessions to compare performance

### Backtesting Workflow
1. **Train** → Complete a training session (model auto-saved)
2. **Backtest** → Select model checkpoint in Backtest tab
3. **Configure** → Adjust strategy parameters (threshold, TP/SL)
4. **Run** → Execute backtest and view comprehensive results
5. **Analyze** → Review equity curve, metrics, and trade statistics

## Key Improvements

**Data Persistence:**
- Training sessions saved automatically on completion
- Model checkpoints managed through UI
- Configuration presets for reproducibility

**Performance Metrics:**
- 15+ backtest metrics (Sharpe, Sortino, Calmar ratios)
- Risk-adjusted returns and drawdown analysis
- Trade-by-trade breakdown

**User Experience:**
- 4-tab navigation (Training, Backtest, History, Config)
- Real-time updates via Server-Sent Events
- Interactive charts with Recharts
- Dark theme optimized for data visualization

## Installation & Running

**Backend:**
```bash
cd web/backend
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd web/frontend
npm install
npm start
```

Access at `http://localhost:3000`

## Technical Stack

**Backend:**
- Flask 3.0 with threading for async operations
- JSON-based session storage
- PyTorch model loading and inference

**Frontend:**
- React 18.2 with Material-UI 5.15
- Recharts 2.12 for data visualization
- Axios for API communication
- Server-Sent Events for real-time updates

## Future Enhancements

Potential additions:
- Model checkpoint comparison (architecture diff)
- Hyperparameter optimization (grid search UI)
- Real-time trading simulation
- Export backtest results to CSV/PDF
- Advanced filtering and search in history
- Model performance leaderboard
