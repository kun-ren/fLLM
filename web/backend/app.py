"""
Flask backend API for training dashboard.
Provides REST endpoints for config management and training control.
"""
import logging
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from controller.config_manager import get_config_manager
from models.crossformer import train_crossformer_rl
from backtest.engine import BacktestEngine, run_backtest_from_config
from models.inference_service import get_inference_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global training state
training_state = {
    "active": False,
    "session_id": None,
    "current_status": None,
    "history": [],
    "thread": None,
    "stop_flag": False
}

# Global backtest state
backtest_state = {
    "active": False,
    "result": None,
    "thread": None
}

# Training history storage
HISTORY_DIR = Path("training_history")
HISTORY_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)


# ── Configuration Endpoints ──────────────────────────────────────────────────

@app.route('/api/config', methods=['GET'])
def get_all_config():
    """Get all hyperparameters."""
    try:
        config = get_config_manager()
        params = {name: param.to_dict() for name, param in config._params.items()}
        return jsonify(params)
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/<name>', methods=['GET'])
def get_config_param(name):
    """Get a specific hyperparameter."""
    try:
        config = get_config_manager()
        param = config.get(name)
        return jsonify(param.to_dict())
    except KeyError:
        return jsonify({"success": False, "error": f"Parameter '{name}' not found"}), 404
    except Exception as e:
        logger.error(f"Error getting param {name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/<name>', methods=['PUT'])
def update_config_param(name):
    """Update a specific hyperparameter."""
    try:
        data = request.json
        config = get_config_manager()

        value = data.get('value')
        mode = data.get('mode', 'single')
        min_val = data.get('min_val')
        max_val = data.get('max_val')

        from controller.schema import ParamMode
        config.set(name, value=value, mode=ParamMode(mode), min_val=min_val, max_val=max_val)

        return jsonify({
            "success": True,
            "param": config.get(name).to_dict()
        })
    except KeyError:
        return jsonify({"success": False, "error": f"Parameter '{name}' not found"}), 404
    except Exception as e:
        logger.error(f"Error updating param {name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save current configuration to a named preset."""
    try:
        data = request.json
        name = data.get('name', 'default')
        config = get_config_manager()
        path = config.save_config(name)
        return jsonify({
            "success": True,
            "path": str(path)
        })
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/load', methods=['POST'])
def load_config():
    """Load a saved configuration preset."""
    try:
        data = request.json
        name = data.get('name', 'default')
        config = get_config_manager()
        config.load_config(name)
        params = {name: param.to_dict() for name, param in config._params.items()}
        return jsonify({
            "success": True,
            "config": params
        })
    except FileNotFoundError:
        return jsonify({"success": False, "error": f"Config '{name}' not found"}), 404
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/presets', methods=['GET'])
def list_presets():
    """List all saved configuration presets."""
    try:
        config = get_config_manager()
        presets = config.list_configs()
        return jsonify({"presets": presets})
    except Exception as e:
        logger.error(f"Error listing presets: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/config/presets/<name>', methods=['DELETE'])
def delete_preset(name):
    """Delete a saved configuration preset."""
    try:
        config = get_config_manager()
        success = config.delete_config(name)
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Preset not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting preset: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Training Control Endpoints ───────────────────────────────────────────────

def training_worker():
    """Background worker for training."""
    global training_state
    session_id = training_state["session_id"]

    try:
        for progress in train_crossformer_rl():
            if training_state["stop_flag"]:
                logger.info("Training stopped by user")
                break

            training_state["current_status"] = progress
            training_state["history"].append({
                **progress,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(progress["log"])

        # Save training session on completion
        if training_state["current_status"].get("status") == "completed":
            save_training_session(session_id, training_state["history"])

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_state["current_status"] = {
            "status": "error",
            "log": str(e),
            "progress": 0.0
        }
    finally:
        training_state["active"] = False


def save_training_session(session_id, history):
    """Save training session to disk."""
    try:
        session_file = HISTORY_DIR / f"{session_id}.json"
        config = get_config_manager()

        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "config": {name: param.to_dict() for name, param in config._params.items()},
            "history": history,
            "final_loss": history[-1]["loss"] if history else None,
            "epochs": history[-1]["epoch"] if history else 0
        }

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Training session saved: {session_file}")
    except Exception as e:
        logger.error(f"Failed to save training session: {e}")


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training with current configuration."""
    from controller.train_worker import get_train_worker

    worker = get_train_worker()

    if worker.is_running():
        return jsonify({
            "success": False,
            "error": "Training already in progress"
        }), 400

    try:
        worker.start(run_type="train")

        return jsonify({
            "success": True,
            "run_id": worker.run_id
        })
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training session."""
    from controller.train_worker import get_train_worker

    worker = get_train_worker()

    if not worker.is_running():
        return jsonify({
            "success": False,
            "error": "No training in progress"
        }), 400

    worker.stop()
    return jsonify({"success": True})


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    if training_state["current_status"]:
        return jsonify(training_state["current_status"])
    else:
        return jsonify({
            "status": "idle",
            "progress": 0.0
        })


@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
    """Get current training progress from train_worker."""
    from controller.train_worker import get_train_worker

    worker = get_train_worker()
    return jsonify({
        "status": worker.status,
        "progress": worker.progress,
        "epoch": worker.epoch,
        "total_epochs": worker.total_epochs,
        "current_loss": worker.current_loss,
        "run_id": worker.run_id,
        "run_type": worker.run_type
    })


@app.route('/api/training/stream')
def training_stream():
    """Server-Sent Events stream for real-time training updates."""
    def generate():
        last_step = -1
        while True:
            if training_state["current_status"]:
                status = training_state["current_status"]
                current_step = status.get("step", 0)

                # Only send if step changed
                if current_step != last_step:
                    event_type = status.get("status", "training")
                    yield f"event: {event_type}\ndata: {jsonify(status).get_data(as_text=True)}\n\n"
                    last_step = current_step

                # Stop streaming if completed or error
                if status.get("status") in ["completed", "error"]:
                    break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


# ── Metrics Endpoints ────────────────────────────────────────────────────────

@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    """Get training history for visualization."""
    session_id = request.args.get('session_id')
    limit = request.args.get('limit', type=int)

    history = training_state["history"]

    if limit:
        history = history[-limit:]

    return jsonify({
        "session_id": training_state["session_id"],
        "history": history
    })


@app.route('/api/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get training summary statistics."""
    history = training_state["history"]

    if not history:
        return jsonify({
            "total_sessions": 0,
            "current_session": None,
            "best_loss": None,
            "total_training_time": 0
        })

    losses = [h["loss"] for h in history if h.get("loss")]
    best_loss = min(losses) if losses else None

    return jsonify({
        "total_sessions": 1,
        "current_session": training_state["session_id"],
        "best_loss": best_loss,
        "total_training_time": len(history)
    })


# ── System Info Endpoints ────────────────────────────────────────────────────

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Get system and model information."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        info["memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"

    return jsonify(info)


# ── Training History Endpoints ───────────────────────────────────────────────

@app.route('/api/history/sessions', methods=['GET'])
def list_training_sessions():
    """List all saved training sessions."""
    try:
        sessions = []
        for session_file in HISTORY_DIR.glob("*.json"):
            with open(session_file) as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "timestamp": data["timestamp"],
                    "final_loss": data.get("final_loss"),
                    "epochs": data.get("epochs"),
                })

        # Sort by timestamp descending
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify({"sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/history/sessions/<session_id>', methods=['GET'])
def get_training_session(session_id):
    """Get detailed training session data."""
    try:
        session_file = HISTORY_DIR / f"{session_id}.json"
        if not session_file.exists():
            return jsonify({"success": False, "error": "Session not found"}), 404

        with open(session_file) as f:
            data = json.load(f)

        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/history/sessions/<session_id>', methods=['DELETE'])
def delete_training_session(session_id):
    """Delete a training session."""
    try:
        session_file = HISTORY_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Session not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/history/compare', methods=['POST'])
def compare_sessions():
    """Compare multiple training sessions."""
    try:
        data = request.json
        session_ids = data.get('session_ids', [])

        if not session_ids:
            return jsonify({"success": False, "error": "No session IDs provided"}), 400

        comparison = []
        for session_id in session_ids:
            session_file = HISTORY_DIR / f"{session_id}.json"
            if session_file.exists():
                with open(session_file) as f:
                    session_data = json.load(f)
                    comparison.append(session_data)

        return jsonify({"sessions": comparison})
    except Exception as e:
        logger.error(f"Error comparing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Backtest Endpoints ───────────────────────────────────────────────────────

def backtest_worker(model_path, params):
    """Background worker for backtesting."""
    global backtest_state
    try:
        engine = BacktestEngine(
            model_path=model_path,
            confidence_threshold=params['confidence_threshold'],
            take_profit_bps=params['take_profit_bps'],
            stop_loss_bps=params['stop_loss_bps'],
            max_hold_periods=params.get('max_hold_periods', 20),
            commission_rate=params['commission_rate'],
            device=params.get('device', 'cuda')
        )

        result = engine.run_backtest(test_data_path=params.get('test_data_path'))

        from dataclasses import asdict
        backtest_state["result"] = asdict(result)
        backtest_state["active"] = False

        logger.info(f"Backtest completed. Win rate: {result.win_rate:.2%}")

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        backtest_state["result"] = {"error": str(e)}
        backtest_state["active"] = False


@app.route('/api/backtest/start', methods=['POST'])
def start_backtest():
    """Start backtest with specified parameters."""
    global backtest_state

    if backtest_state["active"]:
        return jsonify({
            "success": False,
            "error": "Backtest already in progress"
        }), 400

    try:
        data = request.json
        model_path = data.get('model_path')

        if not model_path or not Path(model_path).exists():
            return jsonify({
                "success": False,
                "error": "Invalid model path"
            }), 400

        params = {
            'confidence_threshold': data.get('confidence_threshold', 0.6),
            'take_profit_bps': data.get('take_profit_bps', 5.0),
            'stop_loss_bps': data.get('stop_loss_bps', 10.0),
            'max_hold_periods': data.get('max_hold_periods', 20),
            'commission_rate': data.get('commission_rate', 0.0004),
            'device': data.get('device', 'cuda'),
            'test_data_path': data.get('test_data_path')
        }

        backtest_state = {
            "active": True,
            "result": None,
            "thread": None
        }

        thread = threading.Thread(target=backtest_worker, args=(model_path, params), daemon=True)
        backtest_state["thread"] = thread
        thread.start()

        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/backtest/status', methods=['GET'])
def get_backtest_status():
    """Get current backtest status."""
    if backtest_state["active"]:
        return jsonify({"status": "running"})
    elif backtest_state["result"]:
        return jsonify({"status": "completed", "result": backtest_state["result"]})
    else:
        return jsonify({"status": "idle"})


@app.route('/api/backtest/result', methods=['GET'])
def get_backtest_result():
    """Get backtest result."""
    if backtest_state["result"]:
        return jsonify(backtest_state["result"])
    else:
        return jsonify({"success": False, "error": "No backtest result available"}), 404


# ── Model Checkpoint Endpoints ──────────────────────────────────────────────

@app.route('/api/models/list', methods=['GET'])
def list_model_checkpoints():
    """List all saved model checkpoints."""
    try:
        checkpoints = []
        for ckpt_file in CHECKPOINTS_DIR.glob("*.pt"):
            stat = ckpt_file.stat()
            checkpoints.append({
                "name": ckpt_file.stem,
                "path": str(ckpt_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify({"checkpoints": checkpoints})
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/models/<name>', methods=['DELETE'])
def delete_model_checkpoint(name):
    """Delete a model checkpoint."""
    try:
        ckpt_file = CHECKPOINTS_DIR / f"{name}.pt"
        if ckpt_file.exists():
            ckpt_file.unlink()
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Checkpoint not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting checkpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Inference Endpoints ──────────────────────────────────────────────────────

@app.route('/api/inference/load', methods=['POST'])
def load_inference_model():
    """Load a model for inference."""
    try:
        data = request.json
        model_path = data.get('model_path')
        model_name = data.get('model_name')
        device = data.get('device', 'cuda')

        if not model_path or not Path(model_path).exists():
            return jsonify({
                "success": False,
                "error": "Invalid model path"
            }), 400

        inference_service = get_inference_service()
        loaded_name = inference_service.load_model(model_path, model_name, device)

        return jsonify({
            "success": True,
            "model_name": loaded_name
        })
    except Exception as e:
        logger.error(f"Error loading model for inference: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/unload', methods=['POST'])
def unload_inference_model():
    """Unload a model from inference service."""
    try:
        data = request.json
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({
                "success": False,
                "error": "Model name required"
            }), 400

        inference_service = get_inference_service()
        inference_service.unload_model(model_name)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/models', methods=['GET'])
def list_loaded_models():
    """List all loaded models in inference service."""
    try:
        inference_service = get_inference_service()
        models = inference_service.get_loaded_models()
        active = inference_service.get_active_model()

        return jsonify({
            "models": models,
            "active_model": active
        })
    except Exception as e:
        logger.error(f"Error listing loaded models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/set-active', methods=['POST'])
def set_active_inference_model():
    """Set the active model for inference."""
    try:
        data = request.json
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({
                "success": False,
                "error": "Model name required"
            }), 400

        inference_service = get_inference_service()
        inference_service.set_active_model(model_name)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/predict', methods=['POST'])
def run_inference():
    """Run inference on input data."""
    try:
        data = request.json
        input_data = data.get('input_data')  # Expected: list of lists [[...], [...], ...]
        model_name = data.get('model_name')

        if input_data is None:
            return jsonify({
                "success": False,
                "error": "Input data required"
            }), 400

        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        inference_service = get_inference_service()
        predictions = inference_service.predict(input_tensor, model_name)

        return jsonify({
            "success": True,
            "predictions": predictions
        })
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/predict-batch', methods=['POST'])
def run_batch_inference():
    """Run inference on batch of input data."""
    try:
        data = request.json
        input_data = data.get('input_data')  # Expected: list of samples [[[...]], [[...]], ...]
        model_name = data.get('model_name')

        if input_data is None:
            return jsonify({
                "success": False,
                "error": "Input data required"
            }), 400

        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        inference_service = get_inference_service()
        predictions = inference_service.predict_batch(input_tensor, model_name)

        return jsonify({
            "success": True,
            "predictions": predictions
        })
    except Exception as e:
        logger.error(f"Error running batch inference: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/inference/model-info', methods=['GET'])
def get_inference_model_info():
    """Get information about loaded model."""
    try:
        model_name = request.args.get('model_name')

        inference_service = get_inference_service()
        info = inference_service.get_model_info(model_name)

        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
