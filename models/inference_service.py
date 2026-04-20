"""
Real-time inference API for live trading predictions.
Provides endpoints for single-sample and batch predictions.
"""
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from models.inference import ModelInference

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for managing model inference in production.

    Handles model loading, caching, and prediction generation.
    """

    def __init__(self):
        self.models: Dict[str, ModelInference] = {}
        self.active_model: Optional[str] = None

    def load_model(self, model_path: str, model_name: Optional[str] = None, device: str = 'cuda') -> str:
        """
        Load a model for inference.

        Args:
            model_path: Path to model checkpoint
            model_name: Optional name for the model (defaults to filename)
            device: Device to run on

        Returns:
            Model name/identifier
        """
        if model_name is None:
            model_name = Path(model_path).stem

        logger.info(f"Loading model '{model_name}' from {model_path}")

        self.models[model_name] = ModelInference(model_path, device=device)
        self.active_model = model_name

        logger.info(f"Model '{model_name}' loaded successfully")
        return model_name

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model '{model_name}' unloaded")

            if self.active_model == model_name:
                self.active_model = None

    def set_active_model(self, model_name: str):
        """Set the active model for predictions."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")

        self.active_model = model_name
        logger.info(f"Active model set to '{model_name}'")

    def predict(self, input_data: torch.Tensor, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Run prediction on input data.

        Args:
            input_data: Input tensor [L, C] or [B, L, C]
            model_name: Model to use (defaults to active model)

        Returns:
            Dictionary with predictions
        """
        if model_name is None:
            model_name = self.active_model

        if model_name is None or model_name not in self.models:
            raise ValueError("No active model set")

        model = self.models[model_name]

        # Ensure batch dimension
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)  # [1, L, C]

        predictions = model.predict(input_data)

        # Convert to dict with scalar values
        result = {
            'reversal_confidence': predictions['reversal'].squeeze().cpu().item(),
            'support_level': predictions['support'].squeeze().cpu().item(),
            'resistance_level': predictions['resistance'].squeeze().cpu().item(),
        }

        return result

    def predict_batch(self, input_data: torch.Tensor, model_name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Run prediction on batch of data.

        Args:
            input_data: Input tensor [B, L, C]
            model_name: Model to use (defaults to active model)

        Returns:
            Dictionary with lists of predictions
        """
        if model_name is None:
            model_name = self.active_model

        if model_name is None or model_name not in self.models:
            raise ValueError("No active model set")

        model = self.models[model_name]
        predictions = model.predict(input_data)

        result = {
            'reversal_confidence': predictions['reversal'].squeeze(-1).cpu().tolist(),
            'support_level': predictions['support'].squeeze(-1).cpu().tolist(),
            'resistance_level': predictions['resistance'].squeeze(-1).cpu().tolist(),
        }

        return result

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())

    def get_active_model(self) -> Optional[str]:
        """Get the active model name."""
        return self.active_model

    def get_model_info(self, model_name: Optional[str] = None) -> Dict:
        """Get information about a model."""
        if model_name is None:
            model_name = self.active_model

        if model_name is None or model_name not in self.models:
            raise ValueError("No active model set")

        return self.models[model_name].get_model_info()


# Global inference service instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get the global inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
