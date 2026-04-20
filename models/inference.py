"""
Inference module for trained Crossformer models.
Loads model checkpoint and runs predictions on new data.
"""
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from data_processing.dataset import OHLCDataset, preprocess_dataframe
from models.crossformer_lib.encoder import Encoder as CrossformerEncoder
from models.task_heads import MultiTaskHead
from controller.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Inference wrapper for trained Crossformer models.

    Handles model loading, data preprocessing, and prediction generation.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.encoder = None
        self.taskheads = None
        self.hyperparams = None

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract hyperparameters from checkpoint
        self.hyperparams = checkpoint.get('hyperparams', {})

        # Get model architecture parameters
        d_model = self.hyperparams.get('d_model', 64)
        n_heads = self.hyperparams.get('n_heads', 8)
        n_layers = self.hyperparams.get('n_layers', 4)
        dim_feedforward = self.hyperparams.get('dim_feedforward', 128)
        hidden_dim = self.hyperparams.get('hidden_dim', 128)
        dropout = self.hyperparams.get('dropout', 0.1)
        pooling = self.hyperparams.get('pooling', 'AttentionPooling')
        aggregation_level = self.hyperparams.get('aggregation_level', 4)
        num_tsa_layer = self.hyperparams.get('num_tsa_layer', 1)
        router = self.hyperparams.get('router', False)
        factor = self.hyperparams.get('factor', 1)
        seq_len = self.hyperparams.get('seq_len', 64)
        num_look_ahead = self.hyperparams.get('num_look_ahead', 10)

        # Reconstruct encoder
        self.encoder = CrossformerEncoder(
            num_encoder_layer=n_layers,
            aggregation_level=aggregation_level,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=dim_feedforward,
            num_tsa_layer=num_tsa_layer,
            dropout=dropout,
            total_seg_num=seq_len,
            factor=factor,
            router=router
        ).to(self.device)

        # Reconstruct task heads
        self.taskheads = MultiTaskHead(
            heads=['reversal', 'support', 'resistance'],
            d_model=d_model,
            pred_len=num_look_ahead,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
            n_heads=n_heads,
            d_layers=n_layers
        ).to(self.device)

        # Load weights
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.taskheads.load_state_dict(checkpoint['head'])

        # Set to evaluation mode
        self.encoder.eval()
        self.taskheads.eval()

        logger.info(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 'unknown'):.6f}")

    @torch.no_grad()
    def predict(self, batch_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference on a batch of data.

        Args:
            batch_data: Input tensor [B, L, C]

        Returns:
            Dictionary with predictions:
                - reversal: [B, 1] reversal confidence in [-1, 1]
                - support: [B, 1] support level prediction
                - resistance: [B, 1] resistance level prediction
        """
        batch_data = batch_data.to(self.device)

        # Forward pass
        embedding = self.encoder(batch_data)
        predictions = self.taskheads(embedding)

        return predictions

    @torch.no_grad()
    def predict_dataset(self, dataset: OHLCDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on entire dataset.

        Args:
            dataset: OHLCDataset instance

        Returns:
            Tuple of (reversal_signals, support_levels, resistance_levels, future_prices)
                - reversal_signals: [N] array of reversal confidence scores
                - support_levels: [N] array of predicted support levels
                - resistance_levels: [N] array of predicted resistance levels
                - future_prices: [N, L] array of future price changes
        """
        reversal_signals = []
        support_levels = []
        resistance_levels = []

        logger.info(f"Running inference on {len(dataset)} samples...")

        for i in range(len(dataset)):
            batch_data, reference_k, _ = dataset[i]
            batch_data = batch_data.unsqueeze(0)  # [1, L, C]

            predictions = self.predict(batch_data)

            reversal_signals.append(predictions['reversal'].squeeze().cpu().item())
            support_levels.append(predictions['support'].squeeze().cpu().item())
            resistance_levels.append(predictions['resistance'].squeeze().cpu().item())

        reversal_signals = np.array(reversal_signals)
        support_levels = np.array(support_levels)
        resistance_levels = np.array(resistance_levels)

        # Extract future prices from dataset
        future_prices = dataset.reference_k.cpu().numpy()  # [N, L, K]

        logger.info(f"Inference complete. Reversal range: [{reversal_signals.min():.3f}, {reversal_signals.max():.3f}]")

        return reversal_signals, support_levels, resistance_levels, future_prices

    def get_model_info(self) -> Dict:
        """Get model information and hyperparameters."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "hyperparams": self.hyperparams,
            "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
            "taskheads_params": sum(p.numel() for p in self.taskheads.parameters()),
        }


def load_model_for_inference(checkpoint_path: str, device: str = 'cuda') -> ModelInference:
    """
    Convenience function to load a model for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on

    Returns:
        ModelInference instance
    """
    return ModelInference(checkpoint_path, device)
