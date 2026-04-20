# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a financial machine learning project focused on cryptocurrency trading prediction using time series models. The project uses transformer-based architectures (Crossformer, PatchTST) to predict trading signals from OHLC (Open, High, Low, Close) data with order book features.

## Architecture

### Core Components

**Data Pipeline** (`data_processing/dataset.py`):
- `OHLCDataset`: Processes cryptocurrency OHLC data with order book features (bid/ask volumes across 10 levels)
- Converts price movements to basis points (bps) for better numerical stability
- Extracts time features (month, day, weekday, hour) from timestamps
- Supports sliding window and sequential sampling modes
- Computes look-ahead reference values for profit calculation

**Models** (`models/`):
- `crossformer.py`: Custom Crossformer encoder with temporal and cross-channel attention
  - `CrossformerEncoderTimeStep`: Main encoder with multi-layer attention
  - `EmbeddingHead`: MLP-based prediction head with learnable channel weights, outputs tanh-bounded confidence scores
  - `AttentionPooling`: Weighted pooling across channels and time with configurable decay
- `PatchTST.py`: Example implementation using HuggingFace's PatchTST for binary classification

**Loss Functions** (`models/loss.py`):
- `ProfitLoss`: Custom loss function that optimizes for trading profit
  - Uses weighted cumulative profit across look-ahead windows
  - Includes stop-loss penalty mechanism
  - Applies reward shaping with tanh normalization

**Utility Functions** (`models/functions.py`):
- `power_distance`: Distance metric using power functions
- `amplified_atanh`: Scales [-1,1] inputs to larger ranges while preventing numerical instability

**Google Cloud Integration** (`config/googleCloud.py`):
- Functions to download Binance historical data from GCS buckets
- Supports both file download and direct DataFrame loading from Parquet files

### Data Flow

1. Raw OHLC + order book data → `OHLCDataset`
2. Feature engineering: price gaps in bps, bid/ask imbalance, volume features
3. Optional Z-score normalization for volume-related features
4. Sliding window sampling with look-ahead references
5. Model processes [Batch, Length, Channels] → embedding → confidence score
6. Custom profit-based loss function optimizes for trading performance

## Key Design Patterns

- **Basis Points (bps)**: All price movements are scaled to bps (×1000) for numerical stability
- **Look-ahead References**: Dataset pre-computes future price movements for profit calculation
- **Dual Attention**: Crossformer uses both temporal (within-channel) and cross-channel attention
- **Profit-Oriented Loss**: Loss function directly optimizes expected profit with stop-loss penalties
- **Device-Aware Dataset**: Dataset can be initialized directly on GPU for faster training

## Important Notes

- The project uses PyTorch with CUDA support (device='cuda' by default)
- Training script is embedded in `crossformer.py` (line 192-252)
- Data path: `data/Binance_BTC_USDT_USDT_3m.csv` (3-minute OHLC data)
- Google Cloud credentials expected at `config/google_cloud_key.json`
- Comments and variable names mix English and Chinese
