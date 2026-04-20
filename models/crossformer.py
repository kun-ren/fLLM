import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_processing.dataset import OHLCDataset, preprocess_dataframe
from controller.config_manager import get_config_manager
from models.crossformer_lib.embeding import DSW_embedding
from models.crossformer_lib.encoder import Encoder as CrossformerEncoder
from models.reversal_loss import ReversalLoss
from models.task_heads import MultiTaskHead


# -----------------------------
# Training
# -----------------------------
def train_crossformer_rl(run_id=None):
    """
    Train Crossformer model with ReversalLoss.
    Yields dict with: status, log, progress, loss
    Pure training function - no worker or store dependencies.

    Args:
        run_id: Optional run identifier for checkpoint naming
    """
    config = get_config_manager()

    save_dir = Path(config.get("save_dir").value)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data params
    batch_size = config.get("batch_size").value
    seq_len = config.get("seq_len").value
    num_look_ahead = config.get("num_look_ahead").value

    # Model params
    d_model = config.get("d_model").value
    n_heads = config.get("n_heads").value
    n_layers = config.get("n_layers").value
    dim_feedforward = config.get("dim_feedforward").value
    hidden_dim = config.get("hidden_dim").value
    dropout = config.get("dropout").value
    pooling = config.get("pooling").value
    aggregation_level = config.get("aggregation_level").value
    num_tsa_layer = config.get("num_tsa_layer").value
    router = config.get("router").value
    factor = config.get("factor").value

    # Optimizer params
    lr = config.get("lr").value
    betas = tuple(float(x.strip()) for x in config.get("betas").value.split(","))
    eps = config.get("eps").value
    weight_decay = config.get("weight_decay").value

    # Training params
    epochs = config.get("epochs").value
    device = config.get("device").value

    # Loss params
    tp_bps = config.get("loss_take_profit_bps").value
    sl_bps = config.get("loss_stop_loss_bps").value

    print(f"Model: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    data, close_col = preprocess_dataframe()
    dataset = OHLCDataset(data, close_col, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    C = dataset[0][0].shape[-1]

    yield {
        "status": "initializing",
        "log": f"Channels: {C}, Samples: {len(dataset)}",
        "progress": 0.0,
        "loss": 0.0,
        "total_epochs": epochs,
    }

    encoder = CrossformerEncoder(
        num_encoder_layer=n_layers,
        aggregation_level=aggregation_level,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=dim_feedforward,
        num_tsa_layer=num_tsa_layer,
        dropout=dropout,
        total_seg_num=len(loader),
        factor=factor,
        router=router
    ).to(device)

    dsw_embedding = DSW_embedding(seq_len, d_model)

    taskheads = MultiTaskHead(
        heads=['reversal', 'support', 'resistance'],
        d_model=d_model,
        pred_len=num_look_ahead,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling=pooling,
        n_heads=n_heads,
        d_layers=n_layers
    )

    loss_fn = ReversalLoss(L=num_look_ahead)

    optimizer = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": lr},
         {"params": taskheads.parameters(), "lr": lr}],
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    encoder.train()
    taskheads.train()
    global_step = 0

    best_loss = float("inf")
    best_epoch = 0
    checkpoint_path = None

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for step_idx, (batch_data, reference_k, volatility_50) in enumerate(loader):
            batch_data = batch_data.to(device)
            reference_k = reference_k.to(device)

            x_embeded = dsw_embedding(batch_data)
            embedding = encoder(x_embeded)
            out = taskheads(embedding)
            loss, _ = loss_fn(out, reference_k, volatility_50)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(taskheads.parameters()), max_norm=1.0
            )
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            checkpoint_name = f"run_{run_id}_e{epoch}.pt" if run_id else f"checkpoint_e{epoch}.pt"
            checkpoint_path = save_dir / checkpoint_name
            torch.save(
                {"encoder": encoder.state_dict(),
                 "head": taskheads.state_dict(),
                 "epoch": epoch,
                 "loss": avg_loss,
                 "hyperparams": config.resolve_all()},
                checkpoint_path,
            )
            logging.info(msg=f"  -> New best loss: {avg_loss:.6f} (epoch {epoch})")

        # Yield progress update with only current progress and loss
        yield {
            "status": "training",
            "log": f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}",
            "progress": epoch / epochs,
            "loss": avg_loss,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        }

        scheduler.step()

    logging.info(msg=f"Training complete. Best loss: {best_loss:.6f} at epoch {best_epoch}")

    # Final completion report with all metrics
    yield {
        "status": "completed",
        "log": f"Training complete. Best loss: {best_loss:.6f} at epoch {best_epoch}",
        "progress": 1.0,
        "loss": avg_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


# Alias for backward compatibility
train_crossformer = train_crossformer_rl

if __name__ == "__main__":
    for progress in train_crossformer():
        print(progress["log"])
