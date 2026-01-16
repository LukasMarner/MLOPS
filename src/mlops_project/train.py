"""Training script for Food-101 classification model."""

import logging
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import wandb

from mlops_project.data import get_dataloaders
from mlops_project.model import build_model

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device for training.

    Args:
        device_config: Device configuration ('auto', 'cpu', or 'cuda')

    Returns:
        PyTorch device object
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)

    logger.info(f"Using device: {device}")
    return device


def get_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.

    Args:
        model: PyTorch model
        cfg: Training configuration

    Returns:
        Optimizer instance
    """
    if cfg.optimizer.lower() == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    logger.info(f"Using optimizer: {cfg.optimizer}")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, num_epochs: int):
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        cfg: Training configuration
        num_epochs: Total number of epochs

    Returns:
        Scheduler instance or None
    """
    if not cfg.use_scheduler:
        return None

    if cfg.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif cfg.scheduler.lower() == "step":
        scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    logger.info(f"Using scheduler: {cfg.scheduler}")
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metric: float,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch number
        loss: Current loss value
        metric: Current metric value
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metric": metric,
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved checkpoint to {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Path,
    device: torch.device
) -> int:
    """Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance (can be None)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Epoch number from checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_freq: int = 10,
    gradient_clip: Optional[float] = None
):
    """Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        optimizer: Optimizer instance
        device: Device to train on
        epoch: Current epoch number
        log_freq: Frequency of logging (every N batches)
        gradient_clip: Gradient clipping value (None = no clipping)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_freq == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if wandb.run is not None:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/epoch": epoch,
                    "train/batch": batch_idx
                })

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int
):
    """Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    setup_logging()
    logger.info("Starting training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Initialize W&B
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.info("Initialized Weights & Biases")

    # Get device
    device = get_device(cfg.training.device)

    # Create directories
    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=cfg.data.batch_size,
        img_size=cfg.data.img_size,
        num_workers=cfg.data.num_workers,
        cache_dir=cfg.data.cache_dir,
        use_subset=cfg.data.use_subset,
        subset_ratio=cfg.data.subset_ratio,
        max_samples_per_class=cfg.data.max_samples_per_class
    )

    # Build model
    logger.info("Building model...")
    model = build_model(num_classes=cfg.model.num_classes)
    model = model.to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = get_optimizer(model, cfg.training)

    # Create scheduler
    scheduler = get_scheduler(optimizer, cfg.training, cfg.training.num_epochs)

    # Load checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    if cfg.model.checkpoint_path is not None:
        checkpoint_path = Path(cfg.model.checkpoint_path)
        if checkpoint_path.exists():
            start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from scratch")

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            log_freq=cfg.training.log_freq,
            gradient_clip=cfg.training.gradient_clip
        )

        # Validate
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_loss, val_acc = validate(model, val_loader, device, epoch)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{cfg.training.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )

            if wandb.run is not None:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "train/epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })

            # Save checkpoint
            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc

            if cfg.model.save_checkpoint:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    val_loss,
                    val_acc,
                    Path(cfg.model.checkpoint_dir),
                    is_best=is_best
                )

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_metric:.2f}%")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
