"""Data loading and preprocessing for Food-101 dataset."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class Food101Dataset(Dataset):
    """PyTorch Dataset wrapper for Food-101 Hugging Face dataset.

    Args:
        hf_dataset: Hugging Face dataset object
        transform: Image transformation pipeline
    """

    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        label = item["label"]

        image = self.transform(image)
        return image, label


def get_transforms(img_size: int = 224) -> transforms.Compose:
    """Get image transformation pipeline.

    Args:
        img_size: Target image size (square)

    Returns:
        Composed transforms for preprocessing images
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])


def load_food101(
    split: str,
    cache_dir: str = "data",
    use_subset: bool = False,
    subset_ratio: float = 0.1,
    max_samples_per_class: Optional[int] = None
):
    """Load Food-101 dataset from Hugging Face.

    Args:
        split: Dataset split ('train' or 'validation')
        cache_dir: Directory to cache the dataset
        use_subset: If True, use only a subset of the data for quick testing
        subset_ratio: Fraction of data to use if use_subset is True
        max_samples_per_class: Maximum samples per class (None = no limit)

    Returns:
        Hugging Face dataset object
    """
    logger.info(f"Loading Food-101 {split} split from cache_dir={cache_dir}")
    ds = load_dataset("ethz/food101", split=split, cache_dir=cache_dir)

    # Apply subset if requested (for quick testing)
    if use_subset:
        subset_size = int(len(ds) * subset_ratio)
        logger.info(f"Using subset: {subset_size} samples (ratio={subset_ratio})")
        ds = ds.select(range(subset_size))

    # Limit samples per class if specified
    if max_samples_per_class is not None:
        logger.info(f"Limiting to {max_samples_per_class} samples per class")
        # Group by label and take max_samples_per_class from each
        from collections import defaultdict
        indices_by_label = defaultdict(list)
        for idx, item in enumerate(ds):
            indices_by_label[item["label"]].append(idx)

        selected_indices = []
        for label_indices in indices_by_label.values():
            selected_indices.extend(label_indices[:max_samples_per_class])

        ds = ds.select(sorted(selected_indices))
        logger.info(f"Selected {len(ds)} samples total")

    logger.info(f"Loaded {len(ds)} samples")
    return ds


def get_dataloaders(
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    cache_dir: str = "data",
    use_subset: bool = False,
    subset_ratio: float = 0.1,
    max_samples_per_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache the dataset
        use_subset: If True, use only a subset of the data for quick testing
        subset_ratio: Fraction of data to use if use_subset is True
        max_samples_per_class: Maximum samples per class (None = no limit)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = get_transforms(img_size)

    train_ds = Food101Dataset(
        load_food101(
            "train",
            cache_dir=cache_dir,
            use_subset=use_subset,
            subset_ratio=subset_ratio,
            max_samples_per_class=max_samples_per_class
        ),
        transform
    )
    val_ds = Food101Dataset(
        load_food101(
            "validation",
            cache_dir=cache_dir,
            use_subset=use_subset,
            subset_ratio=subset_ratio,
            max_samples_per_class=max_samples_per_class
        ),
        transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(
        f"Created dataloaders: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches"
    )

    return train_loader, val_loader


def main():
    """CLI entry point for downloading and preprocessing data."""
    parser = argparse.ArgumentParser(description="Download and preprocess Food-101 dataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Directory to cache the dataset"
    )
    parser.add_argument(
        "--use-subset",
        action="store_true",
        help="Use only a subset of the data for quick testing"
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use if --use-subset is set"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Downloading Food-101 dataset...")
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download both splits
    logger.info("Downloading train split...")
    load_food101("train", cache_dir=args.cache_dir, use_subset=args.use_subset, subset_ratio=args.subset_ratio)

    logger.info("Downloading validation split...")
    load_food101("validation", cache_dir=args.cache_dir, use_subset=args.use_subset, subset_ratio=args.subset_ratio)

    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()
