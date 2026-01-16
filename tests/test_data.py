"""Tests for data loading and preprocessing."""

import torch
from torch.utils.data import Dataset

from mlops_project.data import Food101Dataset, get_dataloaders, get_transforms, load_food101


def test_food101_dataset():
    """Test the Food101Dataset class."""
    # Load a small subset for testing
    hf_dataset = load_food101("train", cache_dir="data", use_subset=True, subset_ratio=0.01)
    transform = get_transforms(img_size=224)
    dataset = Food101Dataset(hf_dataset, transform)

    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0

    # Test getting an item
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 101


def test_get_transforms():
    """Test image transformation pipeline."""
    transform = get_transforms(img_size=224)
    assert transform is not None

    # Test that it's a Compose object
    from torchvision import transforms
    assert isinstance(transform, transforms.Compose)


def test_load_food101():
    """Test loading Food-101 dataset."""
    # Test loading with subset
    dataset = load_food101("train", cache_dir="data", use_subset=True, subset_ratio=0.01)
    assert len(dataset) > 0

    # Check dataset has expected fields
    item = dataset[0]
    assert "image" in item
    assert "label" in item


def test_get_dataloaders():
    """Test getting train and validation dataloaders."""
    train_loader, val_loader = get_dataloaders(
        batch_size=8,
        img_size=224,
        num_workers=0,  # Use 0 workers for testing
        cache_dir="data",
        use_subset=True,
        subset_ratio=0.01
    )

    assert train_loader is not None
    assert val_loader is not None

    # Test getting a batch
    images, labels = next(iter(train_loader))
    assert images.shape[0] <= 8  # batch_size
    assert images.shape[1] == 3  # RGB channels
    assert images.shape[2] == 224  # height
    assert images.shape[3] == 224  # width
    assert labels.shape[0] == images.shape[0]
