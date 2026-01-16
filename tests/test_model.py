"""Tests for model architecture."""

import torch

from mlops_project.model import ViTFoodClassifier, build_model


def test_build_model():
    """Test building the model."""
    model = build_model(num_classes=101)
    assert isinstance(model, ViTFoodClassifier)
    assert model is not None


def test_model_forward():
    """Test model forward pass."""
    model = build_model(num_classes=101)
    model.eval()

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, 101)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_model_parameters():
    """Test that model has trainable parameters."""
    model = build_model(num_classes=101)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0

    # Check that ViT backbone is loaded
    assert hasattr(model, "vit")
    assert hasattr(model, "classifier")


def test_model_different_num_classes():
    """Test building model with different number of classes."""
    model = build_model(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10)

