"""Vision Transformer model for Food-101 classification."""

import torch
import torch.nn as nn
from transformers import ViTModel


class ViTFoodClassifier(nn.Module):
    """Vision Transformer classifier for Food-101 dataset.

    Uses a pretrained ViT model and adds a custom classification head
    for 101 food categories.

    Args:
        num_classes: Number of output classes (default: 101 for Food-101)
    """

    def __init__(self, num_classes=101):
        super().__init__()

        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # Add custom classification head
        self.classifier = nn.Linear(
            self.vit.config.hidden_size,
            num_classes
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=x)
        # Extract CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0]
        # Classify
        logits = self.classifier(cls_embedding)
        return logits


def build_model(num_classes=101):
    """Build a ViT-based Food-101 classifier model.

    Args:
        num_classes: Number of output classes (default: 101 for Food-101)

    Returns:
        ViTFoodClassifier model instance
    """
    return ViTFoodClassifier(num_classes)
