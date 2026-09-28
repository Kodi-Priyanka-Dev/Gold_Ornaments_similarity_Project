import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights

from config import NUM_CLASSES, DEVICE


def get_model():
    """
    Load a pretrained ConvNeXt-Base model and modify
    the classifier to generate embedding vectors.
    """

    # Load pretrained ConvNeXt-Base
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = models.convnext_base(weights=weights)

    # Freeze all layers (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    in_features = model.classifier[2].in_features

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512)
    )

    # Wrap the model in a custom class to add L2 normalization
    class NormalizedModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, x):
            features = self.base(x)
            return torch.nn.functional.normalize(features, p=2, dim=1)

    model = NormalizedModel(model)

    # Train only the new classifier
    for param in model.base.classifier.parameters():
        param.requires_grad = True

    # Move model to CPU or GPU
    model = model.to(DEVICE)

    return model


if __name__ == "__main__":

    model = get_model()

    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")