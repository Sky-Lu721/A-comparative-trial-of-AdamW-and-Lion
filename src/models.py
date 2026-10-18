"""CIFAR-10 model architecture."""
import torch
from torchvision.models import resnet18


def create_model() -> torch.nn.Module:
    """Create an unpretrained ResNet-18 with a CIFAR-sized stem."""
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model
