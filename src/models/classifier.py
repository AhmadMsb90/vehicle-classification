import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def get_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    return model

