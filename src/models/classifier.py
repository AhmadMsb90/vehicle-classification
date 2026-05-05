import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# builds a resnet18 model with frozen layers and a new classification head
def build_resnet(num_classes):
    # load default pretrained weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # freeze all backbone parameters
    for p in model.parameters():
        p.requires_grad = False

    # replace final fully connected layer with new classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # return configured model
    return model
