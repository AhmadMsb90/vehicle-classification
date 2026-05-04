import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloaders
from src.models.classifier import get_model
from src.training.trainer import Trainer
from src.config import LR, EPOCHS, MODEL_PATH

dataloaders, classes = get_dataloaders()

model = get_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

trainer = Trainer(model, dataloaders, criterion, optimizer)
trainer.train(EPOCHS)

torch.save(model.state_dict(), MODEL_PATH)