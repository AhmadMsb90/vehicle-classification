import torch
from tqdm import tqdm
from src.config import DEVICE

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer):
        self.model = model.to(DEVICE)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self._train_epoch()
            self._validate()

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0

        loader = tqdm(self.dataloaders['train'], desc="Training")

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(loss=loss.item())

        print("Train Loss:", running_loss / len(self.dataloaders['train']))

    def _validate(self):
        self.model.eval()
        correct, total = 0, 0

        loader = tqdm(self.dataloaders['val'], desc="Validation")

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                loader.set_postfix(acc=100 * correct / total)

        print("Validation Accuracy:", 100 * correct / total)