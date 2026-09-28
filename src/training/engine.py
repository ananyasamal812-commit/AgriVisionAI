import torch
from tqdm import tqdm


class Engine:

    def __init__(self, model, optimizer, criterion, device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, loader):

        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for images, labels in tqdm(loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()

            total_samples += labels.size(0)

        return (
            total_loss / len(loader),
            total_correct / total_samples,
        )