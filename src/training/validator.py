import torch
from tqdm import tqdm


class Validator:

    def __init__(self, model, criterion, device):

        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc="Validation"):

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        avg_loss = total_loss / len(loader)

        accuracy = correct / total

        return avg_loss, accuracy