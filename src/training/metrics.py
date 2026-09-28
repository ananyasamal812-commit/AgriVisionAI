import torch


class Accuracy:

    def __call__(self, outputs, labels):

        predictions = outputs.argmax(dim=1)

        correct = (predictions == labels).sum().item()

        return correct / labels.size(0)