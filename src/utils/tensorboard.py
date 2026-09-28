from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:

    def __init__(self, log_dir="runs"):

        Path(log_dir).mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):

        self.writer.add_scalar(tag, value, step)

    def close(self):

        self.writer.close()