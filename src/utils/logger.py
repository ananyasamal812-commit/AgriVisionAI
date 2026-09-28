from pathlib import Path
import logging


class Logger:

    def __init__(

        self,

        name="AgriVisionAI",

        log_dir="logs",

        level=logging.INFO,

    ):

        Path(log_dir).mkdir(exist_ok=True)

        self.logger = logging.getLogger(name)

        self.logger.setLevel(level)

        if not self.logger.handlers:

            formatter = logging.Formatter(

                "%(asctime)s | %(levelname)s | %(message)s",

                datefmt="%Y-%m-%d %H:%M:%S",

            )

            file_handler = logging.FileHandler(

                Path(log_dir) / "training.log"

            )

            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()

            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

            self.logger.addHandler(console_handler)

    def info(self, message):

        self.logger.info(message)

    def warning(self, message):

        self.logger.warning(message)

    def error(self, message):

        self.logger.error(message)

    def debug(self, message):

        self.logger.debug(message)