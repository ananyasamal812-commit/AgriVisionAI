from pathlib import Path

import yaml


class Config:

    def __init__(self, config_path="configs/config.yaml"):

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(config_path)

        with open(config_path, "r") as f:

            self.config = yaml.safe_load(f)

    def get(self, *keys):

        value = self.config

        for key in keys:
            value = value[key]

        return value

    def __getitem__(self, key):

        return self.config[key]