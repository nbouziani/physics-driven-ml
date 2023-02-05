import sys
import json
import logging
from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # Resource directory
    resources_dir: str = ""
    name_dir: str = "data"

    # Model
    model: str = "encoder-decoder"

    # Domain
    Lx: float = 1.0
    Ly: float = 1.0

    # Test case
    conductivity: str = "circle"
    scale_noise: float = 5e-3

    # Evaluation
    max_eval_steps: int = 5000

    # Dataset
    dataset: str = "poisson"
    ntrain: int = 30

    # Optimisation
    alpha: float = 1e-3
    epochs: int = 100
    learning_rate: float = 1e-3
    evaluation_metric: str = "L2"

    def __post_init__(self):

        assert self.model in {"encoder-decoder", "cnn"}
        assert self.conductivity in {"circle", "random"}

    @classmethod
    def from_file(cls, filepath: str, data_dir: str):
        with open(filepath, "r") as f:
            cfg = json.load(f)
            cfg["resources_dir"] = data_dir
            return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)


def get_logger(name: str = "main"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
