import json
from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # Resource directory
    resources_dir: str = ""

    # Model
    model: str = "encoder"

    # Test case
    conductivity: str = "circle"
    scale_noise: float = 5e-3

    # Optimisation
    alpha: float = 1e-3
    epochs: int = 100
    learning_rate: float = 1e-3

    def __post_init__(self):

        assert self.model in {"encoder", "cnn"}
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
