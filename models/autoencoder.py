import os
import torch
import torch.nn.functional as F

from torch.nn import Module, Flatten, Linear
from training.utils import ModelConfig


class EncoderDecoder(Module):
    """Build a simple toy encoder-decoder model"""

    def __init__(self, config: ModelConfig):
        super(EncoderDecoder, self).__init__()
        self.n = config.input_shape
        self.m = int(self.n/2)
        self.flatten = Flatten()
        self.linear_encoder = Linear(self.n, self.m)
        self.linear_decoder = Linear(self.m, self.n)

    def encode(self, x):
        return F.relu(self.linear_encoder(x))

    def decode(self, x):
        return F.relu(self.linear_decoder(x))

    def forward(self, x):
        # [batch_size, n]
        x = self.flatten(x)
        # [batch_size, m]
        hidden = self.encode(x)
        # [batch_size, n]
        decoded = self.decode(hidden)
        return decoded

    @classmethod
    def from_pretrained(cls, model_dir: str):
        # Load training args
        training_args = os.path.join(model_dir, "training_args.json")
        config = ModelConfig.from_file(training_args)
        # Instantiate model
        model = cls(config)
        # Load pre-trained model state dict
        pretrained_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=torch.device("cpu"))
        model.load_state_dict(pretrained_dict)
        return model
