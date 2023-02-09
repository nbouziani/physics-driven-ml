import os
import torch
import torch.nn.functional as F

from torch.nn import Module, Flatten, Linear
from training.utils import TrainingConfig


class EncoderDecoder(Module):
    """Build a simple toy model"""

    def __init__(self, n):
        super(EncoderDecoder, self).__init__()
        self.n1 = n
        self.n2 = int(n/2)
        self.flatten = Flatten()
        self.encoder_1 = Linear(self.n1, self.n2)
        self.decoder_1 = Linear(self.n2, self.n1)

    def encode(self, x):
        return self.encoder_1(x)

    def decode(self, x):
        return self.decoder_1(x)

    def forward(self, x):
        # [batch_size, n]
        x = self.flatten(x)
        # [batch_size, n2]
        encoded = self.encode(x)
        hidden = F.relu(encoded)
        # [batch_size, n]
        decoded = self.decode(hidden)
        return F.relu(decoded)

    @classmethod
    def from_pretrained(cls, model_dir: str):
        # Load training args
        training_args = os.path.join(model_dir, "training_args.json")
        training_args = TrainingConfig.from_file(training_args)
        # Instantiate model
        model = cls(training_args.input_shape)
        # Load pre-trained model state dict
        pretrained_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=torch.device("cpu"))
        model.load_state_dict(pretrained_dict)
        return model