import os
import torch
import torch.nn.functional as F
from training.utils import ModelConfig
from torch.nn import Module, Sequential, Linear, ReLU, Tanh, MaxPool2d, Conv2d, ConvTranspose2d, BatchNorm2d, Dropout


class CNN(Module):
    """Build a simple toy cnn-based model"""

    def __init__(self, config: ModelConfig):
        super(CNN, self).__init__()

        self.dim = config.input_shape
        self.n = int(self.dim ** 0.5)
        self.m = self.n*(self.n - 2) + 1
        self.dropout = Dropout(p=config.dropout)
        self.linear_encoder = Linear(self.dim, self.n**2)
        self.hidden_e = Linear(self.n**2, self.n**2)
        self.hidden_d = Linear(self.m, self.m)
        self.linear_decoder = Linear(self.m, self.dim)

        self.cnn_encoder = Sequential(Conv2d(1, 32, kernel_size=4, padding=2),
                                      BatchNorm2d(32),
                                      ReLU(True),
                                      MaxPool2d(2, 2),
                                      Conv2d(32, 64, kernel_size=4, padding=1),
                                      BatchNorm2d(64),
                                      ReLU(True),
                                      MaxPool2d(2, 2),
                                      Conv2d(64, 128, kernel_size=3, padding=2),
                                      ReLU(True))

        self.cnn_decoder = Sequential(ConvTranspose2d(128, 64, kernel_size=4, padding=2),
                                      BatchNorm2d(64),
                                      ReLU(True),
                                      ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
                                      BatchNorm2d(32),
                                      ReLU(True),
                                      ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                      Tanh())

    def forward(self, x):
        # x: [batch_size, dim]
        # Reduce dimensionality to form a grid: x -> [batch_size, :, n, n]
        xh = F.relu(self.linear_encoder(x))
        xh = self.hidden_e(self.dropout(xh))
        x_grid = xh.reshape(-1, self.n, self.n)[:, None, :]
        # CNN encoder-decoder
        z = self.cnn_encoder(x_grid)
        y_grid = self.cnn_decoder(z)
        # Recover dimensionality: [batch_size, :, n, n] -> [batch_size, dim]
        yh = y_grid.squeeze().reshape(-1, self.m)
        yh = F.relu(self.hidden_d(yh))
        y = self.linear_decoder(self.dropout(yh))
        return y

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
