import os
import torch
from torch.nn import Module, Sequential, Linear, ReLU, Tanh, MaxPool2d, Conv2d, ConvTranspose2d, BatchNorm2d


class CNN(Module):
    def __init__(self, dim):
        super(CNN, self).__init__()

        self.dim = dim
        self.n = int(self.dim ** 0.5)

        self.linear_in = Linear(self.dim, self.n**2)
        self.linear_in2 = Linear(self.n**2, self.n**2)
        self.linear_out2 = Linear(self.n*(self.n - 2)+1, self.n*(self.n - 2)+1) 
        self.linear_out = Linear(self.n*(self.n - 2)+1, self.dim)

        self.encoder = Sequential(
                            Conv2d(1, 32, kernel_size=4,
                                      padding=2),
                            BatchNorm2d(32),
                            ReLU(True),
                            MaxPool2d(2, 2),
                            Conv2d(32, 64, kernel_size=4,
                                      padding=1),
                            BatchNorm2d(64),
                            ReLU(True),
                            MaxPool2d(2, 2),
                            Conv2d(64, 128, kernel_size=3,
                                      padding=2),
                            ReLU(True))

        self.decoder = Sequential(
                            ConvTranspose2d(128, 64, kernel_size=4,
                                               padding=2),
                            BatchNorm2d(64),
                            ReLU(True),
                            ConvTranspose2d(64, 32, kernel_size=3,
                                               stride=2, padding=1),
                            BatchNorm2d(32),
                            ReLU(True),
                            ConvTranspose2d(32, 1, kernel_size=4, stride=2,
                                               padding=1),
                            Tanh())

    def forward(self, x):
        # x: [batch_size, dim]
        n = self.n
        # Reduce dimensionality to form a grid: x -> [batch_size, :, n, n]
        x = self.linear_in2(F.relu(self.linear_in(x)))
        cnn_input = x.reshape(-1, n, n)[:, None, :]
        # CNN
        encoded = self.encoder(cnn_input)
        decoded = self.decoder(encoded)
        # Recover dimensionality: [batch_size, :, n, n] -> [batch_size, dim]
        y = decoded.squeeze().reshape(-1, n*(n - 2)+1)
        y = self.linear_out(F.relu(self.linear_out2(y)))
        return y
