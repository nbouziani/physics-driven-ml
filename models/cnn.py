from torch.nn import Module, Sequential, Linear, ReLU, Tanh, MaxPool2d, Conv2d, ConvTranspose2d


class CNN(Module):
    def __init__(self, dim):
        super(CNN, self).__init__()

        self.dim = dim
        self.n = int(self.dim ** 0.5)

        self.linear_in = Linear(self.dim, self.n**2)
        self.linear_out = Linear(self.n*(self.n - 2)+1, self.dim)

        self.encoder = Sequential(
                            Conv2d(1, 64, kernel_size=3,
                                      padding=1),
                            ReLU(True),
                            MaxPool2d(2, 2),
                            Conv2d(64, 100, kernel_size=3,
                                      padding=1),
                            ReLU(True))

        self.decoder = Sequential(
                            ConvTranspose2d(100, 64, kernel_size=3,
                                               stride=1, padding=1),
                            ReLU(True),
                            ConvTranspose2d(64, 32, kernel_size=3,
                                               stride=2, padding=1),
                            ReLU(True),
                            ConvTranspose2d(32, 1, kernel_size=4,
                                               padding=1),
                            Tanh())

    def forward(self, x):
        # x: [batch_size, dim]
        n = self.n
        # Reduce dimensionality to form a grid: x -> [batch_size, :, n, n]
        x = self.linear_in(x)
        cnn_input = x.reshape(-1, n, n)[:, None, :]
        # CNN
        encoded = self.encoder(cnn_input)
        decoded = self.decoder(encoded)
        # Recover dimensionality: [batch_size, :, n, n] -> [batch_size, dim]
        y = decoded.squeeze().reshape(-1, n*(n - 2)+1)
        y = self.linear_out(y)
        return y
