from torch.nn import Module, Sequential, ReLU, Tanh, MaxPool2d, Conv2d, ConvTranspose2d


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
