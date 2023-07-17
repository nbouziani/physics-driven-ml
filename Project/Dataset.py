from torch.utils.data import Dataset
import numpy as np
import torch


class Data(Dataset):
    def __init__(self, X, y):
        self.features, self.labels = X, y

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float)
