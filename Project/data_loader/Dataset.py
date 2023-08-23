from torch.utils.data import Dataset
import numpy as np
import torch


class Data(Dataset):
    def __init__(self, X, y):
        self.features, self.labels = X, y

    def __len__(self):
        return self.features.shape[0]

    def extract_upper_triangle(self, matrix):
        indices = np.triu_indices(matrix.shape[1])
        return matrix[:, indices[0], indices[1]]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.extract_upper_triangle(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.float)
