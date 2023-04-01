import os
import numpy as np
import torch

from typing import List
from firedrake import CheckpointFile
from firedrake.ml import load_backend
from torch.utils.data import Dataset

from physics_driven_ml.dataset_processing import BatchElement, BatchedElement


class PDEDataset(Dataset):
    """Dataset reader for PDE-based datasets generated using Firedrake."""

    def __init__(self, dataset: str = "heat_conductivity", dataset_split: str = "train", data_dir: str = ""):
        # Check dataset directory
        dataset_dir = os.path.join(data_dir, "datasets", dataset)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")

        # Get mesh and batch elements (Firedrake functions)
        name_file = dataset_split + "_data.h5"
        mesh, batch_elements = self.load_dataset(os.path.join(dataset_dir, name_file))
        self.mesh = mesh
        self.batch_elements_fd = batch_elements

        # Get PyTorch backend from Firedrake (for mapping from Firedrake to PyTorch and vice versa)
        self._fd_backend = load_backend()

    def load_dataset(self, fname: str):
        data = []
        # Load data
        with CheckpointFile(fname, "r") as afile:
            n = int(np.array(afile.h5pyfile["n"]))
            # Load mesh
            mesh = afile.load_mesh("mesh")
            # Load data
            for i in range(n):
                k = afile.load_function(mesh, "k", idx=i)
                u_obs = afile.load_function(mesh, "u_obs", idx=i)
                data.append((k, u_obs))
        return mesh, data

    def __len__(self) -> int:
        return len(self.batch_elements_fd)

    def __getitem__(self, idx: int) -> BatchElement:
        target_fd, u_obs_fd = self.batch_elements_fd[idx]
        # Convert Firedrake functions to PyTorch tensors
        target, u_obs = [self._fd_backend.to_ml_backend(e) for e in [target_fd, u_obs_fd]]
        return BatchElement(target=target, u_obs=u_obs,
                            target_fd=target_fd, u_obs_fd=u_obs_fd)

    def collate(self, batch_elements: List[BatchElement]) -> BatchedElement:
        # Workaround to enable custom data types (e.g. firedrake.Function) in PyTorch dataloaders
        # See: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        batch_size = len(batch_elements)
        n = max(e.u_obs.size(-1) for e in batch_elements)
        m = max(e.target.size(-1) for e in batch_elements)

        u_obs = torch.zeros(batch_size, n, dtype=batch_elements[0].u_obs.dtype)
        target = torch.zeros(batch_size, m, dtype=batch_elements[0].target.dtype)
        target_fd = []
        u_obs_fd = []
        for i, e in enumerate(batch_elements):
            u_obs[i, :] = e.u_obs
            target[i, :] = e.target
            target_fd.append(e.target_fd)
            u_obs_fd.append(e.u_obs_fd)

        return BatchedElement(u_obs=u_obs, target=target,
                              target_fd=target_fd, u_obs_fd=u_obs_fd,
                              batch_elements=batch_elements)
