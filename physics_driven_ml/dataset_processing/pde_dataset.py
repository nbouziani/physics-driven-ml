import os
import numpy as np
import torch

from typing import List
from firedrake import CheckpointFile
from firedrake.ml.pytorch import *
from torch.utils.data import Dataset

from physics_driven_ml.dataset_processing import BatchElement, BatchedElement, GraphBatchElement, GraphBatchedElement


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
        target, u_obs = [to_torch(e) for e in [target_fd, u_obs_fd]]
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


class StokesDataset(PDEDataset):
    """Dataset reader for Stokes problems generated using Firedrake."""

    def __init__(self, dataset: str = "stokes_cylinder", dataset_split: str = "train", data_dir: str = ""):
        # Check dataset directory
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")

        # Get mesh and batch elements (Firedrake functions)
        name_file = dataset_split + "_data.h5"
        mesh, edge_index, batch_elements = self.load_dataset(os.path.join(dataset_dir, name_file))
        self.mesh = mesh
        self.edge_index = edge_index
        self.batch_elements_fd = batch_elements

    def load_dataset(self, fname: str):
        data = []
        # Load data
        with CheckpointFile(fname, "r") as afile:
            n = int(np.array(afile.h5pyfile["n"]))
            # Load adjacency list
            edge_index = np.array(afile.h5pyfile["edge_index"])
            # Load mesh
            mesh = afile.load_mesh()
            # Load data
            for i in range(n):
                f = afile.load_function(mesh, "f", idx=i)
                u = afile.load_function(mesh, "u", idx=i)
                data.append((f, u))
        return mesh, edge_index, data

    def __getitem__(self, idx: int) -> GraphBatchElement:
        target_f_fd, target_u_fd = self.batch_elements_fd[idx]
        # Convert Firedrake functions to PyTorch tensors
        target_f, target_u = [to_torch(e) for e in [target_f_fd, target_u_fd]]
        return GraphBatchElement(u=target_u, f=target_f,
                                 u_fd=target_u_fd, f_fd=target_f_fd)

    def collate(self, batch_elements: List[GraphBatchElement]) -> GraphBatchedElement:
        # Workaround to enable custom data types (e.g. firedrake.Function) in PyTorch dataloaders
        # See: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        batch_size = len(batch_elements)
        n = max(e.u.size(-1) for e in batch_elements)
        m = max(e.f.size(-1) for e in batch_elements)

        u = torch.zeros(batch_size, n, dtype=batch_elements[0].u.dtype)
        f = torch.zeros(batch_size, m, dtype=batch_elements[0].f.dtype)
        f_fd = []
        u_fd = []
        for i, e in enumerate(batch_elements):
            u[i, :] = e.u
            f[i, :] = e.f
            u_fd.append(e.u_fd)
            f_fd.append(e.f_fd)

        return GraphBatchedElement(u=u, f=f,
                                   u_fd=u_fd, f_fd=f_fd,
                                   batch_elements=batch_elements)
