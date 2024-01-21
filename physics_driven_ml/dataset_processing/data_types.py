from typing import NamedTuple, List, Optional
from firedrake import Function
from torch import Tensor


class BatchElement(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    u_obs: Tensor  # shape = (n,)
    target: Tensor  # shape = (m,)
    u_obs_fd: Function
    target_fd: Function


class BatchedElement(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    u_obs: Tensor  # shape = (batch_size, n)
    target: Tensor  # shape = (batch_size, m)
    u_obs_fd: List[Function]
    target_fd: List[Function]
    batch_elements: Optional[List[BatchElement]] = None


class GraphBatchElement(NamedTuple):
    """Batch element for PDE-based datasets as a tuple of PyTorch and Firedrake tensors."""
    edge_index: Tensor  # shape = (2, num_edges)
    u: Tensor  # shape = (n,)
    f: Tensor  # shape = (m,)
    u_fd: Function
    f_fd: Function


class GraphBatchedElement(NamedTuple):
    """Represent tensors for a list/batch of `BatchElement` that have been collated."""
    edge_index: Tensor  # shape = (2, num_edges)
    u: Tensor  # shape = (batch_size, n)
    f: Tensor  # shape = (batch_size, m)
    u_fd: List[Function]
    f_fd: List[Function]
    batch_elements: Optional[List[BatchElement]] = None
