from typing import List, Optional, Tuple
import random

import numpy as np
from tqdm import tqdm

from .base import BaseClusterAlgorithm

try:
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for ApproximateTorchClusterAlgorithm. "
            "Install the project dependencies with torch included."
        )


def _resolve_torch_dtype(dtype: str):
    _require_torch()
    if not hasattr(torch, dtype):
        raise ValueError(f"Unknown torch dtype: {dtype}")

    torch_dtype = getattr(torch, dtype)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"{dtype} is not a valid torch dtype")

    return torch_dtype


def _resolve_runtime_torch_dtype(device, dtype: str):
    torch_dtype = _resolve_torch_dtype(dtype)
    if device.type == "mps" and torch_dtype == torch.float64:
        return torch.float32
    return torch_dtype


def resolve_torch_device(device: str):
    _require_torch()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")

    return resolved


class ApproximateTorchClusterAlgorithm(BaseClusterAlgorithm):
    """
    Torch-backed approximate clustering algorithm with configurable device.
    """

    def __init__(
        self,
        eps: float = 0.1,
        r_target: int = 10,
        patience: int = 10,
        verbose: bool = False,
        random_seed: int = 23,
        tol: float = 1e-12,
        oversampling: int = 5,
        sorting_strategy: str = "norm",
        device: str = "auto",
        torch_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sorting_strategy not in ["norm", "residual"]:
            raise ValueError("sorting_strategy must be either 'norm' or 'residual'")

        self.eps = eps
        self.r_target = r_target
        self.patience = patience
        self.verbose = verbose
        self.random_seed = random_seed
        self.tol = tol
        self.oversampling = oversampling
        self.sorting_strategy = sorting_strategy
        self.device = device
        self.torch_dtype = torch_dtype
        self.resolved_device_ = None
        self.resolved_torch_dtype_ = None
        self.clusters_: Optional[List[List[np.ndarray]]] = None

    def _to_tensor(self, block: np.ndarray):
        return torch.as_tensor(
            np.asarray(block),
            device=self.resolved_device_,
            dtype=self.resolved_torch_dtype_,
        )

    def _update_top_singular_values(
        self,
        A_new,
        Q=None,
        S=None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if Q is None or S is None:
            Q0, R0 = torch.linalg.qr(A_new, mode="reduced")
            S0 = R0 @ R0.T

            w, U = torch.linalg.eigh(S0)
            w = torch.clamp(w, min=0.0)

            pos = w > self.tol
            if not torch.any(pos):
                return torch.zeros(
                    0,
                    device=self.resolved_device_,
                    dtype=self.resolved_torch_dtype_,
                ), None, None

            w_pos = w[pos]
            U_pos = U[:, pos]

            idx = torch.argsort(w_pos, descending=True)
            w_sorted = w_pos[idx]
            U_sorted = U_pos[:, idx]

            r_keep = min(len(w_sorted), self.r_target + self.oversampling)
            lam = w_sorted[:r_keep]
            U_top = U_sorted[:, :r_keep]

            Q_new = Q0 @ U_top
            S_new = torch.diag(lam)

            singular_vals = torch.sqrt(lam[: min(self.r_target, len(lam))])
            return singular_vals, Q_new, S_new

        r_curr = S.shape[0]

        Y = Q.T @ A_new
        A_in = Q @ Y
        R = A_new - A_in

        Q_res, B = torch.linalg.qr(R, mode="reduced")

        if B.numel() > 0:
            row_norms = torch.linalg.norm(B, dim=1)
            mask = row_norms > self.tol
            B = B[mask, :]
            Q_res = Q_res[:, mask]

        r_res = B.shape[0]

        if r_res == 0:
            Q_ext = Q
            S_ext = S + Y @ Y.T
        else:
            Q_ext = torch.hstack([Q, Q_res])

            S11 = S + Y @ Y.T
            S12 = Y @ B.T
            S22 = B @ B.T

            top = torch.hstack([S11, S12])
            bottom = torch.hstack([S12.T, S22])
            S_ext = torch.vstack([top, bottom])

        w, U = torch.linalg.eigh(S_ext)
        w = torch.clamp(w, min=0.0)

        pos = w > self.tol
        if not torch.any(pos):
            return torch.zeros(
                0,
                device=self.resolved_device_,
                dtype=self.resolved_torch_dtype_,
            ), None, None

        w_pos = w[pos]
        U_pos = U[:, pos]

        idx = torch.argsort(w_pos, descending=True)
        w_sorted = w_pos[idx]
        U_sorted = U_pos[:, idx]

        r_keep = min(len(w_sorted), self.r_target + self.oversampling)
        lam = w_sorted[:r_keep]
        U_top = U_sorted[:, :r_keep]

        Q_new = Q_ext @ U_top
        S_new = torch.diag(lam)

        singular_vals = torch.sqrt(lam[: min(self.r_target, len(lam))])
        return singular_vals, Q_new, S_new

    def _order_and_prepare_first_block(
        self,
        blocks: List[np.ndarray],
        tensor_blocks: List["torch.Tensor"],
        norms_blocks: "torch.Tensor",
        available_indices: List[int],
    ) -> tuple:
        if self.sorting_strategy == "norm":
            ordered_indices = available_indices
            get_next_idx = lambda k: ordered_indices[-(k + 1)]

            idx0 = ordered_indices[0]
            norm_M = norms_blocks[idx0] ** 2
            block0 = tensor_blocks[idx0]
            indeces_block0 = [idx0]
            i = 1
            while block0.shape[1] < self.r_target and i < len(ordered_indices):
                next_idx = ordered_indices[i]
                if blocks[next_idx].shape[1] + block0.shape[1] > self.r_target:
                    break
                block0 = torch.hstack([block0, tensor_blocks[next_idx]])
                indeces_block0.append(next_idx)
                norm_M += norms_blocks[next_idx] ** 2
                i += 1
            ordered_indices = ordered_indices[i:]

            singular_vals, Q, S = self._update_top_singular_values(block0)
        else:
            idx0 = available_indices[0]
            norm_M = norms_blocks[idx0] ** 2
            block0 = tensor_blocks[idx0]
            indeces_block0 = [idx0]
            i = 1
            while block0.shape[1] < self.r_target and i < len(available_indices):
                next_idx = available_indices[i]
                if blocks[next_idx].shape[1] + block0.shape[1] > self.r_target:
                    break
                block0 = torch.hstack([block0, tensor_blocks[next_idx]])
                indeces_block0.append(next_idx)
                norm_M += norms_blocks[next_idx] ** 2
                i += 1
            available_indices = available_indices[i:]

            singular_vals, Q, S = self._update_top_singular_values(block0)

            residual_info = []
            for i in available_indices:
                if i not in indeces_block0:
                    Y = Q.T @ tensor_blocks[i]
                    R = tensor_blocks[i] - Q @ Y
                    residual_info.append((torch.linalg.norm(R, ord="fro"), i))

            if residual_info:
                residual_norms = torch.stack([norm for norm, _ in residual_info])
                sort_order = torch.argsort(residual_norms)
                ordered_indices = [residual_info[i][1] for i in sort_order.tolist()]
            else:
                ordered_indices = []
            get_next_idx = lambda k: ordered_indices[k]

        return (indeces_block0, norm_M, Q, S, ordered_indices, get_next_idx)

    def _compute_max_k_and_order(
        self,
        blocks: List[np.ndarray],
        tensor_blocks: List["torch.Tensor"],
        norms_blocks: "torch.Tensor",
        available_indices: List[int],
    ) -> Tuple[int, List[int]]:
        (
            indeces_block0,
            norm_M,
            Q,
            S,
            ordered_indices,
            get_next_idx,
        ) = self._order_and_prepare_first_block(
            blocks, tensor_blocks, norms_blocks, available_indices
        )

        k = 0
        while k < len(ordered_indices):
            next_idx = get_next_idx(k)
            test_norm_M = norm_M + norms_blocks[next_idx] ** 2
            singular_vals, Q, S = self._update_top_singular_values(
                tensor_blocks[next_idx], Q=Q, S=S
            )
            lower_bound = torch.sum(singular_vals ** 2).item()

            error = (
                0.0
                if lower_bound >= test_norm_M
                else ((test_norm_M - lower_bound) / test_norm_M) ** 0.5
            )

            if self.verbose:
                print(f"Error: {error}")

            if error > self.eps:
                break

            norm_M = test_norm_M
            k += 1

        if self.sorting_strategy == "norm":
            cluster_indices = (
                indeces_block0 + ordered_indices[-k:] if k > 0 else indeces_block0
            )
        else:
            cluster_indices = indeces_block0 + ordered_indices[:k]

        return k, cluster_indices

    def _cluster_blocks(self, blocks: List[np.ndarray]) -> List[List[tuple]]:
        random.seed(self.random_seed)

        tensor_blocks = [self._to_tensor(block) for block in blocks]
        norms_blocks = torch.stack(
            [torch.linalg.norm(block, ord="fro") for block in tensor_blocks]
        )
        available_indices = torch.argsort(norms_blocks, descending=True).tolist()

        clusters = []
        i_cluster_ones = 0
        pbar = tqdm(total=len(available_indices), desc="Clustering blocks")

        while available_indices:
            if self.verbose:
                print(f"Remaining blocks: {len(available_indices)}")

            max_k, cluster_indices = self._compute_max_k_and_order(
                blocks, tensor_blocks, norms_blocks, available_indices
            )

            if max_k == 0:
                i_cluster_ones += 1
                if i_cluster_ones >= self.patience:
                    for idx in available_indices:
                        clusters.append([(idx, blocks[idx])])
                    pbar.update(len(available_indices))
                    if self.verbose:
                        print(
                            f"Patience {self.patience} reached. Stopping clustering."
                        )
                    break
            else:
                i_cluster_ones = 0

            cluster = [(idx, blocks[idx]) for idx in cluster_indices]
            clusters.append(cluster)
            pbar.update(len(cluster_indices))
            available_indices = [
                idx for idx in available_indices if idx not in cluster_indices
            ]

        pbar.close()
        return clusters

    def _assign_labels(
        self,
        original_blocks: List[np.ndarray],
        clusters_with_idx: List[List[tuple]],
    ) -> np.ndarray:
        labels = np.zeros(len(original_blocks), dtype=int)
        for cluster_idx, cluster_items in enumerate(clusters_with_idx):
            for orig_idx, _block in cluster_items:
                labels[orig_idx] = cluster_idx
        return labels

    def fit(self, X: List[np.ndarray]) -> "ApproximateTorchClusterAlgorithm":
        _require_torch()
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be a non-empty list of numpy arrays")

        self.resolved_device_ = resolve_torch_device(self.device)
        self.resolved_torch_dtype_ = _resolve_runtime_torch_dtype(
            self.resolved_device_, self.torch_dtype
        )

        clusters_with_idx = self._cluster_blocks(X)
        self.labels_ = self._assign_labels(X, clusters_with_idx)

        self.clusters_ = []
        for cluster_items in clusters_with_idx:
            cluster_blocks = [block for orig_idx, block in cluster_items]
            self.clusters_.append(cluster_blocks)

        return self

    def compute_error_bound(self, blocks: List[np.ndarray]) -> float:
        _require_torch()
        self.resolved_device_ = resolve_torch_device(self.device)
        self.resolved_torch_dtype_ = _resolve_runtime_torch_dtype(
            self.resolved_device_, self.torch_dtype
        )

        tensor_blocks = [self._to_tensor(block) for block in blocks]
        norms_blocks = torch.stack(
            [torch.linalg.norm(block, ord="fro") for block in tensor_blocks]
        )
        available_indices = torch.argsort(norms_blocks, descending=True).tolist()

        (
            _indeces_block0,
            norm_M,
            Q,
            S,
            ordered_indices,
            get_next_idx,
        ) = self._order_and_prepare_first_block(
            blocks, tensor_blocks, norms_blocks, available_indices
        )

        singular_vals = torch.zeros(
            0,
            device=self.resolved_device_,
            dtype=self.resolved_torch_dtype_,
        )
        for k in range(len(ordered_indices)):
            next_idx = get_next_idx(k)
            norm_M += norms_blocks[next_idx] ** 2
            singular_vals, Q, S = self._update_top_singular_values(
                tensor_blocks[next_idx], Q=Q, S=S
            )

        lower_bound = torch.sum(singular_vals ** 2).item()
        error = (
            0.0 if lower_bound >= norm_M else ((norm_M - lower_bound) / norm_M) ** 0.5
        )
        return error
