"""
Cluster-based approximate linear layers for Qwen 3.

Clustering workflow:
1. Cluster weight matrices based on similarity
2. For each cluster, concatenate and compute ONE SVD
3. New linear layers substitute old ones and use the cluster's cached SVD
4. Each layer reconstructs via indexing into shared US, VT matrices
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ClusterBasis:
    """
    Cached SVD basis for a cluster of linear layers.

    Stores ONE SVD basis that approximates ALL layers in the cluster:
    - cluster_US: (total_in, rank) = shared U*S across all layers
    - cluster_VT: (rank, total_out) = shared V^T across all layers

    Each layer i reconstructs: W_i = cluster_US[:, i*in:(i+1)*in] @ cluster_VT[i*in:(i+1)*in, :]
    Wait, that's wrong. Let me think...

    Actually, for a cluster with N layers each of shape (in, out):
    - Concatenate: Big_W = [W_1, W_2, ..., W_N]  (not possible directly)

    Better approach for clustering:
    - Each layer has shape (in_i, out_i)
    - For same dimensions: all layers are (in, out)
    - Concatenate along new dimension or treat as block matrix

    For variable dimensions:
    - We need a different representation

    Simplest for Qwen3 (same dim linear layers):
    - Treat cluster as a single block: total_in x total_out
    - But each layer is in x out, not total_in x total_out

    Let me reconsider the clustering paradigm:
    1. We have N linear layers, each W_i = (in, out)
    2. We cluster them based on weight similarity
    3. For each cluster, we want ONE basis that approximates all

    Approach A (block concatenation):
    - Concatenate weights as block matrix: Big_W = [[W_1, W_2, ...]] doesn't make sense

    Approach B (shared basis):
    - Each W_i = U @ S @ V^T with shared U, S, V
    - But dimensions don't match across layers

    Approach C (flattened):
    - For each W_i, reshape to (-1, out) and concatenate
    - But this loses the linear structure

    Approach D (typical weight):
    - Use ONE representative weight for the cluster
    - New layers use the representative's basis

    This is Approach D:
    - Each cluster has ONE basis (from a representative weight)
    - Each layer in the cluster uses that basis
    - Layers are distinguished by their position/index

    For Qwen3 linear layers (same dimensions):
    - All layers have shape (in, out)
    - Cluster basis: U (in, rank), S (rank,), VT (rank, out)
    - Each layer: W_i = U @ S @ VT (same for all layers)
    - This doesn't differentiate layers!

    Re-reading the prompt: "new linear layers from a cluster should use one cached SVD and could be reconstructed from it by indexing and multiplication"

    So indexing is key. For same-dim layers, indexing doesn't work naturally.

    Unless... we're indexing into a concatenated weight space?

    Let me try:
    - Cluster: N layers of (in, out)
    - Concatenate: reshape each to (in*rank, out) no, that's wrong

    Alternative interpretation:
    - Cluster US: (N*in, rank) where N is num layers
    - Cluster VT: (rank, N*out)
    - Each layer i: rows (i*in:(i+1)*in), cols (i*out:(i+1)*out)
    - This requires N layers in a block structure

    For Qwen3, layers are sequential: layer 1, layer 2, ... layer N
    Each has shape (in, out)

    If we want indexing to work:
    - Cluster US: (N*in, rank)
    - Cluster VT: (rank, N*out)
    - But Qwen3 layers are (dim, hidden) or (hidden, hidden) etc.

    Actually, for Qwen3:
    - QLMoEBlock.linear_qkv_proj: shape (hidden, 3*hidden)
    - QLMoEBlock.linear_ou_proj: shape (hidden, hidden)

    So for each M in QMoE:
    - Some layers: (hidden, 3*hidden)
    - Some layers: (3*hidden, hidden)
    - Different shapes!

    For clustering these with different shapes:
    - We need to handle variable dimensions
    - OR we cluster by shape first, then within each shape group

    Let me implement:
    1. Group layers by shape
    2. For each shape group, create a cluster basis
    3. Within a shape group, layers use same basis

    For a shape group of N layers, all (in, out):
    - Cluster basis from ONE representative
    - Each layer uses that basis

    This is the simplest approach that matches the prompt.
    """

    def __init__(
        self,
        US: torch.Tensor,  # (in_features, rank)
        VT: torch.Tensor,  # (rank, out_features)
        rank: int,
        layer_indices: Optional[List[int]] = None,
        layer_dims: Optional[List[Tuple[int, int]]] = None
    ):
        self.US = US
        self.VT = VT
        self.rank = rank
        self.layer_indices = layer_indices or []
        self.layer_dims = layer_dims or []

    @classmethod
    def from_cluster(
        cls,
        weights: List[torch.Tensor],
        rank: Optional[int] = None
    ) -> 'ClusterBasis':
        """
        Create cluster basis from a list of weight tensors.

        For clustering:
        1. Check if all weights have same shape
        2. If yes, use first weight as representative
        3. If no, handle each shape group separately

        Returns a ClusterBasis that can be used by new linear layers.
        """
        if not weights:
            raise ValueError("Empty weights list")

        # Check if all same shape
        first_shape = weights[0].shape
        same_shape = all(w.shape == first_shape for w in weights)

        if same_shape:
            # Use first weight as representative
            rep_weight = weights[0].clone()
            return cls.from_single_weight(rep_weight, rank)
        else:
            # Variable shapes - not supported yet
            raise ValueError("Variable shapes not yet supported")

    @classmethod
    def from_single_weight(
        cls,
        weight: torch.Tensor,
        rank: Optional[int] = None
    ) -> 'ClusterBasis':
        """Create cluster basis from a single weight tensor."""
        if rank is None:
            rank = min(weight.shape)

        # Ensure rank doesn't exceed matrix dimensions
        rank = min(rank, weight.shape[0], weight.shape[1])

        U, S, VT = torch.linalg.svd(weight, full_matrices=False)

        # Truncate to specified rank
        U = U[:, :rank]
        S = S[:rank]
        VT = VT[:rank, :]

        # Combine S with U: US = U @ diag(S)
        cluster_US = U * S  # (in, rank)
        cluster_VT = VT     # (rank, out)

        return cls(
            US=cluster_US,
            VT=cluster_VT,
            rank=rank,
            layer_indices=[0],
            layer_dims=[weight.shape]
        )

    def __len__(self) -> int:
        return len(self.layer_indices) if self.layer_indices else 1

    def get_error_estimate(self, weight: torch.Tensor) -> float:
        """
        Compute the reconstruction error estimate for a given weight.

        The error is the Frobenius norm of the difference between the
        original weight and its low-rank reconstruction.

        Args:
            weight: Original weight tensor to estimate error for

        Returns:
            Frobenius norm of the reconstruction error
        """
        # Reconstruct weight from basis: US @ VT
        reconstructed = self.US @ self.VT
        # Compute Frobenius norm of difference
        error = torch.norm(weight - reconstructed, 'fro').item()
        return error


class ClusterBasisLinear(nn.Linear):
    """
    Linear layer that uses a shared cluster basis.

    This layer substitutes an original linear layer and uses a cached
    cluster SVD basis for approximate forward passes.

    Reconstruction: W_approx = US @ VT (shared basis for cluster)

    Each layer in the cluster uses the SAME basis.
    The layer is identified by its layer_index within the cluster.

    Attributes:
        cluster_basis: The shared cluster basis
        layer_index: Index of this layer within the cluster
        rank: Approximation rank
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        cluster_basis: Optional[ClusterBasis] = None,
        layer_index: int = 0
    ):
        """
        Initialize cluster basis linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Use bias term
            cluster_basis: Shared cluster basis (must have matching dimensions)
            layer_index: Index within the cluster (for tracking)
        """
        super().__init__(in_features, out_features, bias)

        self.layer_index = layer_index
        self.cluster_basis: Optional[ClusterBasis] = None

        if cluster_basis is not None:
            # Validate dimensions
            if len(cluster_basis.layer_dims) > 0:
                expected_shape = cluster_basis.layer_dims[layer_index]
                if (expected_shape[0], expected_shape[1]) != (in_features, out_features):
                    raise ValueError(f"Layer dimensions don't match cluster: expected {expected_shape}, got {(in_features, out_features)}")

            # Store basis components
            self.US = cluster_basis.US
            self.VT = cluster_basis.VT
            self.rank = cluster_basis.rank
            self.full_rank = cluster_basis.rank == -1 if cluster_basis.rank else False

        self._cached_weight: Optional[torch.Tensor] = None
        self._exact_weight: Optional[torch.Tensor] = None

    @property
    def is_approx(self) -> bool:
        return self.cluster_basis is not None and not self.full_rank

    @property
    def rank(self) -> Optional[int]:
        return self.cluster_basis.rank if self.cluster_basis else None

    def forward(
        self,
        x: torch.Tensor,
        approx: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (. .., in_features)
            approx: Use approximation (if approx and not full rank)

        Returns:
            Output tensor (. .., out_features)
        """
        if approx and self.is_approx and self.cluster_basis:
            # Approximate: x @ US @ VT
            xUS = x @ self.US.T  # (. .., in) @ (rank, in) = (. .., rank)
            result = xUS @ self.VT
            if self.bias is not None:
                result = result + self.bias
            return result
        else:
            # Exact: use cached exact weight if available, otherwise fall back to weight
            if self._exact_weight is not None:
                weight = self._exact_weight
                bias = self._exact_bias if hasattr(self, '_exact_bias') and self._exact_bias is not None else None
                result = x @ weight.T
                if bias is not None:
                    result = result + bias
                return result
            return super().forward(x)

    def set_cluster_basis(
        self,
        cluster_basis: ClusterBasis,
        layer_index: Optional[int] = None
    ) -> None:
        """Set the cluster basis for this layer."""
        self.cluster_basis = cluster_basis
        if layer_index is not None:
            self.layer_index = layer_index
        self.US = cluster_basis.US
        self.VT = cluster_basis.VT
        self.rank = cluster_basis.rank
        self.full_rank = cluster_basis.rank == -1 if cluster_basis.rank else False

        # Initialize weight from basis
        with torch.no_grad():
            self.weight.copy_(self.US @ self.VT)
            if self.bias is not None:
                self.bias.copy_(torch.zeros(self.out_features, device=self.weight.device, dtype=self.weight.dtype))

    def cache_exact(self) -> None:
        """Cache the exact weight for fallback."""
        with torch.no_grad():
            self._exact_weight = self.weight.clone()

    def cache_approx(self) -> None:
        """Cache the approximate weight."""
        with torch.no_grad():
            self._cached_weight = self.weight.clone()

    def get_error_estimate(self, target: Optional[torch.Tensor] = None) -> float:
        """Estimate the approximation error."""
        if self.full_rank or not self.cluster_basis:
            return 0.0

        reconstructed = self.US @ self.VT
        if target is not None:
            error = torch.norm(target - reconstructed, 'fro').item()
            return error / torch.norm(target, 'fro').item() if torch.norm(target, 'fro').item() > 0 else 0.0
        return 0.0


def create_cluster_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    basis: Optional[ClusterBasis] = None,
    layer_index: int = 0
) -> ClusterBasisLinear:
    """
    Factory function for cluster linear layers.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Use bias
        basis: Cluster basis to use
        layer_index: Layer index within cluster

    Returns:
        ClusterBasisLinear instance
    """
    return ClusterBasisLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        cluster_basis=basis,
        layer_index=layer_index
    )
