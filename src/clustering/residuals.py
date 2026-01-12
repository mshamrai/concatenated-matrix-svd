import numpy as np
from numpy.linalg import svd, norm
import random
from typing import List, Optional, Tuple
from .base import BaseClusterAlgorithm
from tqdm import tqdm
from scipy.linalg import qr


class ResidualsClusterAlgorithm(BaseClusterAlgorithm):
    """
    Residuals-based clustering algorithm that groups data blocks based on
    their singular value residuals and error thresholds.
    
    Attributes:
        eps (float): Error threshold for clustering.
        r_target (int): Target rank for approximation.
        patience (int): Number of iterations to wait before stopping when max_k=0.
        verbose (bool): Whether to print progress information.
        random_seed (int): Random seed for reproducibility.
        sorting_strategy (str): Sorting strategy ('norm' or 'residual').
        labels_ (np.ndarray): Cluster labels for each data block.
        clusters_ (List[List[np.ndarray]]): List of clusters, each containing blocks.
    """
    
    def __init__(
        self,
        eps: float = 0.1,
        r_target: int = 10,
        patience: int = 10,
        verbose: bool = False,
        random_seed: int = 23,
        sorting_strategy: str = 'norm',
        reorth_every: int = 10,
        **kwargs
    ):
        """
        Initialize the ResidualsClusterAlgorithm.
        
        Args:
            eps: Error threshold for clustering.
            r_target: Target rank for approximation.
            patience: Number of iterations to wait before stopping when max_k=0.
            verbose: Whether to print progress information.
            random_seed: Random seed for reproducibility.
            sorting_strategy: Sorting strategy - 'norm' (by Frobenius norm) or 'residual' (by residual norm).
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        if sorting_strategy not in ['norm', 'residual']:
            raise ValueError("sorting_strategy must be either 'norm' or 'residual'")
        self.eps = eps
        self.r_target = r_target
        self.patience = patience
        self.verbose = verbose
        self.random_seed = random_seed
        self.sorting_strategy = sorting_strategy
        self.clusters_: Optional[List[List[np.ndarray]]] = None
    
    @staticmethod
    def _project_away(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute residual R ≈ (I - QQ^T)A with one reorthogonalization pass."""
        if Q.size == 0:
            return A
        R = A - Q @ (Q.T @ A)
        # cheap reorth pass
        R = R - Q @ (Q.T @ R)
        return R
    
    @staticmethod
    def _orth_basis_pruned(X: np.ndarray) -> np.ndarray:
        """
        Thin QR but keep only columns corresponding to numerically
        nonzero diagonals of the R factor (avoids spurious directions).
        """
        if X.size == 0:
            return X
        Qx, Rx = qr(X, mode="economic")
        if Rx.size == 0:
            return Qx[:, :0]
        diag_abs = np.abs(np.diag(Rx))
        eps = np.finfo(X.dtype).eps
        rtol = eps * max(X.shape)

        # scale-aware threshold
        scale = np.linalg.norm(X, ord='fro')
        tau = rtol * scale
        keep = diag_abs > tau
        if np.any(keep):
            return Qx[:, keep]
        return Qx[:, :0]
    
    @staticmethod
    def _dynamic_tol(x: np.ndarray, c: float = 10.0) -> float:
        # relative tolerance for "is this numerically new?"
        return c * np.finfo(x.dtype).eps * np.sqrt(max(x.shape[0], 1))
    
    def _maybe_reorth_Q(self, Q: np.ndarray) -> np.ndarray:
        if Q.size == 0:
            return Q

        m, n = Q.shape

        # --- Hard correctness trigger: Q is wide (cannot be orthonormal) ---
        hard = (n > m)

        if not hard:
            return Q

        # Thin QR (economic). If m < n, SciPy returns Q as (m, m), which is correct.
        Q, R = qr(Q, mode="economic")

        # Prune numerically dependent directions (rank reveal-ish)
        d = np.abs(np.diag(R))
        if d.size == 0:
            return Q[:, :0]

        tau = np.finfo(Q.dtype).eps * max(Q.shape) * d.max()
        keep = d > tau
        return Q[:, keep] if np.any(keep) else Q[:, :0]
    
    def _append_if_new(self, Q: np.ndarray, V: np.ndarray) -> np.ndarray:
        if V.size == 0:
            return Q
        if Q.size == 0:
            return V

        # remove the component already in span(Q)
        V_perp = V - Q @ (Q.T @ V)
        # optional second pass (cheap, helps when Q is large)
        V_perp = V_perp - Q @ (Q.T @ V_perp)

        # "is it new?" test
        nV = np.linalg.norm(V, 'fro')
        nVp = np.linalg.norm(V_perp, 'fro')
        if nV == 0 or nVp <= self._dynamic_tol(Q) * nV:
            return Q  # adds nothing

        # orthonormalize only the new part (small QR)
        Vb = self._orth_basis_pruned(V_perp)
        return np.hstack([Q, Vb]) if Vb.size else Q

    
    def _estimate_singular_lower_bounds(
        self,
        A_new: np.ndarray,
        Q: Optional[np.ndarray] = None,
        sorted_lower_bounds: Optional[List[float]] = None
    ) -> tuple:
        """Estimate singular value lower bounds incrementally."""
        if Q is None and sorted_lower_bounds is None:
            Q = self._orth_basis_pruned(A_new)
            _, S, _ = svd(A_new, full_matrices=False)
            sorted_lower_bounds = [float(S[t]) for t in range(len(S))]
            sorted_lower_bounds.sort(reverse=True)
            return sorted_lower_bounds, Q

        R = self._project_away(Q, A_new)
        if np.linalg.norm(R, 'fro') <= self._dynamic_tol(A_new) * np.linalg.norm(A_new, 'fro'):
            return sorted_lower_bounds, Q   # nothing new to add
        
        _, S, _ = svd(R, full_matrices=False)
        if len(sorted_lower_bounds) < self.r_target or S[0] > sorted_lower_bounds[-1]:
            sorted_lower_bounds = sorted_lower_bounds + [float(s) for s in S]
            sorted_lower_bounds.sort(reverse=True)

        V = self._orth_basis_pruned(R)
        Q = self._append_if_new(Q, V)
        Q = self._maybe_reorth_Q(Q)

        return sorted_lower_bounds, Q
    
    def _order_and_prepare_first_block(self, blocks: List[np.ndarray],
                                      norms_blocks: List[float],
                                      available_indices: List[int]) -> tuple:
        if self.sorting_strategy == 'norm':
            # available_indices is already sorted by norm descending
            ordered_indices = available_indices
            # Get indices to process: first, then from smallest (end)
            get_next_idx = lambda k: ordered_indices[-(k+1)]
            
            # Initialize with first block
            idx0 = ordered_indices[0]
            norm_M = norms_blocks[idx0] ** 2
            block0 = blocks[idx0]
            indeces_block0 = [idx0]
            i = 1
            while block0.shape[1] < self.r_target and i < len(ordered_indices):
                next_idx = ordered_indices[i]
                if blocks[next_idx].shape[1] + block0.shape[1] > self.r_target:
                    break
                block0 = np.hstack([block0, blocks[next_idx]])
                indeces_block0.append(next_idx)
                norm_M += norms_blocks[next_idx] ** 2
                i += 1
            # Remove used indices from ordered_indices
            ordered_indices = ordered_indices[i:]
            
            lower_bounds, Q = self._estimate_singular_lower_bounds(block0)
        else:  # residual
            # Choose block with max norm as initial
            idx0 = available_indices[0]
            norm_M = norms_blocks[idx0] ** 2
            block0 = blocks[idx0]
            indeces_block0 = [idx0]
            i = 1
            while block0.shape[1] < self.r_target and i < len(available_indices):
                next_idx = available_indices[i]
                if blocks[next_idx].shape[1] + block0.shape[1] > self.r_target:
                    break
                block0 = np.hstack([block0, blocks[next_idx]])
                indeces_block0.append(next_idx)
                norm_M += norms_blocks[next_idx] ** 2
                i += 1
            # Remove used indices from available_indices
            available_indices = available_indices[i:]
            lower_bounds, Q = self._estimate_singular_lower_bounds(block0)
            
            # Compute and sort by residual norms
            residual_info = []
            for i in available_indices:
                if i not in indeces_block0:
                    R = self._project_away(Q, blocks[i])
                    residual_info.append((norm(R, 'fro'), i))
            
            residual_info.sort()
            ordered_indices = [i for _, i in residual_info]
            # Get indices to process: in order
            get_next_idx = lambda k: ordered_indices[k]
        
        return (indeces_block0, norm_M, lower_bounds, Q, ordered_indices, get_next_idx)
    
    def _compute_max_k_and_order(
        self,
        blocks: List[np.ndarray],
        norms_blocks: List[float],
        available_indices: List[int]
    ) -> Tuple[int, List[int]]:
        """
        Compute both the block order and maximum cluster size in one pass.
        
        For 'norm' strategy: uses pre-sorted available_indices.
        For 'residual' strategy: sorts by residual norm and adds in order.
        
        Args:
            blocks: List of all blocks.
            norms_blocks: Frobenius norms of all blocks.
            available_indices: Indices of blocks still available (pre-sorted for 'norm' strategy).
            
        Returns:
            Tuple of (max_k, cluster_indices) where cluster_indices is ordered list.
        """
        (indeces_block0, norm_M, lower_bounds, Q,
         ordered_indices, get_next_idx) = self._order_and_prepare_first_block(blocks, norms_blocks, available_indices)
        
        # Try adding blocks according to strategy
        k = 0
        while k < len(ordered_indices):
            next_idx = get_next_idx(k)
            test_norm_M = norm_M + norms_blocks[next_idx] ** 2

            lower_bounds, Q = self._estimate_singular_lower_bounds(
                blocks[next_idx], Q, lower_bounds
            )
            lower_bounds = lower_bounds[:self.r_target]
            lower_bound = sum(sigma ** 2 for sigma in lower_bounds)

            error = 0 if lower_bound >= test_norm_M else ((test_norm_M - lower_bound) / test_norm_M) ** 0.5

            if error > self.eps:
                break
            
            if self.verbose:
                print(f"Error: {error:.6f}, k: {k+1}, Q shape: {Q.shape}")

            norm_M = test_norm_M
            k += 1
        
        # Build cluster indices based on strategy
        if self.sorting_strategy == 'norm':
            # First + last k blocks
            cluster_indices = indeces_block0 + ordered_indices[-k:] if k > 0 else indeces_block0
        else:  # residual
            # First + k blocks in order
            cluster_indices = indeces_block0 + ordered_indices[:k]
        
        return k, cluster_indices
    
    def _cluster_blocks(self, blocks: List[np.ndarray]) -> List[List[tuple]]:
        """
        Cluster the blocks using the selected sorting strategy.
        
        Args:
            blocks: List of data blocks to cluster.
            
        Returns:
            List of clusters, each containing (index, block) tuples.
        """
        random.seed(self.random_seed)
        
        # Compute norms once for all blocks
        norms_blocks = [np.linalg.norm(block, 'fro') for block in blocks]
        
        # sort once at the beginning
        available_indices = sorted(range(len(blocks)), key=lambda i: norms_blocks[i], reverse=True)
        
        clusters = []
        i_cluster_ones = 0

        pbar = tqdm(total=len(available_indices), desc="Clustering blocks")
        
        while available_indices:
            if self.verbose:
                print(f"Remaining blocks: {len(available_indices)}")
            
            # Compute order and max_k in single pass
            max_k, cluster_indices = self._compute_max_k_and_order(
                blocks, norms_blocks, available_indices
            )
            
            if max_k == 0:
                i_cluster_ones += 1
                if i_cluster_ones >= self.patience:
                    # Add all remaining blocks as individual clusters
                    for idx in available_indices:
                        clusters.append([(idx, blocks[idx])])
                    pbar.update(len(available_indices))
                    if self.verbose:
                        print(f"Patience {self.patience} reached. Stopping clustering.")
                    break
            else:
                i_cluster_ones = 0
            
            # Create cluster with (index, block) tuples
            cluster = [(idx, blocks[idx]) for idx in cluster_indices]
            clusters.append(cluster)

            pbar.update(len(cluster_indices))
            
            # Remove clustered blocks from available
            available_indices = [idx for idx in available_indices if idx not in cluster_indices]
        
        pbar.close()
        return clusters
    
    def _assign_labels(
        self,
        original_blocks: List[np.ndarray],
        clusters_with_idx: List[List[tuple]]
    ) -> np.ndarray:
        """
        Assign cluster labels to each original block.
        
        Args:
            original_blocks: Original list of blocks.
            clusters_with_idx: Clusters containing (index, block) tuples.
            
        Returns:
            Array of cluster labels.
        """
        n_blocks = len(original_blocks)
        labels = np.zeros(n_blocks, dtype=int)
        
        # Assign labels based on cluster membership
        for cluster_idx, cluster_items in enumerate(clusters_with_idx):
            for orig_idx, block in cluster_items:
                labels[orig_idx] = cluster_idx
        
        return labels
    
    def fit(self, X: List[np.ndarray]) -> 'ResidualsClusterAlgorithm':
        """
        Fit the clustering algorithm to the data blocks.
        
        Args:
            X: List of data blocks (matrices) to cluster.
            
        Returns:
            self: Fitted estimator.
        """
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be a non-empty list of numpy arrays")
        
        # Perform clustering
        clusters_with_idx = self._cluster_blocks(X)
        
        # Assign labels
        self.labels_ = self._assign_labels(X, clusters_with_idx)
        
        # Store clusters as list of blocks (without indices)
        self.clusters_ = []
        for cluster_items in clusters_with_idx:
            cluster_blocks = [block for orig_idx, block in cluster_items]
            self.clusters_.append(cluster_blocks)
        
        return self
    
    def compute_error_bound(self, blocks: List[np.ndarray]) -> float:
        norms_blocks = [np.linalg.norm(block, 'fro') for block in blocks]
        available_indices = sorted(range(len(blocks)), key=lambda i: norms_blocks[i], reverse=True)

        (_, norm_M, lower_bounds, Q,
         ordered_indices, get_next_idx) = self._order_and_prepare_first_block(blocks, norms_blocks, available_indices)
        
        for k in range(len(ordered_indices)):
            next_idx = get_next_idx(k)
            norm_M += norms_blocks[next_idx] ** 2

            lower_bounds, Q = self._estimate_singular_lower_bounds(
                blocks[next_idx], Q, lower_bounds
            )
            lower_bounds = lower_bounds[:self.r_target]

        lower_bound = sum(sigma ** 2 for sigma in lower_bounds)
        error = 0 if lower_bound >= norm_M else ((norm_M - lower_bound) / norm_M) ** 0.5
        return error