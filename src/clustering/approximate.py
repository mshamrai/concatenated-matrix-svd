import numpy as np
from numpy.linalg import qr, eigh, norm
import random
from typing import List, Optional, Tuple
from .base import BaseClusterAlgorithm
from tqdm import tqdm


class ApproximateClusterAlgorithm(BaseClusterAlgorithm):
    """
    Approximate clustering algorithm that uses incremental singular value
    approximation to group data blocks based on error thresholds.
    
    Attributes:
        eps (float): Error threshold for clustering.
        r_target (int): Number of leading singular values to track.
        patience (int): Number of iterations to wait before stopping when max_k=0.
        verbose (bool): Whether to print progress information.
        random_seed (int): Random seed for reproducibility.
        tol (float): Tolerance for numerical rank decisions.
        oversampling (int): Extra dimensions to keep above r_target for stability.
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
        tol: float = 1e-12,
        oversampling: int = 5,
        sorting_strategy: str = 'norm',
        **kwargs
    ):
        """
        Initialize the ApproximateClusterAlgorithm.
        
        Args:
            eps: Error threshold for clustering.
            r_target: Number of leading singular values to track.
            patience: Number of iterations to wait before stopping when max_k=0.
            verbose: Whether to print progress information.
            random_seed: Random seed for reproducibility.
            tol: Tolerance for numerical rank decisions.
            oversampling: Extra dimensions to keep above r_target for stability.
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
        self.tol = tol
        self.oversampling = oversampling
        self.sorting_strategy = sorting_strategy
        self.clusters_: Optional[List[List[np.ndarray]]] = None
    
    def _update_top_singular_values(
        self,
        A_new: np.ndarray,
        Q: Optional[np.ndarray] = None,
        S: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Incrementally approximate the top r_target singular values of
        M = [A_1, A_2, ..., A_new] without ever forming M explicitly.

        State is (Q, S) such that approximately M M^T ≈ Q S Q^T with
        rank(S) <= r_target + oversampling.

        Parameters
        ----------
        A_new : (m, k) ndarray
            New block of columns to append to M.
        Q : (m, r) ndarray or None
            Current orthonormal basis of the approximate top subspace.
        S : (r, r) ndarray or None
            Current small Gram matrix in that basis.

        Returns
        -------
        singular_vals : (min(r_target, r_eff),) ndarray
            Approximate top singular values of the full concatenated M
            after appending A_new, sorted in descending order.
        Q_new : ndarray
            Updated orthonormal basis.
        S_new : ndarray
            Updated small Gram matrix.
        """
        A_new = np.asarray(A_new)

        # ---------- First block: initialize ----------
        if Q is None or S is None:
            # Thin QR of A_new
            Q0, R0 = qr(A_new, mode='reduced')   # A_new = Q0 R0

            # Build initial Gram S0 = R0 R0^T
            S0 = R0 @ R0.T

            # Eigen-decomp of S0
            w, U = eigh(S0)  # w ascending
            w = np.maximum(w, 0.0)

            # Keep only positive eigenvalues
            pos = w > self.tol
            if not np.any(pos):
                # Everything is numerically zero
                return np.zeros(0), None, None

            w_pos = w[pos]
            U_pos = U[:, pos]

            # Sort by descending eigenvalue
            idx = np.argsort(w_pos)[::-1]
            w_sorted = w_pos[idx]
            U_sorted = U_pos[:, idx]

            # Truncate to r_target + oversampling
            r_keep = min(len(w_sorted), self.r_target + self.oversampling)
            lam = w_sorted[:r_keep]
            U_top = U_sorted[:, :r_keep]

            Q_new = Q0 @ U_top
            S_new = np.diag(lam)

            singular_vals = np.sqrt(lam[:min(self.r_target, len(lam))])
            return singular_vals, Q_new, S_new

        # ---------- Subsequent blocks: update ----------
        r_curr = S.shape[0]

        # Project A_new into current subspace and residual
        Y = Q.T @ A_new            # (r_curr, k)
        A_in = Q @ Y               # (m, k)
        R = A_new - A_in           # (m, k)

        # QR on residual
        Q_res, B = qr(R, mode='reduced')  # R = Q_res B

        # Prune numerically zero rows in B
        if B.size > 0:
            row_norms = np.linalg.norm(B, axis=1)
            mask = row_norms > self.tol
            B = B[mask, :]
            Q_res = Q_res[:, mask]
        else:
            mask = np.array([], dtype=bool)

        r_res = B.shape[0]

        if r_res == 0:
            # No new directions: only in-span update
            Q_ext = Q
            S_ext = S + Y @ Y.T
        else:
            # Extended basis
            Q_ext = np.hstack([Q, Q_res])  # (m, r_curr + r_res)

            # Build extended Gram matrix S_ext of size (r_curr + r_res)
            # Top-left block: old S plus in-span contribution
            S11 = S + Y @ Y.T              # (r_curr, r_curr)
            # Off-diagonal blocks: cross-terms
            S12 = Y @ B.T                  # (r_curr, r_res)
            # Bottom-right block: residual Gram
            S22 = B @ B.T                  # (r_res, r_res)

            S_ext = np.empty((r_curr + r_res, r_curr + r_res),
                             dtype=A_new.dtype)
            S_ext[:r_curr, :r_curr] = S11
            S_ext[:r_curr, r_curr:] = S12
            S_ext[r_curr:, :r_curr] = S12.T
            S_ext[r_curr:, r_curr:] = S22

        # Eigen-decomposition of S_ext
        w, U = eigh(S_ext)  # ascending
        w = np.maximum(w, 0.0)

        pos = w > self.tol
        if not np.any(pos):
            # Degenerate: everything vanished numerically
            return np.zeros(0), None, None

        w_pos = w[pos]
        U_pos = U[:, pos]

        idx = np.argsort(w_pos)[::-1]  # descending
        w_sorted = w_pos[idx]
        U_sorted = U_pos[:, idx]

        # Truncate to r_target + oversampling
        r_keep = min(len(w_sorted), self.r_target + self.oversampling)
        lam = w_sorted[:r_keep]
        U_top = U_sorted[:, :r_keep]

        # Compress basis and Gram
        Q_new = Q_ext @ U_top           # (m, r_keep)
        S_new = np.diag(lam)            # (r_keep, r_keep)

        # Top singular values (approximate)
        singular_vals = np.sqrt(lam[:min(self.r_target, len(lam))])

        return singular_vals, Q_new, S_new
    
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

            singular_vals, Q, S = self._update_top_singular_values(block0)
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

            singular_vals, Q, S = self._update_top_singular_values(block0)
            
            # Compute and sort by residual norms
            residual_info = []
            for i in available_indices:
                if i not in indeces_block0:
                    Y = Q.T @ blocks[i]
                    R = blocks[i] - Q @ Y
                    residual_info.append((norm(R, 'fro'), i))
            
            residual_info.sort()
            ordered_indices = [i for _, i in residual_info]
            # Get indices to process: in order
            get_next_idx = lambda k: ordered_indices[k]
        
        return (indeces_block0, norm_M, Q, S,
                ordered_indices, get_next_idx)

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

        (indeces_block0, norm_M, Q,
         S, ordered_indices, get_next_idx) = self._order_and_prepare_first_block(
            blocks, norms_blocks, available_indices)

        k = 0
        
        # Try adding blocks according to strategy
        while k < len(ordered_indices):
            next_idx = get_next_idx(k)
            test_norm_M = norm_M + norms_blocks[next_idx] ** 2
            singular_vals, Q, S = self._update_top_singular_values(
                blocks[next_idx], Q=Q, S=S
            )
            lower_bound = sum(sigma ** 2 for sigma in singular_vals)
            
            error = 0 if lower_bound >= test_norm_M else \
                    ((test_norm_M - lower_bound) / test_norm_M) ** 0.5
            
            if self.verbose:
                print(f"Error: {error}")
            
            if error > self.eps:
                break
            
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
            
            # Update progress bar
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
    
    def fit(self, X: List[np.ndarray]) -> 'ApproximateClusterAlgorithm':
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

        (_, norm_M, Q,
         S, ordered_indices, get_next_idx) = self._order_and_prepare_first_block(
            blocks, norms_blocks, available_indices)
        
        for k in range(len(ordered_indices)):
            next_idx = get_next_idx(k)
            norm_M += norms_blocks[next_idx] ** 2
            singular_vals, Q, S = self._update_top_singular_values(
                blocks[next_idx], Q=Q, S=S
            )

        lower_bound = sum(sigma ** 2 for sigma in singular_vals)
        
        error = 0 if lower_bound >= norm_M else \
                ((norm_M - lower_bound) / norm_M) ** 0.5
        
        return error