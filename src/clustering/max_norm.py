import random
from typing import List, Optional, Tuple
import numpy as np
from .base import BaseClusterAlgorithm


class MaxNormClusterAlgorithm(BaseClusterAlgorithm):
    """
    Max Norm Clustering algorithm.
    
    This algorithm clusters data blocks based on their Frobenius norms,
    using Weyl's inequality to determine cluster composition.
    
    Attributes:
        eps (float): Error threshold for clustering.
        patience (int): Number of single-block clusters before stopping.
        verbose (bool): Whether to print progress information.
        random_seed (int): Random seed for reproducibility.
        labels_ (np.ndarray): Cluster labels for each data point.
        clusters_ (List[List[np.ndarray]]): List of clusters, each containing blocks.
    """
    
    def __init__(
        self,
        eps: float = 0.1,
        r_target: int = 10,
        patience: int = 10,
        verbose: bool = False,
        random_seed: int = 23,
        **kwargs
    ):
        """
        Initialize the Weyl Max Norm Clustering algorithm.
        
        Args:
            eps: Error threshold for clustering.
            r_target: Target rank for approximation.
            patience: Number of single-block clusters before stopping.
            verbose: Whether to print progress information.
            random_seed: Random seed for reproducibility.
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.patience = patience
        self.verbose = verbose
        self.random_seed = random_seed
        self.r_target = r_target
        self.clusters_: Optional[List[List[np.ndarray]]] = None
    
    def _sort_blocks_by_norm(
        self,
        blocks: List[np.ndarray],
        norms_blocks: List[float]
    ) -> tuple:
        """Sort blocks by their norms in descending order."""
        order = np.argsort(norms_blocks)[::-1]
        blocks = [blocks[i] for i in order]
        norms_blocks = [norms_blocks[i] for i in order]
        return blocks, norms_blocks
    
    def _max_k_for_block(self, norms_blocks: List[float], blocks: List[np.ndarray]) -> Tuple[int, int]:
        """
        Calculate maximum number of blocks that can be clustered
        with the largest block while maintaining error threshold.
        """
        last_k = 0
        first_k = 1
        block0 = blocks[0]
        max_A = norms_blocks[0] ** 2

        while first_k < len(blocks) and blocks[first_k].shape[1] + block0.shape[1] <= self.r_target:
            block0 = np.hstack([block0, blocks[first_k]])
            max_A += norms_blocks[first_k] ** 2
            first_k += 1

        norm_M = max_A
        
        while len(norms_blocks) - first_k - last_k > 0:
            norm_M += norms_blocks[-(last_k+1)] ** 2
            error = ((norm_M - max_A) / norm_M) ** 0.5
            if error > self.eps:
                return first_k, last_k
            last_k += 1

        return first_k, last_k
    
    def _cluster_blocks(self, blocks: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Cluster the blocks using Weyl max norm approach.
        
        Args:
            blocks: List of data blocks to cluster.
            
        Returns:
            List of clusters, each containing blocks.
        """
        random.seed(self.random_seed)
        norms_blocks = [np.linalg.norm(A, 'fro') for A in blocks]
        blocks, norms_blocks = self._sort_blocks_by_norm(blocks, norms_blocks)
        
        clusters = []
        i_cluster_ones = 0
        
        while len(blocks) > 0:
            if self.verbose:
                print(f"Remaining blocks: {len(blocks)}")
            
            first_k, last_k = self._max_k_for_block(norms_blocks, blocks)
            
            if last_k == 0:
                i_cluster_ones += 1
                if i_cluster_ones >= self.patience:
                    for block in blocks:
                        clusters.append([block])
                    if self.verbose:
                        print(f"Patience {self.patience} reached. Stopping clustering.")
                    break
                else:
                    cluster = [blocks[0]]
            else:
                i_cluster_ones = 0
                cluster = blocks[:first_k] + blocks[-last_k:]
            
            clusters.append(cluster)
            # Remove clustered blocks
            blocks = blocks[first_k:-last_k] if last_k > 0 else blocks[first_k:]
            norms_blocks = norms_blocks[first_k:-last_k] if last_k > 0 else norms_blocks[first_k:]
        
        return clusters
    
    def fit(self, X: np.ndarray) -> 'MaxNormClusterAlgorithm':
        """
        Fit the clustering algorithm to the data.
        
        Args:
            X: Input data. If X is a list of arrays, treats them as blocks.
               Otherwise, treats X as a single block.
            
        Returns:
            self: Fitted estimator.
        """
        # Handle input as list of blocks or single array
        if isinstance(X, list):
            blocks = X
        else:
            # Treat as single block
            blocks = [X]
        
        # Perform clustering
        self.clusters_ = self._cluster_blocks(blocks)
        
        # Assign labels based on cluster membership
        self.labels_ = self._assign_labels(blocks)
        
        return self
    
    def _assign_labels(self, original_blocks: List[np.ndarray]) -> np.ndarray:
        """
        Assign cluster labels to each original block.
        
        Args:
            original_blocks: Original list of blocks.
            
        Returns:
            Array of cluster labels.
        """
        n_blocks = len(original_blocks)
        labels = np.zeros(n_blocks, dtype=int)
        
        # Create a mapping from block identity to original index
        block_to_idx = {id(block): idx for idx, block in enumerate(original_blocks)}
        
        # Assign labels based on cluster membership
        for cluster_idx, cluster in enumerate(self.clusters_):
            for block in cluster:
                if id(block) in block_to_idx:
                    labels[block_to_idx[id(block)]] = cluster_idx
        
        return labels
    
    def compute_error_bound(self, blocks: List[np.ndarray]) -> float:
        """
        Compute error bound using Weyl's inequality.
        
        Args:
            blocks: List of data blocks.
            
        Returns:
            Weyl error bound.
        """
        norms = [np.linalg.norm(A, 'fro') for A in blocks]
        blocks, norms = self._sort_blocks_by_norm(blocks, norms)
        block0 = blocks[0]
        max_norm_sq = norms[0] ** 2
        max_norm_k = 1
        while max_norm_k < len(blocks) and blocks[max_norm_k].shape[1] + block0.shape[1] <= self.r_target:
            block0 = np.hstack([block0, blocks[max_norm_k]])
            max_norm_sq += norms[max_norm_k] ** 2
            max_norm_k += 1

        total_norm_sq = sum(n**2 for n in norms)
        
        if total_norm_sq == 0:
            return 0.0
        
        error_bound = np.sqrt((total_norm_sq - max_norm_sq) / total_norm_sq)
        return error_bound