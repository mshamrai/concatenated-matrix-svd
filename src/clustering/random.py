import random
from typing import List, Optional
import numpy as np
from .base import BaseClusterAlgorithm


class RandomClusterAlgorithm(BaseClusterAlgorithm):
    """
    Random Clustering algorithm.
    
    This algorithm randomly assigns data blocks to clusters.
    Useful as a baseline for comparison with other clustering methods.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        random_seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress information.
        labels_ (np.ndarray): Cluster labels for each data point.
        clusters_ (List[List[np.ndarray]]): List of clusters, each containing blocks.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        random_seed: int = 23,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the Random Clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to form.
            random_seed: Random seed for reproducibility.
            verbose: Whether to print progress information.
            **kwargs: Additional parameters (ignored, for compatibility).
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.verbose = verbose
        self.clusters_: Optional[List[List[np.ndarray]]] = None
    
    def _cluster_blocks(self, blocks: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Randomly assign blocks to clusters.
        
        Args:
            blocks: List of data blocks to cluster.
            
        Returns:
            List of clusters, each containing blocks.
        """
        random.seed(self.random_seed)
        
        n_blocks = len(blocks)
        # Adjust n_clusters if it's larger than number of blocks
        actual_n_clusters = min(self.n_clusters, n_blocks)
        
        if self.verbose:
            print(f"Randomly assigning {n_blocks} blocks to {actual_n_clusters} clusters")
        
        # Initialize empty clusters
        clusters = [[] for _ in range(actual_n_clusters)]
        
        # Randomly shuffle blocks
        block_indices = list(range(n_blocks))
        random.shuffle(block_indices)
        
        # Assign blocks to clusters in round-robin fashion after shuffling
        for idx, block_idx in enumerate(block_indices):
            cluster_idx = idx % actual_n_clusters
            clusters[cluster_idx].append(blocks[block_idx])
        
        if self.verbose:
            cluster_sizes = [len(c) for c in clusters]
            print(f"Cluster sizes: {cluster_sizes}")
        
        return clusters
    
    def fit(self, X: np.ndarray) -> 'RandomClusterAlgorithm':
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
    