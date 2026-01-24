import numpy as np
from typing import List, Optional
from hdbscan import HDBSCAN as HDBSCAN_
from .base import BaseClusterAlgorithm


class HDBSCANClusterAlgorithm(BaseClusterAlgorithm):
    """
    HDBSCAN clustering algorithm wrapper.
    
    This algorithm applies HDBSCAN (Hierarchical Density-Based Spatial Clustering
    of Applications with Noise) to data blocks using block-level features.
    
    Attributes:
        min_cluster_size (int): Minimum size of clusters.
        min_samples (int): Number of samples in a neighborhood for a point to be a core point.
        metric (str): Distance metric to use.
        random_seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress information.
        labels_ (np.ndarray): Cluster labels for each data point/block.
        clusters_ (List[List[np.ndarray]]): List of clusters, each containing blocks.
        hdbscan_ (HDBSCAN_): Fitted HDBSCAN instance.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = None,
        metric: str = 'euclidean',
        random_seed: int = 23,
        verbose: bool = False,
        cluster_selection_epsilon: float = 0.0,
        **kwargs
    ):
        """
        Initialize the HDBSCAN clustering algorithm.
        
        Args:
            min_cluster_size: Minimum size of clusters.
            min_samples: Number of samples in a neighborhood for a point to be a core point.
                        If None, defaults to min_cluster_size.
            metric: Distance metric to use.
            random_seed: Random seed for reproducibility.
            verbose: Whether to print progress information.
            cluster_selection_epsilon: Distance threshold for cluster selection.
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.random_seed = random_seed
        self.verbose = verbose
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.clusters_: Optional[List[List[np.ndarray]]] = None
        self.hdbscan_: Optional[HDBSCAN_] = None
    
    def fit(self, X) -> 'HDBSCANClusterAlgorithm':
        """
        Fit the HDBSCAN clustering algorithm to the data.
        
        Args:
            X: Input data. Can be:
               - A single numpy array of shape (n_samples, n_features)
               - A list of numpy arrays (blocks), which will be flattened or
                 clustered based on block-level features.
            
        Returns:
            self: Fitted estimator.
        """
        # Handle list of blocks
        if isinstance(X, list):
            blocks = X
            features = np.array([block.flatten() for block in blocks])
            
            if self.verbose:
                print(f"Clustering {len(blocks)} blocks with feature dim {features.shape[1]}")
            
            # Apply HDBSCAN
            self.hdbscan_ = HDBSCAN_(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                core_dist_n_jobs=-1  # Use all available cores
            )
            
            self.labels_ = self.hdbscan_.fit_predict(features)
            
            if self.verbose:
                n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
                n_noise = list(self.labels_).count(-1)
                print(f"Found {n_clusters} clusters and {n_noise} noise points")
            
            # Group blocks into clusters
            self.clusters_ = self._create_clusters_from_labels(blocks, self.labels_)
            
        else:
            # Handle single array
            if self.verbose:
                print(f"Clustering array of shape {X.shape}")
            
            self.hdbscan_ = HDBSCAN_(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                core_dist_n_jobs=-1
            )
            
            self.labels_ = self.hdbscan_.fit_predict(X)
            
            if self.verbose:
                n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
                n_noise = list(self.labels_).count(-1)
                print(f"Found {n_clusters} clusters and {n_noise} noise points")
            
            # For consistency, create clusters_ even for single array case
            self.clusters_ = None
        
        return self
    
    def _create_clusters_from_labels(
        self,
        blocks: List[np.ndarray],
        labels: np.ndarray
    ) -> List[List[np.ndarray]]:
        """
        Group blocks into clusters based on labels.
        
        Note: HDBSCAN assigns -1 to noise points, which are grouped separately.
        
        Args:
            blocks: List of data blocks.
            labels: Cluster label for each block.
            
        Returns:
            List of clusters, each containing blocks.
        """
        unique_labels = set(labels)
        clusters = []
        
        for label in sorted(unique_labels):
            cluster = [blocks[idx] for idx, lbl in enumerate(labels) if lbl == label]
            if label == -1:
                # Add each noise point as a separate singleton cluster
                for block in cluster:
                    clusters.append([block])
            else:
                clusters.append(cluster)
        
        return clusters
