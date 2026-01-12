import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans as SKLearnKMeans
from .base import BaseClusterAlgorithm


class KMeansClusterAlgorithm(BaseClusterAlgorithm):
    """
    K-Means clustering algorithm wrapper using sklearn.
    
    This algorithm applies K-Means clustering to flatten data blocks or uses
    block-level features for clustering.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        random_seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress information.
        labels_ (np.ndarray): Cluster labels for each data point/block.
        clusters_ (List[List[np.ndarray]]): List of clusters, each containing blocks.
        kmeans_ (SKLearnKMeans): Fitted sklearn KMeans instance.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        random_seed: int = 23,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the K-Means clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to form.
            random_seed: Random seed for reproducibility.
            verbose: Whether to print progress information.
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.verbose = verbose
        self.clusters_: Optional[List[List[np.ndarray]]] = None
        self.kmeans_: Optional[SKLearnKMeans] = None
    
    def fit(self, X) -> 'KMeansClusterAlgorithm':
        """
        Fit the K-Means clustering algorithm to the data.
        
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
            
            # Apply sklearn KMeans
            self.kmeans_ = SKLearnKMeans(
                n_clusters=min(self.n_clusters, len(blocks)),
                random_state=self.random_seed,
                verbose=1 if self.verbose else 0
                )
            
            self.labels_ = self.kmeans_.fit_predict(features)
            
            # Group blocks into clusters
            self.clusters_ = self._create_clusters_from_labels(blocks, self.labels_)
            
        else:
            # Handle single array
            if self.verbose:
                print(f"Clustering array of shape {X.shape}")
            
            self.kmeans_ = SKLearnKMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                random_state=self.random_seed,
                verbose=1 if self.verbose else 0,
                init=self.init,
                n_init=self.n_init,
                tol=self.tol
            )
            
            self.labels_ = self.kmeans_.fit_predict(X)
            
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
        
        Args:
            blocks: List of data blocks.
            labels: Cluster label for each block.
            
        Returns:
            List of clusters, each containing blocks.
        """
        n_clusters = len(np.unique(labels))
        clusters = [[] for _ in range(n_clusters)]
        
        for idx, label in enumerate(labels):
            clusters[label].append(blocks[idx])
        
        return clusters
