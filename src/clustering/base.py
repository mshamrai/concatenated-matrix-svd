from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseClusterAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        labels_ (np.ndarray): Cluster labels for each data point.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the clustering algorithm.
        
        Args:
            **kwargs: Additional algorithm-specific parameters.
        """
        self.labels_: Optional[np.ndarray] = None
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterAlgorithm':
        """
        Fit the clustering algorithm to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            self: Fitted estimator.
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the algorithm and return cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            labels: Cluster labels for each sample.
        """
        self.fit(X)
        return self.labels_