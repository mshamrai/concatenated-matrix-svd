from .approximate import ApproximateClusterAlgorithm
from .max_norm import MaxNormClusterAlgorithm
from .residuals import ResidualsClusterAlgorithm
from .kmeans import KMeansClusterAlgorithm
from .hdbscan import HDBSCANClusterAlgorithm
from .random import RandomClusterAlgorithm
from .approximate_torch import ApproximateTorchClusterAlgorithm


def get_clustering_algorithm(algorithm, **kwargs):
    """Factory function to get appropriate clustering algorithm."""
    algorithms = {
        'approximate': ApproximateClusterAlgorithm,
        'approximate_torch': ApproximateTorchClusterAlgorithm,
        'max_norm': MaxNormClusterAlgorithm,
        'residuals': ResidualsClusterAlgorithm,
        'kmeans': KMeansClusterAlgorithm,
        'hdbscan': HDBSCANClusterAlgorithm,
        'random': RandomClusterAlgorithm,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")

    if algorithms[algorithm] is None:
        raise ImportError(
            f"Algorithm '{algorithm}' requires optional dependencies that are not installed."
        )

    return algorithms[algorithm](**kwargs)
