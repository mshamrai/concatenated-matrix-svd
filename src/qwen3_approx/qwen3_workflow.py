#!/usr/bin/env python3
"""
Qwen 3 Cluster Workflow.

Complete pipeline for:
1. Downloading Qwen 3 model weights
2. Extracting linear layers
3. Clustering by weight similarity
4. Substituting with cluster basis layers
5. Saving cluster configurations

Usage:
    python src/qwen3_approx/qwen3_workflow.py --model qwen3-7b --eps 0.05 --r_target 32
"""

import argparse
import os
import time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .layers import ClusterBasis, ClusterBasisLinear
from . import (
    get_linear_layers,
    cluster_layers,
    substitute_layers,
    ClusterConfig,
)

from src.clustering import ApproximateClusterAlgorithm


class Qwen3ClusterProcessor:
    """
    Processor for clustering and compressing Qwen 3 model layers.

    Workflow:
    1. Load model and extract linear layers
    2. Cluster layers by weight similarity
    3. Compute SVD basis for each cluster
    4. Substitute layers with cluster basis layers
    5. (Optional) Fine-tune to reduce error

    Attributes:
        model: Qwen 3 model
        original_layers: List of original linear layers
        clusters: Cluster configurations
        metadata: Cluster metadata
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = './qwen3_approx',
        exclude_layers: Optional[List[str]] = None
    ):
        """
        Initialize the cluster processor.

        Args:
            model_name: Name of Qwen 3 model (e.g., 'Qwen/Qwen3-7B')
            cache_dir: Directory to cache model weights
            exclude_layers: Optional list of layer names to exclude
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.exclude_layers = exclude_layers or []

        self.model: Optional[nn.Module] = None
        self.original_layers: List[nn.Linear] = []
        self.clusters: List[ClusterConfig] = []
        self.metadata: Dict = {}

        # Clustering parameters
        self.eps: float = 0.05
        self.r_target: Optional[int] = None
        self.sorting_strategy: str = 'residual'
        self.n_clusters: Optional[int] = None

        # Results
        self.cluster_stats: Dict = {}

    def download_model(self, use_approx: bool = True) -> 'Qwen3ClusterProcessor':
        """
        Download Qwen 3 model weights.

        Args:
            use_approx: Whether to use approximate clustering

        Returns:
            Self for chaining
        """
        print(f"Downloading model: {self.model_name}")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Download model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        print(f"Model loaded: {self.model_name}")
        return self

    def extract_layers(self) -> 'Qwen3ClusterProcessor':
        """
        Extract all linear layers from the model.

        Returns:
            Self for chaining
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call download_model() first.")

        print("Extracting linear layers...")
        self.original_layers = get_linear_layers(
            self.model,
            exclude_names=self.exclude_layers
        )

        print(f"Extracted {len(self.original_layers)} linear layers")
        return self

    def cluster(
        self,
        eps: Optional[float] = None,
        r_target: Optional[int] = None,
        sorting_strategy: str = 'residual',
        n_clusters: Optional[int] = None
    ) -> 'Qwen3ClusterProcessor':
        """
        Cluster linear layers by weight similarity.

        Args:
            eps: Error threshold (default: 0.05)
            r_target: Target rank (default: None, use full rank)
            sorting_strategy: Sorting strategy ('residual' or 'norm')
            n_clusters: Number of clusters (default: None, auto-detect)

        Returns:
            Self for chaining
        """
        if self.original_layers:
            self.eps = eps or self.eps
            self.r_target = r_target or self.r_target
            self.sorting_strategy = sorting_strategy
            self.n_clusters = n_clusters or self.n_clusters

            print(f"Clustering {len(self.original_layers)} layers...")
            print(f"  eps={self.eps}, r_target={self.r_target}")

            self.clusters, self.metadata = cluster_layers(
                self.original_layers,
                eps=self.eps,
                r_target=self.r_target,
                sorting_strategy=self.sorting_strategy,
                n_clusters=self.n_clusters
            )

            print(f"Created {len(self.clusters)} clusters")
            self.cluster_stats = self._compute_stats()

            return self

        raise ValueError("No layers to cluster. Call extract_layers() first.")

    def _compute_stats(self) -> Dict:
        """Compute statistics about clustering results."""
        stats = {
            'total_layers': len(self.original_layers),
            'total_clusters': len(self.clusters),
            'avg_layers_per_cluster': 0,
            'max_layers_per_cluster': 0,
            'min_layers_per_cluster': float('inf'),
            'compression_ratio': 0,
            'avg_rank': 0,
            'max_rank': 0,
            'min_rank': float('inf'),
            'avg_error': 0,
            'total_params_original': 0,
            'total_params_compressed': 0,
        }

        if not self.clusters:
            return stats

        layer_counts = [len(c.layer_indices) for c in self.clusters]
        stats['avg_layers_per_cluster'] = sum(layer_counts) / len(layer_counts)
        stats['max_layers_per_cluster'] = max(layer_counts)
        stats['min_layers_per_cluster'] = min(layer_counts)

        # Compute rank statistics
        ranks = [c.rank for c in self.clusters]
        stats['avg_rank'] = sum(ranks) / len(ranks)
        stats['max_rank'] = max(ranks)
        stats['min_rank'] = min(ranks)

        # Compute error statistics
        errors = [c.error_estimate for c in self.clusters]
        stats['avg_error'] = sum(errors) / len(errors)

        # Compute compression ratio
        original_params = sum(torch.nn.Parameter(c.layer_shapes).numel() for c in self.clusters)
        compressed_params = sum(
            c.cluster_basis.US.numel() + c.cluster_basis.VT.numel()
            for c in self.clusters
        )
        stats['compression_ratio'] = original_params / compressed_params if compressed_params > 0 else 0

        return stats

    def substitute(
        self,
        use_approx: bool = True
    ) -> nn.Module:
        """
        Substitute layers with cluster basis layers.

        Args:
            use_approx: Whether to use approximate forward pass

        Returns:
            Model with substituted layers
        """
        if self.clusters:
            print("Substituting layers with cluster basis...")
            self.model = substitute_layers(
                self.model,
                self.clusters,
                self.metadata,
                use_approx=use_approx
            )
            print("Layer substitution complete")

        return self.model

    def get_stats(self) -> Dict:
        """
        Get statistics about clustering results.

        Returns:
            Dictionary with statistics
        """
        if not self.cluster_stats:
            self.cluster_stats = self._compute_stats()

        return self.cluster_stats

    def save_clusters(self, save_path: str) -> None:
        """
        Save cluster configurations to disk.

        Args:
            save_path: Path to save cluster configurations
        """
        if not self.clusters:
            raise ValueError("No clusters to save")

        # Save cluster configurations
        import json

        save_path = os.path.join(save_path, 'clusters.json')

        cluster_data = {
            'model_name': self.model_name,
            'eps': self.eps,
            'r_target': self.r_target,
            'sorting_strategy': self.sorting_strategy,
            'n_clusters': self.n_clusters,
            'stats': self.get_stats(),
            'clusters': []
        }

        for i, cluster in enumerate(self.clusters):
            cluster_info = {
                'cluster_id': cluster.cluster_id,
                'rank': cluster.rank,
                'error_estimate': cluster.error_estimate,
                'weight_norm': cluster.weight_norm,
                'layer_indices': cluster.layer_indices,
                'layer_shapes': cluster.layer_shapes,
                # Note: US and VT tensors are not saved directly
                # Instead, we save the representative weight
            }
            cluster_data['clusters'].append(cluster_info)

        with open(save_path, 'w') as f:
            json.dump(cluster_data, f, indent=2)

        print(f"Saved clusters to {save_path}")

        # Save cluster basis files
        for i, cluster in enumerate(self.clusters):
            save_dir = os.path.join(save_path, f'cluster_{i}')
            os.makedirs(save_dir, exist_ok=True)

            # Save US and VT
            US_path = os.path.join(save_dir, 'US.pt')
            VT_path = os.path.join(save_dir, 'VT.pt')

            torch.save(cluster.cluster_basis.US, US_path)
            torch.save(cluster.cluster_basis.VT, VT_path)

            print(f"Saved cluster {i} basis to {save_dir}")

    def load_clusters(self, save_path: str) -> None:
        """
        Load cluster configurations from disk.

        Args:
            save_path: Path to load cluster configurations from
        """
        import json

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Clusters not found at {save_path}")

        # Load cluster configurations
        with open(save_path, 'r') as f:
            cluster_data = json.load(f)

        for cluster_info in cluster_data['clusters']:
            # Load basis tensors
            save_dir = os.path.join(save_path, f'cluster_{cluster_info["cluster_id"]}')

            US_path = os.path.join(save_dir, 'US.pt')
            VT_path = os.path.join(save_dir, 'VT.pt')

            cluster_basis = ClusterBasis.from_single_weight(
                torch.zeros(1, 1),  # Dummy weight
                rank=cluster_info['rank']
            )
            # TODO: Need to save/load US and VT differently
            # For now, skip loading


def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description='Qwen 3 Cluster Workflow')

    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-7B',
        help='Name of Qwen 3 model to use'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./qwen3_approx',
        help='Directory to cache model weights'
    )

    parser.add_argument(
        '--eps',
        type=float,
        default=0.05,
        help='Error threshold for clustering (default: 0.05)'
    )

    parser.add_argument(
        '--r-target',
        type=int,
        default=32,
        help='Target rank for approximation (default: 32)'
    )

    parser.add_argument(
        '--sorting-strategy',
        type=str,
        default='residual',
        choices=['residual', 'norm'],
        help='Sorting strategy for clustering'
    )

    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (default: auto-detect)'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='./qwen3_approx_results',
        help='Path to save cluster configurations'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip model download (use existing model)'
    )

    parser.add_argument(
        '--skip-substitute',
        action='store_true',
        help='Skip layer substitution'
    )

    parser.add_argument(
        '--skip-save',
        action='store_true',
        help='Skip saving cluster configurations'
    )

    args = parser.parse_args()

    # Create processor
    processor = Qwen3ClusterProcessor(
        model_name=args.model,
        cache_dir=args.cache_dir,
    )

    # Download model (skip if requested)
    if not args.skip_download:
        processor.download_model(use_approx=True)

    # Extract layers
    processor.extract_layers()

    # Cluster layers
    processor.cluster(
        eps=args.eps,
        r_target=args.r_target,
        sorting_strategy=args.sorting_strategy,
        n_clusters=args.n_clusters
    )

    # Print statistics
    stats = processor.get_stats()
    print("\nClustering Statistics:")
    print(f"  Total layers: {stats['total_layers']}")
    print(f"  Total clusters: {stats['total_clusters']}")
    print(f"  Avg layers per cluster: {stats['avg_layers_per_cluster']:.2f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Avg rank: {stats['avg_rank']:.2f}")
    print(f"  Avg error: {stats['avg_error']:.4f}")

    # Substitute layers (skip if requested)
    if not args.skip_substitute:
        processor.substitute(use_approx=True)

    # Save clusters (skip if requested)
    if not args.skip_save:
        processor.save_clusters(args.save_path)

    print("\nWorkflow complete!")

    return processor


if __name__ == '__main__':
    main()
