#!/usr/bin/env python3
"""
Qwen 3 Cluster Example.

Demonstrates how to:
1. Create a simple model with linear layers
2. Cluster the layers
3. Substitute with cluster basis layers
4. Verify the approximation

Usage:
    python src/qwen3_approx/example.py

Or with custom parameters:
    python src/qwen3_approx/example.py --eps 0.03 --r-target 16
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import ClusterBasis, ClusterBasisLinear
from . import cluster_layers, substitute_layers, get_linear_layers

from src.clustering import ApproximateClusterAlgorithm


class SimpleMLP(nn.Module):
    """Simple MLP for testing layer clustering."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        layer_sizes = [embed_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(num_layers - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


def create_test_model(hidden_dim: int = 512, num_layers: int = 3):
    """Create a simple test model."""
    embed_dim = hidden_dim  # Start with same dimension
    return SimpleMLP(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )


def main():
    """Run the clustering example."""
    parser = argparse.ArgumentParser(description='Qwen 3 Cluster Example')

    parser.add_argument(
        '--embed-dim',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=512,
        help='Hidden dimension (default: 512)'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of layers (default: 3)'
    )

    parser.add_argument(
        '--eps',
        type=float,
        default=0.05,
        help='Error threshold (default: 0.05)'
    )

    parser.add_argument(
        '--r-target',
        type=int,
        default=32,
        help='Target rank (default: 32)'
    )

    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (default: auto)'
    )

    parser.add_argument(
        '--sorting-strategy',
        type=str,
        default='residual',
        choices=['residual', 'norm'],
        help='Sorting strategy (default: residual)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen 3 Layer Clustering Example")
    print("=" * 60)

    # Create test model
    print(f"\nCreating test model: {args.embed_dim} -> {args.hidden_dim} MLP ({args.num_layers} layers)")
    model = create_test_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model.eval()

    # Print model info
    print(f"\nModel summary:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")

    # Get linear layers
    print(f"\nExtracting linear layers...")
    layers = get_linear_layers(model)
    print(f"Found {len(layers)} linear layers")

    # Cluster layers
    print(f"\nClustering layers...")
    print(f"  eps={args.eps}, r_target={args.r_target}")
    print(f"  sorting_strategy={args.sorting_strategy}")

    clusters, metadata = cluster_layers(
        layers,
        eps=args.eps,
        r_target=args.r_target,
        sorting_strategy=args.sorting_strategy,
        n_clusters=args.n_clusters
    )

    print(f"\nCreated {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}:")
        print(f"    Layers: {len(cluster.layer_indices)}")
        print(f"    Shape: {cluster.layer_shapes[0]}")
        print(f"    Rank: {cluster.rank}")
        print(f"    Error estimate: {cluster.error_estimate:.6f}")

    # Substitute layers
    print(f"\nSubstituting layers with cluster basis...")
    substituted_model = substitute_layers(
        model,
        clusters,
        metadata,
        use_approx=True
    )

    # Debug: print basis info
    print(f"\nBasis info:")
    for cluster in clusters:
        basis = cluster.cluster_basis
        print(f"  Cluster {cluster.cluster_id}:")
        print(f"    rank={basis.rank}")
        print(f"    US.shape={basis.US.shape}")
        print(f"    VT.shape={basis.VT.shape}")

    # Check all layers
    print(f"\nLayer info:")
    found_cluster_layers = 0
    for name, module in substituted_model.named_modules():
        if isinstance(module, ClusterBasisLinear):
            found_cluster_layers += 1
            print(f"  Layer {name}:")
            print(f"    cluster_basis={module.cluster_basis is not None}")
            print(f"    is_approx={module.is_approx}")
            print(f"    full_rank={module.full_rank if hasattr(module, 'full_rank') else 'N/A'}")
            print(f"    has _exact_weight={module._exact_weight is not None}")
            if module._exact_weight is not None:
                approx_weight = module.US @ module.VT
                error = torch.norm(module._exact_weight - approx_weight, 'fro').item()
                print(f"    reconstruction error (frobenius)={error:.6f}")

    print(f"  Found {found_cluster_layers} ClusterBasisLinear layers")

    # Test forward pass
    print(f"\nTesting forward pass...")
    test_input = torch.randn(1, 512)  # Batch size 1, embed_dim

    with torch.no_grad():
        original_output = model(test_input)
        substituted_output = substituted_model(test_input)

        error = torch.abs(original_output - substituted_output).mean().item()
        print(f"  Original output shape: {original_output.shape}")
        print(f"  Substituted output shape: {substituted_output.shape}")
        print(f"  Mean absolute error: {error:.6f}")

        # Test with larger batch
        test_input_large = torch.randn(8, 512)
        with torch.no_grad():
            orig_large = model(test_input_large)
            subst_large = substituted_model(test_input_large)
            error_large = torch.abs(orig_large - subst_large).mean().item()
            print(f"  Large batch MAE: {error_large:.6f}")

    # Create summary
    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    orig_params = sum(p.numel() for p in model.parameters())
    subst_params = sum(p.numel() for p in substituted_model.parameters())
    compression_ratio = orig_params / subst_params if subst_params > 0 else 0

    print(f"Original model params: {orig_params:,}")
    print(f"Substituted model params: {subst_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    print(f"\nClustering configuration:")
    print(f"  eps: {args.eps}")
    print(f"  r_target: {args.r_target}")
    print(f"  sorting_strategy: {args.sorting_strategy}")

    print(f"\nNote: This example uses a simple MLP for testing.")
    print(f"Real Qwen 3 model would have many more layers and dimensions.")
    print("=" * 60)


if __name__ == '__main__':
    main()
