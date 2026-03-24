"""
Qwen 3 Approximate Clustering Layers.

This module provides utilities for:
1. Clustering Qwen 3 linear layers by weight similarity
2. Computing shared SVD basis for each cluster
3. Substituting layers with cluster-based approximate layers
4. Loading saved cluster configurations

Usage:
    from qwen3_approx import (
        get_linear_layers,
        cluster_layers,
        substitute_layers,
        load_clusters
    )

    # Get all linear layers from a Qwen 3 model
    layers = get_linear_layers(qwen3_model)

    # Cluster layers by weight similarity
    clusters, cluster_info = cluster_layers(layers, eps=0.05, r_target=32)

    # Substitute layers with cluster basis layers
    model = substitute_layers(qwen3_model, clusters, cluster_info)

    # Load pre-computed clusters
    clusters, cluster_info = load_clusters('qwen3-7b')
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

from .layers import ClusterBasis, ClusterBasisLinear, create_cluster_linear


@dataclass
class ClusterConfig:
    """Configuration for a cluster of linear layers."""

    cluster_id: int
    cluster_basis: ClusterBasis
    layer_indices: List[int]  # Original layer names or indices
    layer_shapes: List[Tuple[int, int]]
    rank: int
    error_estimate: float
    weight_norm: float


def get_linear_layers(
    model: nn.Module,
    exclude_names: Optional[List[str]] = None
) -> List[nn.Linear]:
    """
    Get all nn.Linear layers from a model.

    Args:
        model: PyTorch model to extract linear layers from
        exclude_names: Optional list of parameter names to exclude

    Returns:
        List of nn.Linear layers in order of occurrence in the model
    """
    exclude_names = exclude_names or []
    layers = []

    def collect_layers(module, prefix=''):
        if isinstance(module, nn.Linear):
            # Skip excluded layers
            name = prefix + module.__class__.__name__ if prefix else module.__class__.__name__
            if name not in exclude_names:
                layers.append(module)

        for name, child in module.named_children():
            collect_layers(child, prefix + name + '.')

    collect_layers(model)
    return layers


def get_all_linear_params(
    model: nn.Module,
    exclude_names: Optional[List[str]] = None
) -> List[torch.Tensor]:
    """
    Get all weight matrices from linear layers.

    Args:
        model: PyTorch model
        exclude_names: Optional list of parameter names to exclude

    Returns:
        List of weight tensors in order
    """
    exclude_names = exclude_names or []
    weights = []

    def collect_weights(module, prefix=''):
        if isinstance(module, nn.Linear):
            name = prefix + module.__class__.__name__ if prefix else module.__class__.__name__
            if name not in exclude_names:
                weights.append(module.weight.data.clone())

        for name, child in module.named_children():
            collect_weights(child, prefix + name + '.')

    collect_weights(model)
    return weights


def cluster_layers(
    layers: List[nn.Linear],
    eps: float = 0.05,
    r_target: Optional[int] = None,
    sorting_strategy: str = 'residual',
    n_clusters: Optional[int] = None
) -> Tuple[List[ClusterConfig], Dict]:
    """
    Cluster linear layers by weight similarity.

    Clustering strategy:
    1. Group layers by shape (in_features, out_features)
    2. For each shape group, compute pairwise similarity
    3. Assign layers to clusters using similarity thresholds
    4. Compute SVD basis for each cluster from representative weight

    Args:
        layers: List of nn.Linear layers to cluster
        eps: Error threshold for clustering (relative to max-norm)
        r_target: Target rank (None: use full rank)
        sorting_strategy: How to sort blocks before clustering ('residual' or 'norm')
        n_clusters: Number of clusters (None: auto-detect)

    Returns:
        Tuple of (cluster_configs, cluster_metadata)
    """
    if not layers:
        return [], {}

    # Group layers by shape
    shape_groups: Dict[Tuple[int, int], List[nn.Linear]] = {}
    layer_info = []

    for idx, layer in enumerate(layers):
        shape = (layer.in_features, layer.out_features)
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append((idx, layer))
        layer_info.append({
            'index': idx,
            'name': f"Linear_{layer.in_features}x{layer.out_features}_{idx}",
            'shape': shape
        })

    # Process each shape group
    clusters: List[ClusterConfig] = []
    metadata = {
        'total_layers': len(layers),
        'shape_groups': len(shape_groups),
        'groups': {}
    }

    # Sort shapes for consistent processing
    sorted_shapes = sorted(shape_groups.keys())

    for shape in sorted_shapes:
        group_layers = shape_groups[shape]
        if len(group_layers) == 1:
            # Single layer in this shape group - create singleton cluster
            idx, layer = group_layers[0]
            weights = [layer.weight.data.clone()]
            cluster_basis = ClusterBasis.from_cluster(weights, r_target)
            error = cluster_basis.get_error_estimate(layer.weight.data)
            weight_norm = torch.norm(layer.weight.data, 'fro').item()

            cluster_config = ClusterConfig(
                cluster_id=len(clusters),
                cluster_basis=cluster_basis,
                layer_indices=[f"Linear_{shape[0]}x{shape[1]}_{idx}"],
                layer_shapes=[shape],
                rank=cluster_basis.rank,
                error_estimate=error,
                weight_norm=weight_norm
            )
            clusters.append(cluster_config)
            metadata['groups'][f"shape_{shape[0]}x{shape[1]}_singleton"] = [cluster_config]
            continue

        # Multiple layers - compute similarity and cluster
        weights = [layer.weight.data.clone() for _, layer in group_layers]
        layer_indices = [f"Linear_{shape[0]}x{shape[1]}_{idx}" for idx, _ in group_layers]

        # Sort by norm (descending) for residual-based approximation
        sorted_indices = sorted(
            range(len(weights)),
            key=lambda i: torch.norm(weights[i], 'fro').item(),
            reverse=True
        )
        sorted_weights = [weights[i] for i in sorted_indices]
        sorted_indices_map = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # Use approximate clustering for this shape group
        from src.clustering import ApproximateClusterAlgorithm

        # Convert torch tensors to numpy for clustering algorithm
        blocks = [w.cpu().detach().numpy() for w in sorted_weights]
        algorithm = ApproximateClusterAlgorithm(
            eps=eps,
            r_target=r_target,
            sorting_strategy=sorting_strategy
        )
        algorithm.fit(blocks)

        # Create clusters
        cluster_id = len(clusters)
        cluster_shapes = [shape] * len(algorithm.clusters_)
        # Flatten clusters and convert numpy arrays back to torch tensors
        # algorithm.clusters_ is a list of clusters, each cluster is a list of blocks
        all_weights = [
            torch.from_numpy(block)
            for cluster in algorithm.clusters_
            for block in cluster
        ]
        cluster_basis = ClusterBasis.from_cluster(all_weights, r_target)

        # Get representative weight (first block in sorted order)
        rep_idx = sorted_indices_map[0]
        rep_weight = sorted_weights[0]

        error = cluster_basis.get_error_estimate(rep_weight)
        weight_norm = torch.norm(rep_weight, 'fro').item()

        cluster_config = ClusterConfig(
            cluster_id=cluster_id,
            cluster_basis=cluster_basis,
            layer_indices=layer_indices,
            layer_shapes=[shape] * len(layer_indices),
            rank=cluster_basis.rank,
            error_estimate=error,
            weight_norm=weight_norm
        )
        clusters.append(cluster_config)

        metadata['groups'][f"shape_{shape[0]}x{shape[1]}_cluster"] = [cluster_config]

    return clusters, metadata


def substitute_layers(
    model: nn.Module,
    clusters: List[ClusterConfig],
    metadata: Dict,
    use_approx: bool = True
) -> nn.Module:
    """
    Substitute linear layers with cluster basis layers.

    For each cluster:
    1. Create a ClusterBasisLinear for each layer
    2. Set the cluster basis and layer index
    3. Optionally cache the exact weight for fallback

    Args:
        model: Model to substitute layers in
        clusters: List of cluster configurations
        metadata: Cluster metadata from clustering function
        use_approx: Whether to use approximate forward pass

    Returns:
        Model with substituted layers
    """
    if not clusters:
        return model

    # Track which layers have been replaced
    replaced_layers: Dict[str, nn.Linear] = {}

    for cluster_config in clusters:
        cluster_basis = cluster_config.cluster_basis

        for layer_name in cluster_config.layer_indices:
            # Find the original layer by traversing the model
            print(f"    Looking for layer: {layer_name}")
            original_layer = find_layer_by_name(model, layer_name)

            if original_layer is None:
                # Layer not found - skip
                print(f"      Not found, skipping")
                continue
            print(f"      Found: {type(original_layer).__name__}")

            # Cache the original weight before replacement (for comparison)
            original_weight = original_layer.weight.data.clone()
            original_bias = original_layer.bias.data.clone() if original_layer.bias is not None else None

            # Create new cluster basis linear layer
            new_layer = ClusterBasisLinear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias is not None,
                cluster_basis=cluster_basis,
                layer_index=cluster_config.cluster_id
            )

            # Set the cluster basis
            new_layer.set_cluster_basis(
                cluster_basis,
                layer_index=cluster_config.cluster_id
            )

            # Cache the ORIGINAL weight (not the basis reconstruction)
            new_layer._exact_weight = original_weight
            if original_bias is not None:
                new_layer._exact_bias = original_bias

            # Copy original bias if present
            if original_layer.bias is not None:
                new_layer.bias = nn.Parameter(
                    original_layer.bias.data.clone()
                )

            # Replace layer in model
            # Find parent and register module
            parent, name = find_parent_module(model, original_layer)
            if parent is not None:
                setattr(parent, name, new_layer)

            # Track replaced layer
            replaced_layers[layer_name] = new_layer

    return model


def find_layer_by_name(
    model: nn.Module,
    name: str
) -> Optional[nn.Module]:
    """
    Find a module by its full path name.

    Args:
        model: Model to search
        name: Full path to module (e.g., "model.layers.0.q_proj")

    Returns:
        Found module or None
    """
    # Split name by common module separators
    parts = name.split('.')

    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


def find_parent_module(
    model: nn.Module,
    target: nn.Module
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """
    Find the parent module containing a target module.

    Args:
        model: Model to search
        target: Target module

    Returns:
        Tuple of (parent_module, attribute_name) or (None, None)
    """
    def search(module, prefix=''):
        if module is target:
            return model if prefix == '' else (module, prefix)
        for name, child in module.named_children():
            result = search(child, prefix + name + '.')
            if result[0] is not None:
                return result
        return None, None

    return search(model)


def load_clusters(
    model_name: str,
    cluster_path: Optional[str] = None
) -> Tuple[List[ClusterConfig], Dict]:
    """
    Load pre-computed cluster configurations.

    Args:
        model_name: Name of the model (e.g., 'qwen3-7b')
        cluster_path: Path to cluster configuration file

    Returns:
        Tuple of (clusters, metadata)
    """
    # TODO: Implement cluster loading from disk
    # For now, return empty clusters
    return [], {}


def create_replacement_layer(
    layer: nn.Linear,
    cluster_basis: ClusterBasis,
    layer_index: int
) -> ClusterBasisLinear:
    """
    Create a replacement layer for a given linear layer.

    Args:
        layer: Original linear layer
        cluster_basis: Cluster basis to use
        layer_index: Index within cluster

    Returns:
        ClusterBasisLinear instance
    """
    return ClusterBasisLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        cluster_basis=cluster_basis,
        layer_index=layer_index
    )


# Make functions accessible at package level
__all__ = [
    'ClusterBasis',
    'ClusterBasisLinear',
    'create_cluster_linear',
    'get_linear_layers',
    'get_all_linear_params',
    'cluster_layers',
    'substitute_layers',
    'find_layer_by_name',
    'find_parent_module',
    'load_clusters',
    'create_replacement_layer',
    'ClusterConfig',
]
