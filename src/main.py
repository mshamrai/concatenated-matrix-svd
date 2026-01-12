import argparse
import time
import numpy as np

from src.data import get_data_reader
from src.clustering import get_clustering_algorithm
from src.metrics import compute_errors, compute_effective_compression_ratio, filter_clusters


def prepare_blocks(data, dataset):
    """Convert raw data to list of blocks depending on dataset type."""
    if dataset == 'qualcomm':
        # Already in block format (K, m, n)
        return [data[i] for i in range(len(data))]
    elif dataset == 'bigearth':
        # Shape (N, H, W, C) or other layout -> flatten spatial dimensions
        # Convert to list of (H*W, C) blocks or similar
        N, H, W, C = data.shape
        blocks = [data[i].reshape(-1, 20) for i in range(N)]
        return blocks
    elif dataset == 'pdebench':
        # Shape depends on data structure, flatten appropriately
        # Assuming shape (N, ...) where we flatten all but first dimension
        blocks = [data[i].reshape(-1, 67) for i in range(len(data))]
        return blocks
    elif dataset == 'smolvlm':
        # Already in block format from SmolVLMDataReader
        return [data[i] for i in range(len(data))]
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")


def main():
    parser = argparse.ArgumentParser(description='Greedy Spectral Clustering Experiments')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['qualcomm', 'bigearth', 'pdebench', 'smolvlm'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=False,
                        help='Path to the data directory')
    
    # Algorithm arguments
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['approximate', 'max_norm', 'residuals', 'kmeans', 'hdbscan', 'random'],
                        help='Clustering algorithm to use')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Error threshold for clustering')
    parser.add_argument('--r_target', type=int, default=10,
                        help='Target rank for approximation (for approximate/residuals)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of iterations to wait before stopping when max_k=0')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress information')
    parser.add_argument('--random_seed', type=int, default=23,
                        help='Random seed for reproducibility')
    
    # Algorithm-specific arguments
    parser.add_argument('--sorting_strategy', type=str, default='norm',
                        choices=['norm', 'residual'],
                        help='Sorting strategy for approximate/residuals algorithms')
    parser.add_argument('--oversampling', type=int, default=5,
                        help='Oversampling parameter for approximate algorithm')
    parser.add_argument('--tol', type=float, default=1e-12,
                        help='Tolerance for approximate algorithm')
    
    # KMeans and Random-specific arguments
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for kmeans and random algorithms')
    
    # HDBSCAN-specific arguments
    parser.add_argument('--min_cluster_size', type=int, default=2,
                        help='Minimum cluster size for hdbscan algorithm')
    parser.add_argument('--min_samples', type=int, default=None,
                        help='Minimum samples for hdbscan algorithm')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Distance metric for hdbscan algorithm')
    parser.add_argument('--cluster_selection_epsilon', type=float, default=0.0,
                        help='Cluster selection epsilon for hdbscan algorithm')
    
    # Dataset-specific arguments
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (for bigearth)')
    parser.add_argument('--layout', type=str, default='NHWC',
                        choices=['NHWC', 'NCHW', 'HWCN', 'CHWN'],
                        help='Layout for bigearth dataset')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum files for PDEBench dataset')
    
    # SmolVLM-specific arguments
    parser.add_argument('--model_path', type=str, default='HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
                        help='Model path for SmolVLM dataset')
    parser.add_argument('--target_dim', type=int, default=768,
                        help='Target dimension for SmolVLM weight reshaping')
    
    args = parser.parse_args()
    
    # Prepare dataset-specific kwargs
    data_kwargs = {'data_path': args.data_path}
    if args.dataset == 'qualcomm':
        pass  # Uses default parameters
    elif args.dataset == 'bigearth':
        data_kwargs = {
            'root': args.data_path,
            'max_samples': args.max_samples,
            'layout': args.layout,
            'random_seed': args.random_seed,
        }
    elif args.dataset == 'pdebench':
        data_kwargs = {
            'path': args.data_path,
            'max_samples': args.max_samples,
            'random_seed': args.random_seed,
        }
    elif args.dataset == 'smolvlm':
        data_kwargs = {
            'model_path': args.model_path,
            'target_dim': args.target_dim,
        }
    
    # Load data
    print(f"Loading {args.dataset} dataset from {args.data_path}")
    data_reader = get_data_reader(args.dataset, **data_kwargs)
    data = data_reader.read()
    print(f"Loaded data with shape: {data.shape}")
    
    # Prepare blocks
    blocks = prepare_blocks(data, args.dataset)
    print(f"Prepared {len(blocks)} data blocks")
    if blocks:
        print(f"Block shape: {blocks[0].shape}")
    
    # Prepare algorithm-specific kwargs
    algo_kwargs = {
        'eps': args.eps,
        'patience': args.patience,
        'verbose': args.verbose,
        'random_seed': args.random_seed,
    }

    if args.algorithm in ['approximate', 'residuals', 'max_norm']:
        algo_kwargs['r_target'] = args.r_target
    
    if args.algorithm in ['approximate', 'residuals']:
        algo_kwargs['sorting_strategy'] = args.sorting_strategy
    
    if args.algorithm == 'approximate':
        algo_kwargs['oversampling'] = args.oversampling
        algo_kwargs['tol'] = args.tol
    
    if args.algorithm == 'kmeans':
        algo_kwargs['n_clusters'] = args.n_clusters
    
    if args.algorithm == 'random':
        algo_kwargs['n_clusters'] = args.n_clusters
    
    if args.algorithm == 'hdbscan':
        algo_kwargs['min_cluster_size'] = args.min_cluster_size
        algo_kwargs['min_samples'] = args.min_samples
        algo_kwargs['metric'] = args.metric
        algo_kwargs['cluster_selection_epsilon'] = args.cluster_selection_epsilon
    
    # Create and run clustering algorithm
    print(f"\nRunning {args.algorithm} algorithm with parameters:")
    for k, v in algo_kwargs.items():
        print(f"  {k}: {v}")
    
    clustering = get_clustering_algorithm(args.algorithm, **algo_kwargs)
    
    start_time = time.time()
    clustering.fit(blocks)
    clustering_time = time.time() - start_time
    
    clusters = clustering.clusters_
    print(f"\nClustering completed in {clustering_time:.2f} seconds")
    print(f"Formed {len(clusters)} clusters")
    
    # Filter singletons
    clusters_filtered = filter_clusters(clusters, args.r_target)
    print(f"{len(clusters_filtered)} clusters after removing singletons")
    
    # Compute metrics
    print("\nComputing metrics...")
    start_time = time.time()
    
    # Reconstruction errors
    reconstruction_errors = compute_errors(clusters_filtered, r=args.r_target)
    
    # Compression ratio
    compression_ratio = compute_effective_compression_ratio(
        clusters, args.r_target, blocks
    )

    metrics_time = time.time() - start_time
    print(f"Metric computation took {metrics_time:.2f} seconds")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of blocks: {len(blocks)}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Non-singleton clusters: {len(clusters_filtered)}")
    print(f"\nReconstruction Error: {np.mean(reconstruction_errors):.6f} ± {np.std(reconstruction_errors):.6f}")
    print(f"Compression Ratio: {compression_ratio:.4f}")
    print(f"\nTotal time: {clustering_time + metrics_time:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()