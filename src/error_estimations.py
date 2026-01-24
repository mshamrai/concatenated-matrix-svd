"""
Script to compute error estimations for clustering algorithms.

This script evaluates different error estimation methods:
1. True error (ground truth from SVD)
2. Weyl bound (max norm method)
3. Residual bound (residual method with norm ordering)
4. Incremental estimate with norm ordering (approximate method)
5. Incremental estimate with residual ordering (approximate method)
"""

import argparse
import csv
import numpy as np
from typing import List
import random

from src.clustering.max_norm import MaxNormClusterAlgorithm
from src.clustering.residuals import ResidualsClusterAlgorithm
from src.clustering.approximate import ApproximateClusterAlgorithm


def compute_true_error(blocks: List[np.ndarray], r_target: int) -> float:
    """
    Compute the true reconstruction error using SVD.
    
    Args:
        blocks: List of data blocks to cluster.
        r_target: Target rank for approximation.
        
    Returns:
        True relative reconstruction error.
    """
    M = np.hstack(blocks)
    norm_M = np.linalg.norm(M, 'fro')
    
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    r_c = min(r_target, len(S))
    S_r = S[:r_c]
    U_r = U[:, :r_c]
    VT_r = VT[:r_c, :]
    M_r = (U_r * S_r) @ VT_r
    
    error = np.linalg.norm(M - M_r, 'fro') / norm_M
    return error


def compute_all_estimates(blocks: List[np.ndarray], r_target: int, 
                          cluster_size: int) -> dict:
    """
    Compute all error estimates for a given cluster.
    
    Args:
        blocks: List of data blocks.
        r_target: Target rank.
        cluster_size: Size of cluster.
        
    Returns:
        Dictionary with all error estimates.
    """
    return {
        'cluster_size': cluster_size,
        'true_error': compute_true_error(blocks, r_target),
        'weyl_bound': MaxNormClusterAlgorithm(r_target=r_target).compute_error_bound(blocks),
        'residual_bound_norm': ResidualsClusterAlgorithm(r_target=r_target, sorting_strategy='norm').compute_error_bound(
            blocks
        ),
        'residual_bound_residual': ResidualsClusterAlgorithm(r_target=r_target, sorting_strategy='residual').compute_error_bound(
            blocks
        ),
        'incremental_norm': ApproximateClusterAlgorithm(r_target=r_target, sorting_strategy='norm').compute_error_bound(
            blocks
        ),
        'incremental_residual': ApproximateClusterAlgorithm(r_target=r_target, sorting_strategy='residual').compute_error_bound(
            blocks
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute clustering error estimations')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['qualcomm', 'bigearth', 'pdebench', 'smolvlm'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data directory')
    
    # Experiment parameters
    parser.add_argument('--r_target', type=int, default=10,
                        help='Target rank for approximation')
    parser.add_argument('--min_cluster_size', type=int, default=2,
                        help='Minimum cluster size to evaluate')
    parser.add_argument('--max_cluster_size', type=int, default=50,
                        help='Maximum cluster size to evaluate')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for cluster sizes')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of random trials per cluster size')
    parser.add_argument('--random_seed', type=int, default=23,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='error_estimations.csv',
                        help='Output CSV file path')
    
    # Dataset-specific arguments
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples for bigearth/pdebench')
    parser.add_argument('--layout', type=str, default='NHWC',
                        help='Layout for bigearth')
    parser.add_argument('--model_path', type=str, default='HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
                        help='Model path for SmolVLM')
    parser.add_argument('--target_dim', type=int, default=768,
                        help='Target dimension for SmolVLM')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.dataset} dataset from {args.data_path}")
    from src.data import get_data_reader
    from src.main import prepare_blocks
    
    data_kwargs = {}
    if args.dataset == 'qualcomm':
        data_kwargs = {'data_path': args.data_path}
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
    
    data_reader = get_data_reader(args.dataset, **data_kwargs)
    data = data_reader.read()
    blocks = prepare_blocks(data, args.dataset)
    
    print(f"Prepared {len(blocks)} blocks")
    print(f"Block shape: {blocks[0].shape}")
    
    # Run experiments
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    results = []
    cluster_sizes = range(args.min_cluster_size, args.max_cluster_size + 1, args.step)
    
    print(f"\nRunning error estimation experiments...")
    print(f"Cluster sizes: {args.min_cluster_size} to {args.max_cluster_size} (step {args.step})")
    print(f"Trials per size: {args.num_trials}")
    print(f"Target rank: {args.r_target}")
    
    for cluster_size in cluster_sizes:
        if cluster_size > len(blocks):
            print(f"Skipping cluster size {cluster_size} (exceeds available blocks)")
            break
        
        print(f"\nProcessing cluster size {cluster_size}...")
        
        for trial in range(args.num_trials):
            # Randomly sample blocks
            sampled_blocks = random.sample(blocks, cluster_size)
            
            # Compute all estimates
            result = compute_all_estimates(sampled_blocks, args.r_target, cluster_size)
            result['trial'] = trial
            results.append(result)
            
            # Write result to CSV incrementally
            if trial == 0 and cluster_size == args.min_cluster_size:
                # First trial: create file and write header
                fieldnames = ['cluster_size', 'trial', 'true_error', 'weyl_bound', 
                              'residual_bound_norm', 'residual_bound_residual',
                              'incremental_norm', 'incremental_residual']
                with open(args.output, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(result)
            else:
                # Append to existing file
                with open(args.output, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(result)
            
            print(f"  Completed {trial + 1}/{args.num_trials} trials")
    
    # Results already saved incrementally
    print(f"\nSaved {len(results)} results to {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
