import numpy as np


def compute_error(cluster, r=None):
    m, n = cluster[0].shape
    if r is None:
        r = min(m, n)
    M = np.hstack(cluster)
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    r_c = min(r, len(S))
    S_r = S[:r_c]
    U_r = U[:, :r_c]
    VT_r = VT[:r_c, :]
    M_r = (U_r * S_r) @ VT_r
    compression_error = np.linalg.norm(M - M_r, 'fro') / np.linalg.norm(M, 'fro')
    return compression_error


def compute_errors(clusters, r=None):
    errors = []
    for cluster in clusters:
        error = compute_error(cluster, r)
        errors.append(error)
    return errors


def compute_effective_compression_ratio(clusters, r_target, blocks):
    m = blocks[0].shape[0]

    def compute_parameters(cluster, r):
        Kn = sum(block.shape[1] for block in cluster)
        if len(cluster) == 1:
            return Kn * m
        return min(r * (m + Kn), Kn * m)
    
    n_blocks = sum(block.shape[1] for block in blocks)
    
    return (m * n_blocks) / sum([compute_parameters(cluster, r_target) for cluster in clusters])


def filter_clusters(clusters, r_target):
    m = clusters[0][0].shape[0]
    clusters_filtered = []
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        Kn = sum(block.shape[1] for block in cluster)
        if r_target < Kn * m / (m + Kn):
            clusters_filtered.append(cluster)
    return clusters_filtered