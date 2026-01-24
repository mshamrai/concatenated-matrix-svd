import h5py
import numpy as np
import glob
import os
import random
from .base import BaseDataReader


class PDEBenchDataReader(BaseDataReader):
    """Data reader for PDEBench HDF5 data."""
    
    def __init__(self, path: str, max_samples: int = None, random_seed: int = 42):
        """
        Initialize the PDEBench data reader.
        
        Args:
            path: Path to directory containing HDF5 files.
            max_samples: Maximum number of samples to read (None for all).
            random_seed: Random seed for sampling (default: 42).
        """
        self.path = path
        self.max_samples = max_samples
        self.random_seed = random_seed
    
    def read(self) -> np.ndarray:
        """
        Read and return PDEBench data as a numpy array.
        
        Returns:
            np.ndarray: Array of tensors stacked along first dimension.
        """
        files = sorted(glob.glob(os.path.join(self.path, "*.hdf5")))
        
        if not files:
            raise RuntimeError(f"No HDF5 files found in {self.path}")
        
        # Randomly shuffle files for random sampling
        random.seed(self.random_seed)
        random.shuffle(files)
        
        tensors = []
        total_samples = 0
        
        for fname in files:
            with h5py.File(fname, "r") as f:
                tensor = np.array(f["tensor"])
                
                # If we need specific number of samples
                if self.max_samples is not None:
                    remaining = self.max_samples - total_samples
                    if remaining <= 0:
                        break
                    
                    # If this file has more samples than we need, randomly sample from it
                    if tensor.shape[0] > remaining:
                        indices = random.sample(range(tensor.shape[0]), remaining)
                        tensor = tensor[sorted(indices)]
                    
                tensors.append(tensor)
                total_samples += tensor.shape[0]
                
                # Stop if we have enough samples
                if self.max_samples is not None and total_samples >= self.max_samples:
                    break
        
        data = np.concatenate(tensors, axis=0)
        
        return data