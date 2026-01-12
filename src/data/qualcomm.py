import numpy as np
import scipy.io as sio
from glob import glob
from .base import BaseDataReader


class QualcommDataReader(BaseDataReader):
    """Data reader for Qualcomm UE*.mat files."""
    
    def __init__(self, data_path: str, reshape: bool = True, to_real: bool = True):
        """
        Initialize the Qualcomm data reader.
        
        Args:
            data_path: Path to directory containing UE*.mat files.
            reshape: Whether to reshape the data.
            to_real: Whether to convert complex data to real by stacking real and imaginary parts.
        """
        self.data_path = data_path
        self.reshape = reshape
        self.to_real = to_real
    
    def read(self) -> np.ndarray:
        """
        Read and return Qualcomm data as a numpy array.
        
        Returns:
            np.ndarray: Array of blocks with shape (K, ...) where K is the number of UE files.
        """
        mat_files = sorted(glob(f'{self.data_path}/UE*.mat'))        # UE1.mat, UE2.mat, ...
        if not mat_files:
            raise RuntimeError("No UE*.mat files found in the current directory.")

        K = len(mat_files)                         # number of blocks available
        print(f"Found {K} UE files → {K} blocks")

        blocks = []
        for p in mat_files:
            Hf = sio.loadmat(p, squeeze_me=True, variable_names='Hf')['Hf']
            if self.reshape:
                Hf = self._reshape_hf(Hf)
            if self.to_real:
                Hf = self._to_real_hf(Hf)
            blocks.append(Hf)
        
        return np.array(blocks)
    
    @staticmethod
    def _to_real_hf(A):
        """Convert complex array to real by stacking real and imaginary parts."""
        return np.vstack([A.real, A.imag])
    
    @staticmethod
    def _reshape_hf(Hf):
        """Reshape Hf array from (NRB, Nr, Nt, Nc) to (Nr*Nt*Nc, NRB)."""
        NRB, Nr, Nt, Nc = Hf.shape
        return Hf.reshape(Nr*Nt*Nc, NRB)