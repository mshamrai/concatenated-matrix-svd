import os
import glob
import random
import numpy as np
import rasterio
from rasterio.enums import Resampling
from .base import BaseDataReader


class BigEarthNetDataReader(BaseDataReader):
    """Data reader for BigEarthNet-S1 satellite data."""
    
    def __init__(
        self, 
        root: str = "data/BigEarthNet-S1",
        channels: list = None,
        max_samples: int = 5000,
        layout: str = "NHWC",
        random_seed: int = 42
    ):
        """
        Initialize the BigEarthNet data reader.
        
        Args:
            root: Folder that contains patch folders like S1_BEN_.../
            channels: List of channels to load (default: ["VV", "VH"])
            max_samples: Maximum number of samples to load (None for all)
            layout: Output layout - "NHWC", "NCHW", "HWCN", or "CHWN"
            random_seed: Random seed for sampling (default: 42)
        """
        self.root = root
        self.channels = channels if channels is not None else ["VV", "VH"]
        self.max_samples = max_samples
        self.layout = layout
        self.random_seed = random_seed
    
    def read(self) -> np.ndarray:
        """
        Read and return BigEarthNet-S1 data as a numpy array.
        
        Returns:
            np.ndarray: The data array with shape depending on layout parameter.
        """
        patch_dirs = self._find_patch_dirs()
        
        # Randomly sample if max_samples is specified
        if self.max_samples is not None and self.max_samples < len(patch_dirs):
            random.seed(self.random_seed)
            patch_dirs = random.sample(patch_dirs, self.max_samples)

        samples = []
        for d in patch_dirs:
            chans = []
            for ch in self.channels:
                tifs = glob.glob(os.path.join(d, f"*_{ch}.tif"))
                if not tifs:
                    break
                band = self._read_band_tif(tifs[0])
                chans.append(band)
            if len(chans) != len(self.channels):
                continue  # skip incomplete sample
            # stack H,W,C
            hwc = np.stack(chans, axis=-1)
            samples.append(hwc.astype("float32"))

        if not samples:
            raise RuntimeError("No valid S1 patches found. Check root path and file naming.")

        X = np.stack(samples, axis=0)  # (N,H,W,C)
        
        if self.layout == "NHWC":
            return X
        elif self.layout == "NCHW":
            return np.transpose(X, (0, 3, 1, 2))
        elif self.layout == "HWCN":
            return np.transpose(X, (1, 2, 3, 0))
        elif self.layout == "CHWN":
            return np.transpose(X, (3, 1, 2, 0))
        else:
            raise ValueError(f"Unknown layout: {self.layout}")
    
    def _find_patch_dirs(self):
        """Find BigEarthNet-S1 patch directories containing required channel TIF files."""
        patch_dirs = []
        for d in os.listdir(self.root):
            part = os.path.join(self.root, d)
            if not os.path.isdir(part):
                continue
            for sub in os.listdir(part):
                full = os.path.join(part, sub)
                if not os.path.isdir(full):
                    continue
                if (glob.glob(os.path.join(full, "*_VV.tif")) and
                    glob.glob(os.path.join(full, "*_VH.tif"))):
                    patch_dirs.append(full)
        print(f"Found {len(patch_dirs)} patch directories in {self.root}")
        return patch_dirs
    
    @staticmethod
    def _read_band_tif(path, hw=None):
        """Read a single band TIF file and handle nodata values."""
        with rasterio.open(path) as src:
            if hw is not None and (src.height != hw[0] or src.width != hw[1]):
                data = src.read(
                    1, out_shape=(hw[0], hw[1]),
                    resampling=Resampling.bilinear
                )
            else:
                data = src.read(1)
            # handle nodata if present
            if src.nodata is not None:
                mask = (data == src.nodata)
                if mask.any():
                    data = data.astype("float32")
                    data[mask] = np.nan
            return data
