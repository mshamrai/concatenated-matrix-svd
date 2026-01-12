import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseDataReader


class SmolVLMDataReader(BaseDataReader):
    """Data reader for SmolVLM model weights."""
    
    def __init__(
        self,
        model_path: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        target_dim: int = 768,
        min_params: int = 2,
        dtype: str = "float32"
    ):
        """
        Initialize the SmolVLM data reader.
        
        Args:
            model_path: Path or name of the HuggingFace model to load.
            target_dim: Target dimension for reshaping weight tensors (default: 768).
            min_params: Minimum number of dimensions for a weight to be included (default: 2).
            dtype: Data type for model loading (default: "float32").
        """
        self.model_path = model_path
        self.target_dim = target_dim
        self.min_params = min_params
        self.dtype = getattr(torch, dtype)
    
    def read(self) -> np.ndarray:
        """
        Read and return SmolVLM model weights as a numpy array.
        
        Returns:
            np.ndarray: Array of weight blocks with shape (K, target_dim, ...).
        """
        print(f"Loading SmolVLM model from {self.model_path}")
        
        # Load model
        processor = AutoProcessor.from_pretrained(self.model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )
        
        # Extract weights
        blocks = []
        state_dict = model.state_dict()
        
        for weight_name in state_dict:
            weight_tensor = state_dict[weight_name]
            
            # Skip tensors with too few dimensions
            if len(weight_tensor.shape) < self.min_params:
                continue
            
            # Reshape to (-1, target_dim, 1) and convert to numpy
            reshaped = weight_tensor.reshape(-1, self.target_dim, 1).cpu().numpy()
            blocks.append(reshaped)
        
        print(f"Extracted {len(blocks)} weight blocks from model")
        
        # Concatenate all blocks along the first dimension
        concatenated = np.concatenate(blocks, axis=0)
        
        return concatenated
