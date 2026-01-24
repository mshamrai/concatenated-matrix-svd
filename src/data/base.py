from abc import ABC, abstractmethod
import numpy as np

class BaseDataReader(ABC):
    """Abstract base class for data readers."""
    
    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Read and return data as a numpy array.
        
        Returns:
            np.ndarray: The data array.
        """
        pass