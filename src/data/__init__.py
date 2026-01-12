from .qualcomm import QualcommDataReader
from .bigearth import BigEarthNetDataReader 
from .pdebench import PDEBenchDataReader
from .smolvlm import SmolVLMDataReader


def get_data_reader(dataset, **kwargs):
    """Factory function to get appropriate data reader."""
    readers = {
        'qualcomm': QualcommDataReader,
        'bigearth': BigEarthNetDataReader,
        'pdebench': PDEBenchDataReader,
        'smolvlm': SmolVLMDataReader,
    }
    
    if dataset not in readers:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(readers.keys())}")
    
    return readers[dataset](**kwargs)