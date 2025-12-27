"""
Utility functions for the ranking package.
"""

import gc
from typing import Optional


def clear_gpu_memory() -> bool:
    """
    Clear GPU memory to avoid OOM errors.
    
    Returns:
        True if GPU memory was cleared successfully, False otherwise.
    """
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        print("✓ GPU memory cleared")
        return True
    except ImportError:
        # cupy not available
        return False
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")
        return False


def check_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available for computation.
    
    Returns:
        True if GPU is available, False otherwise.
    """
    try:
        import cudf
        import cugraph
        return True
    except ImportError:
        return False


# Common stop words for text processing
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
    'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
}
