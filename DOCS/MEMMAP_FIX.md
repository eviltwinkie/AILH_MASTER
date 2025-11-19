# Fix for "cannot mmap an empty file" Error

## Problem
```python
ValueError: cannot mmap an empty file
```

This occurs when trying to create a numpy memmap on a file that doesn't exist or is empty.

## Solution

Replace this:
```python
mel_memmap = np.memmap(mel_memmap_path, dtype=np.float32, mode="w+", shape=mel_shape)
```

With this:
```python
# Pre-allocate file before mmapping
import os
from pathlib import Path

# Ensure directory exists
Path(mel_memmap_path).parent.mkdir(parents=True, exist_ok=True)

# Calculate total bytes needed
total_bytes = int(np.prod(mel_shape)) * np.dtype(np.float32).itemsize

# Pre-allocate file to correct size
with open(mel_memmap_path, 'wb') as f:
    f.seek(total_bytes - 1)
    f.write(b'\0')

# Now mmap will work (use r+ mode, not w+)
mel_memmap = np.memmap(mel_memmap_path, dtype=np.float32, mode="r+", shape=mel_shape)
```

## Or Use the Helper Function

Copy this function to your file:
```python
from pathlib import Path
from typing import Tuple, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)

def safe_init_memmap(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.dtype(np.float32),
    mode: Literal["r+", "r", "w+", "c"] = "w+",
) -> np.memmap:
    """
    Create a NumPy memmap ensuring the requested shape is non-empty.
    
    Args:
        path: Path to memmap file
        shape: Shape of the array
        dtype: Data type (default: float32)
        mode: File mode (default: w+)
    
    Returns:
        np.memmap: Memory-mapped array
        
    Raises:
        RuntimeError: If shape is empty/invalid
    """
    total_elems = int(np.prod(shape))
    if total_elems <= 0:
        msg = (
            f"Attempted to create zero-sized memmap at {path} "
            f"with shape={shape}, total_elems={total_elems}. "
            "Likely cause: no WAV files / segments discovered."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Allocating memmap at %s with shape=%s, dtype=%s", path, shape, dtype)
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)
```

Then use it:
```python
from pathlib import Path

mel_memmap_path = Path("/path/to/output/PIPELINE_FEATURES.DAT")
mel_memmap = safe_init_memmap(
    mel_memmap_path, 
    mel_shape, 
    dtype=np.dtype(np.float32), 
    mode="w+"
)
```

## Why This Happens

NumPy's `memmap` with `mode="w+"` tries to:
1. Open/create the file
2. Memory-map it

But `mmap.mmap()` fails if the file is empty (0 bytes). The file needs to exist with the correct size before mmapping.

## Notes

- The `safe_init_memmap()` function handles this automatically
- Your working pipeline.py already uses this function
- NumPy should handle this internally but doesn't in all cases
