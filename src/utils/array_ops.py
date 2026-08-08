import numpy as np
from numpy.typing import NDArray

def normalize(array: NDArray):
    return array / np.linalg.norm(array, axis=-1, keepdims=True)
