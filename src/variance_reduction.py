
import numpy as np

def antithetic_pairs(Z: np.ndarray) -> np.ndarray:
    """
    Given standard normal draws Z, return an array with antithetic pairing [Z, -Z].
    If len(Z) is N, returns 2N samples.
    """
    return np.concatenate([Z, -Z])
