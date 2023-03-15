import numpy as np
from numba import njit, prange

import prefs


@njit(parallel=True)
def normalize_vectors(vs, n):    
    for i in prange(n):
        length = np.sqrt(vs[i][0] * vs[i][0] + vs[i][1] * vs[i][1])
        vs[i] /= length

def generateRandomPoints(n):
    result = np.random.rand(n, 2)

    # apply scaling
    result[:, 0] *= prefs.WIDTH
    result[:, 1] *= prefs.HEIGHT
    
    return result

def generateNormalizedDirections(n):    
    result = np.random.rand(n, 2) * 2 - 1        
    # normalize
    normalize_vectors(result, n)
    return result