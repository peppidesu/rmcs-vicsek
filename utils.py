import numpy as np
from numba import njit, prange

import prefs

@njit
def vector_length(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1])

def normalize_vector(v):    
    return v / vector_length(v)    

@njit(parallel=True)
def normalize_vectors(vs, n):    
    for i in prange(n):
        length = vector_length(vs[i])
        vs[i] /= length



def generateRandomPoints(n):
    result = np.random.rand(n, 2)

    # apply scaling
    result[:, 0] *= prefs.width
    result[:, 1] *= prefs.height
    
    return result

def generateNormalizedDirections(n):    
    result = np.random.rand(n, 2) * 2 - 1        
    # normalize
    normalize_vectors(result, n)
    return result

