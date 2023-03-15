import pycuda.driver as cuda
import pycuda.autoinit # don't remove this!!!
import numpy as np

import kernelLoader
import prefs

mod = kernelLoader.load_from_file('./vicsek_kernel.cu')

vicsek_update = mod.get_function("vicsek_update")

def update(points, directions, eta):
    points     = points.astype(np.float32)
    directions = directions.astype(np.float32)

    new_points      = np.zeros_like(points)
    new_directions  = np.zeros_like(directions)
    
    eta = eta.astype(np.float32)
    
    vicsek_update(
        cuda.In(points), 
        cuda.In(directions), 
        cuda.Out(new_points), 
        cuda.Out(new_directions), 

        np.int32(prefs.AGENT_COUNT), 
        np.float32(prefs.RADIUS), 
        np.float32(prefs.ALPHA), 
        np.float32(prefs.DT), 
        cuda.In(eta), 
        np.float32(prefs.WIDTH), 
        np.float32(prefs.HEIGHT),
        np.float32(prefs.WALL_REPELL_RADIUS),
        np.float32(prefs.WALL_REPELL_FACTOR),
        block=(512, 1, 1), 
        grid=(int(prefs.AGENT_COUNT / 512) + 1, 1)
    )

    return new_points, new_directions