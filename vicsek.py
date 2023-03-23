import pycuda.driver as cuda
import pycuda.autoinit # don't remove this!!!
import numpy as np

import kernelLoader
import prefs


mod = kernelLoader.load_from_file('./vicsek_kernel.cu')

vicsek_update = mod.get_function("vicsek_update")

def update(points, directions, aliveMask, predatorPos):
    points      = points.astype(np.float32)
    directions  = directions.astype(np.float32)
    aliveMask   = aliveMask.astype(np.bool8)
    predatorPos = predatorPos.astype(np.float32)

    new_points      = np.zeros_like(points)
    new_directions  = np.zeros_like(directions)
    
    vicsek_update(
        cuda.In(points), 
        cuda.In(directions), 
        cuda.In(aliveMask),
        cuda.Out(new_points), 
        cuda.Out(new_directions), 
        cuda.In(predatorPos),
        np.int32(prefs.agentCount), 
        np.float32(prefs.flockingRadius), 
        np.float32(prefs.detectionRadius),
        np.float32(prefs.flockingContribution), 
        np.float32(prefs.preyDamping),
        np.float32(prefs.timeStep), 
        np.float32(prefs.preySpeed),
        np.float32(prefs.directionNoiseFactor),
        np.float32(prefs.width), 
        np.float32(prefs.height),
        np.float32(prefs.wallRepellRadius),
        np.float32(prefs.wallRepellFactor),
        block=(512, 1, 1), 
        grid=(int(prefs.agentCount / 512) + 1, 1)
    )

    return new_points, new_directions