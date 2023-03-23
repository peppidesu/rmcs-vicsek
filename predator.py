import numpy as np
from numba import njit, prange

import prefs
from utils import vector_length, normalize_vector

@njit(parallel=True)
def distancesToPoint(points, target, n):
    distances = np.zeros(n)
    for i in prange(n):
        distances[i] = vector_length(points[i] - target)
    return distances

def findClosestAgent(predator, points, aliveMask, n):    
    distances = distancesToPoint(points, predator, n)[aliveMask]
    indices = np.arange(n)[aliveMask]
    closest = np.argmin(distances)
    return indices[closest], distances[closest]


def update_predator(predatorPos, predatorDir, satisfaction, aliveMask, preys):
    targetDirection = predatorDir
    if (satisfaction <= 0):        
        closestIdx, closestDistance = findClosestAgent(predatorPos, preys, aliveMask, prefs.agentCount)            
        if (closestDistance < prefs.predatorKillRadius):
            aliveMask[closestIdx] = False
            satisfaction = prefs.predatorSatisfactionDelay
        else: 
            targetDirection = (preys[closestIdx] - predatorPos) / closestDistance                        
            
    else:
        satisfaction -= prefs.timeStep
    
    predatorDir = prefs.predatorDamping * targetDirection + (1 - prefs.predatorDamping) * predatorDir

    netWallFactor = (1.0/prefs.wallRepellRadius) * prefs.wallRepellFactor * prefs.timeStep
    predatorDir[0] -= (
        max(predatorPos[0] - prefs.width + prefs.wallRepellRadius, 0) 
        * netWallFactor
    )
    predatorDir[0] += (
        max(prefs.wallRepellRadius - predatorPos[0], 0) 
        * netWallFactor
    )
    predatorDir[1] -= (
        max(predatorPos[1] - prefs.height + prefs.wallRepellRadius, 0) 
        * netWallFactor
    )
    predatorDir[1] += (
        max(prefs.wallRepellRadius - predatorPos[1], 0) 
        * netWallFactor
    )

    predatorDir = normalize_vector(predatorDir)

    predatorPos += predatorDir * prefs.predatorSpeed * prefs.timeStep

    return predatorPos, predatorDir, satisfaction, aliveMask
