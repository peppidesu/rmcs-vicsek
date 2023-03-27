import numpy as np
import pygame
import prefs
from numba import config
import vicsek
import utils
import json
import predator

def update_preys(points, directions, aliveMask, predatorPos):

    return vicsek.update(
        points, 
        directions,
        aliveMask,
        predatorPos
    )


def draw(screen, points, predatorPos, aliveMask):
    pygame.draw.circle(screen, (0,16,32), predatorPos, prefs.detectionRadius)    
    pygame.draw.circle(screen, (32,0,0), predatorPos, prefs.predatorKillRadius)    
    for i in range(0, prefs.agentCount, 1):
        point = (int(points[i][0]), int(points[i][1]))       
        color = (255, 255, 255) if aliveMask[i] else (255,0,0)
        pygame.draw.circle(screen, color, point, 2)

    
    pygame.draw.circle(screen, (255,255,0), predatorPos, prefs.predatorKillRadius - 4)

def main():
    config.THREADING_LAYER = 'threadsafe'

    if prefs.randomSeed != None:
        np.random.seed(prefs.randomSeed)

    preyPositions = utils.generateRandomPoints(prefs.agentCount)
    preyDirections = utils.generateNormalizedDirections(prefs.agentCount)
    aliveMask = np.full(prefs.agentCount, True)
    
    predatorPos = utils.generateRandomPoints(1)[0]
    predatorDir = utils.generateNormalizedDirections(1)[0]
    satisfaction = 0

    # pygame boilerplate ._.
    pygame.init()
    screen = pygame.display.set_mode((prefs.width, prefs.height))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))
        
        preyPositions, preyDirections = update_preys(preyPositions, preyDirections, aliveMask, predatorPos)
        predatorPos, predatorDir, satisfaction, aliveMask = predator.update_predator(predatorPos, predatorDir, satisfaction, aliveMask, preyPositions)
        draw(screen, preyPositions, predatorPos, aliveMask)
        
        # update screen
        pygame.display.update()
    


    

timesteps = 50000
groups = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
samples = 20
slices = 20

def simulate():
    result = []
    preyPositions = utils.generateRandomPoints(prefs.agentCount)
    preyDirections = utils.generateNormalizedDirections(prefs.agentCount)
    aliveMask = np.full(prefs.agentCount, True)
    predatorPos = utils.generateRandomPoints(1)[0]
    predatorDir = utils.generateNormalizedDirections(1)[0]
    satisfaction = 0   

    for i in range(timesteps):
        preyPositions, preyDirections = update_preys(preyPositions, preyDirections, aliveMask, predatorPos)
        predatorPos, predatorDir, satisfaction, aliveMask = predator.update_predator(predatorPos, predatorDir, satisfaction, aliveMask, preyPositions)
        if i % 500 == 0:                
            print(f"\033[1A- Progress: {100 * i/timesteps:0.2f} %  ")
        if i % (timesteps // slices) == 0:
            result.append(sum(map(int, aliveMask)))
    result.append(sum(map(int, aliveMask)))
    return result

def main_crunch():
    print("\033[2J")
    config.THREADING_LAYER = 'threadsafe'

    results = {}
    timelabels = list(
        map(
            lambda x: f"T = {x}", 
            list(range(0, timesteps, timesteps // slices)) + [timesteps]
        )
    )
    dictify_time = lambda x: dict(zip(timelabels, x))
    dictify_time_t = lambda x: list(map(dictify_time, x))
    for i,g in enumerate(groups):
        prefs.flockingContribution = g   
        subresults = np.empty((samples, slices+1))
        print(f"Taking samples for alpha = {g} ...")
        for j in range(samples):
            print(f"Sample {j+1}/{samples}")
            print()
            np.random.seed(j)
            subresults[j] = simulate()

        results[f"alpha = {g}"] = {
            "data": dictify_time_t(subresults.tolist()),
            "min":  dictify_time(np.min(subresults, axis=0).tolist()),
            "max": dictify_time(np.max(subresults, axis=0).tolist()),
            "mean": dictify_time(np.mean(subresults, axis=0).tolist()),
            "std": dictify_time(np.std(subresults, axis=0).tolist()),
            "median": dictify_time(np.median(subresults, axis=0).tolist()),
            "25p": dictify_time(np.percentile(subresults, 25, axis=0).tolist()),
            "75p": dictify_time(np.percentile(subresults, 75, axis=0).tolist())
        }

        
    with open("data.json", "w") as file:
        file.write(json.dumps(
            {                
                "timesteps": timesteps,                

                "results": results,
            }, 
            indent=4))

if __name__ == "__main__":
    main_crunch()