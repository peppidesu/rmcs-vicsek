import numpy as np
import pygame
import prefs
from numba import config
import vicsek
import utils
import cProfile
import pstats
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
    

def main_crunch():
    print("\033[2J")
    config.THREADING_LAYER = 'threadsafe'

    if prefs.randomSeed != None:
        np.random.seed(prefs.randomSeed)

    results = []
    prefs.flockingContribution = 0
    for i in range(11):
        preyPositions = utils.generateRandomPoints(prefs.agentCount)
        preyDirections = utils.generateNormalizedDirections(prefs.agentCount)
        aliveMask = np.full(prefs.agentCount, True)
    
        predatorPos = utils.generateRandomPoints(1)[0]
        predatorDir = utils.generateNormalizedDirections(1)[0]
        satisfaction = 0   
        for j in range(200000):
            preyPositions, preyDirections = update_preys(preyPositions, preyDirections, aliveMask, predatorPos)
            predatorPos, predatorDir, satisfaction, aliveMask = predator.update_predator(predatorPos, predatorDir, satisfaction, aliveMask, preyPositions)        
            if j % 500 == 0:                
                print(f"\033[1ASimulation {i}/10 - Progress: {j/2000:0.2f} %    ")
        results.append(sum(map(int, aliveMask)))
        prefs.flockingContribution += 0.1     
    print(results)

if __name__ == "__main__":
    main()