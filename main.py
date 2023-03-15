import numpy as np
import pygame
import prefs
from numba import njit, config, threading_layer, prange
import vicsek
import utils

points = None
directions = None

def init():
    points = utils.generateRandomPoints(prefs.AGENT_COUNT)
    directions = utils.generateNormalizedDirections(prefs.AGENT_COUNT)

def loop():
    noise = np.random.rand(prefs.AGENT_COUNT, 2) * 2 - 1
    noise *= prefs.NOISE_FACTOR

    points, directions = vicsek.update(
        points, 
        directions, 
        noise
    )

def draw():
    # draw a circle
    for i in range(prefs.AGENT_COUNT):
        point = (int(points[i][0]), int(points[i][1]))       
        pygame.draw.circle(screen, (255, 255, 255), point, 2)
    
if __name__ == "__main__":
    config.THREADING_LAYER = 'threadsafe'

    if prefs.SEED != None:
        np.random.seed(prefs.SEED)

    init()

    # pygame boilerplate ._.
    pygame.init()
    screen = pygame.display.set_mode((prefs.WIDTH, prefs.HEIGHT))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))

        loop()
        draw()
        
        # update screen
        pygame.display.update()