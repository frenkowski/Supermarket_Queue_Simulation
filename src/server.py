import os
import numpy as np

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from model import *


def agent_portrayal(agent):
    if isinstance(agent, ObstacleAgent):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 0.95,
            "h": 0.95,
            "Color": "black"
        }

    if isinstance(agent, CashierAgent):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 0.95,
            "h": 0.95,
            "Color": "green" if agent.open else "red"
        }

    if isinstance(agent, CustomerAgent):
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 0,
            "r": 0.8,
            "Color": "blue"
        }

    return {}


with open(os.path.join(os.getcwd(), '../resources', 'map2.txt')) as f:
    capacity = int(f.readline())
    world = [list(c) for c in f.read().split('\n')[::-1] if c]

width = len(world[0])
height = len(world)

grid = CanvasGrid(agent_portrayal, width, height, width*40, height*40)
server = ModularServer(SupermarketModel,
                       [grid],
                       "Supermarket Model",
                       {"N": capacity, "world": world, "width": width, "height": height})
server.port = 8521  # The default
server.launch()
