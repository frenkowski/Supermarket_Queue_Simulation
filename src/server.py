import os

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np

from model import SupermarketModel, ObstacleAgent, CashierAgent


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

    return {}

with open(os.path.join(os.getcwd(), '../resources', 'map1.txt')) as f:
    world = [list(c) for c in f.read().split('\n')[::-1] if c]

width = len(world[0])
height = len(world)

grid = CanvasGrid(agent_portrayal, width, height, width*40, height*40)
server = ModularServer(SupermarketModel,
                       [grid],
                       "Supermarket Model",
                       {"N":10, "world": world, "width": width, "height": height})
server.port = 8521 # The default
server.launch()
