import os
import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
import tornado

from visualization.canvas_grid_with_terrain import CanvasGridWithTerrain
from server import CustomModularServer
from model import *


def agent_portrayal(agent):
    if isinstance(agent, ObstacleAgent):
        return {}
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
            "text": agent.unique_id,
            "text_color": "white",
            "Color": "#00000044",
        }

    if isinstance(agent, CashierAgent):
        return {
            "Shape": "sprite",
            "sprite": "images/characters/cashier" if agent.open else None,
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
            "Color": "#0000bc44" if agent.open else "#bc000044",
        }

    if isinstance(agent, CustomerAgent):
        return {
            "Shape": "sprite",
            "sprite": agent.sprite,
            "Filled": "true",
            "Layer": 1,
            "Color": "blue",
            # "text": agent.unique_id,
            # "text_color": "white"
        }

    raise Exception('Undefined render function for agent \'{}\''.format(type(agent)))


with open(os.path.join(os.getcwd(), '..', 'resources', 'map3-snake.txt')) as f:
    capacity, lane_switch_boundary = map(int, f.readline().strip().split(' '))
    terrain_map_name = f.readline().strip()
    world = [list(c) for c in f.read().split('\n') if c]

width = len(world[0])
height = len(world)
tile_size = 24

grid = CanvasGridWithTerrain(agent_portrayal, width, height, terrain_map_name, width*tile_size, height*tile_size)
server = CustomModularServer(
    SupermarketModel,
    [grid],
    "Supermarket Model",
    {"N": capacity, "B": lane_switch_boundary, "world": world, "width": width, "height": height}
)

print(server.settings)
server.port = 8521  # The default
server.launch()
