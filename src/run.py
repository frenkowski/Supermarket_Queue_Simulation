import os
import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules.ChartVisualization import ChartModule
from mesa.visualization.modules.TextVisualization import TextElement
from mesa.visualization.modules.PieChartVisualization import PieChartModule
from mesa.visualization.UserParam import UserSettableParameter

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

queueType = UserSettableParameter('choice', 'Queue', value=QueueType.CLASSIC.name,
                                  choices=[QueueType.CLASSIC.name, QueueType.SNAKE.name])

with open(os.path.join(os.getcwd(), '..', 'resources', 'map3.txt')) as f:
    capacity, lane_switch_boundary = map(int, f.readline().strip().split(' '))
    terrain_map_name = f.readline().strip()
    world = [list(c) for c in f.read().split('\n') if c]

width = len(world[0])
height = len(world)
tile_size = 24

grid = CanvasGridWithTerrain(agent_portrayal, width, height, terrain_map_name, width*tile_size, height*tile_size)

# Label MUST match with value of model variables added to data collector.
piechart_agents_num_element = PieChartModule([{"Label": "Agent in queue",
                                               "Color": "#AA0000"},
                                              {"Label": "Agent that shopping",
                                                  "Color": "#FF8040"},
                                              {"Label": "Agent in payment",
                                                  "Color": "#800000"}
                                             ], 300, 300,
                                  data_collector_name='datacollector')

agent_in_queue_chart = ChartModule([{"Label": "Agent in queue", "Color": "#AA0000"},
                                    {"Label": "Avg. number of agent in queue", "Color": "#0000A0"}
                                    ], data_collector_name='datacollector')

avg_time_agent_in_queue_chart = ChartModule([{"Label": "Avg. time spent in queue", "Color": "#408080"},
                                        ], data_collector_name='datacollector')

server = CustomModularServer(
    SupermarketModel,
    [grid, piechart_agents_num_element, agent_in_queue_chart, avg_time_agent_in_queue_chart],
    "Supermarket Model",
    {"N": capacity, "B": lane_switch_boundary, "world": world,
        "width": width, "height": height, "Q": queueType, "terrain_map_name": terrain_map_name }
)

print(server.settings)
server.port = 8521  # The default
server.launch()
