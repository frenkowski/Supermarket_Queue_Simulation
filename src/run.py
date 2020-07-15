import os
import numpy as np

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules.ChartVisualization import ChartModule
from mesa.visualization.modules.TextVisualization import TextElement
from mesa.visualization.modules.PieChartVisualization import PieChartModule
import tornado

from server import CustomModularServer
from model import *

def agent_portrayal(agent):
    if isinstance(agent, ObstacleAgent):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 0.95,
            "h": 0.95,
            "Color": "black",
        }

    if isinstance(agent, CashierAgent):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 0.95,
            "h": 0.95,
            "Color": "green" if agent.open else "red",
        }

    if isinstance(agent, CustomerAgent):
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 0,
            "r": 0.8,
            "Color": "blue",
            "text": agent.unique_id,
            "text_color": "white"
        }

    return {}


with open(os.path.join(os.getcwd(), '..', 'resources', 'map2.txt')) as f:
    capacity, lane_switch_boundary = map(int, f.readline().strip().split(' '))
    world = [list(c) for c in f.read().split('\n') if c]

width = len(world[0])
height = len(world)

grid = CanvasGrid(agent_portrayal, width, height, width*40, height*40)

# Label MUST match with value of model variables added to data collector. 
piechart_agents_num_element = PieChartModule([{"Label": "Agent in queue", 
                                               "Color": "#AA0000"},
                                              {"Label": "Agent that shopping",
                                                  "Color": "#FF8040"},
                                              {"Label": "Agent in payment",
                                                  "Color": "#800000"}
                                             ], 250, 250,
                                  data_collector_name='datacollector')

agent_in_queue_chart = ChartModule([{"Label": "Agent in queue", "Color": "#AA0000"}, {"Label": "Avg. number of agent in queue", "Color": "#0000A0"}
                             ], data_collector_name='datacollector')

#avg_agent_in_queue_chart = ChartModule([{"Label": "Avg. number of agent in queue", "Color": "#0000A0"},
#                                    ], data_collector_name='datacollector')

avg_time_agent_in_queue_chart = ChartModule([{"Label": "Avg. time spent in queue", "Color": "#408080"},
                                        ], data_collector_name='datacollector')


server = CustomModularServer(
    SupermarketModel,
    [grid, piechart_agents_num_element, agent_in_queue_chart, avg_time_agent_in_queue_chart],
    "Supermarket Model",
    {"N": capacity, "B": lane_switch_boundary, "world": world, "width": width, "height": height}
)

print(server.settings)
server.port = 8521  # The default
server.launch()
