import os

import colorama
from mesa.visualization.modules.ChartVisualization import ChartModule
from mesa.visualization.modules.PieChartVisualization import PieChartModule
from mesa.visualization.UserParam import UserSettableParameter

from enums import QueueType
from model import CashierAgent, CashRegisterAgent, CustomerAgent, ObstacleAgent, SupermarketModel
from server import CustomModularServer
from visualization.canvas_grid_with_terrain import CanvasGridWithTerrain


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
            "sprite": "images/characters/cashier" if agent.working else None,
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
        }

    if isinstance(agent, CashRegisterAgent):
        return {
            "Shape": "sprite",
            "sprite": "images/characters/cash-register" if agent.open else None,
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
        }

    if isinstance(agent, CustomerAgent):
        return {
            "Shape": "sprite",
            "sprite": agent.sprite,
            "Filled": "true",
            "Layer": 1,
            # "text": agent.unique_id,
            # "text_color": "white"
        }

    raise Exception('Undefined render function for agent \'{}\''.format(type(agent)))


colorama.init(autoreset=True)
queue_type = UserSettableParameter('choice', 'Queue Type', value=QueueType.CLASSIC.name,
                                   choices=[QueueType.CLASSIC.name, QueueType.SNAKE.name])

with open(os.path.join(os.getcwd(), '..', 'resources', 'map.txt')) as f:
    width, height = map(int, f.readline().strip().split(' '))

tile_size = 24
terrain_map_name = 'map'
grid = CanvasGridWithTerrain(agent_portrayal, width, height, terrain_map_name, width * tile_size, height * tile_size)

# Label MUST match with value of model variables added to data collector.
num_agents_phases_piechart = PieChartModule([{"Label": "Shopping",
                                              "Color": "#008000"},
                                             {"Label": "Queued",
                                              "Color": "#ff8c00"},
                                             {"Label": "Paying",
                                              "Color": "#dc143c"}
                                             ], 300, 300, data_collector_name='datacollector')

agent_in_queue_chart = ChartModule([{"Label": "Queued", "Color": "#ff8c00"},
                                    {"Label": "Queued (AVG)", "Color": "#00008b"}
                                    ], data_collector_name='datacollector')

avg_agent_time_in_queue_chart = ChartModule([{"Label": "Queued Time (AVG)", "Color": "#008b8b"}],
                                            data_collector_name='datacollector')

server = CustomModularServer(
    SupermarketModel,
    [grid, num_agents_phases_piechart, agent_in_queue_chart, avg_agent_time_in_queue_chart],
    "Supermarket Model",
    {
        "type": queue_type,
    }
)

print(server.settings)
server.port = 8521  # The default
server.launch()
