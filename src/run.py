import os

import colorama
from mesa.visualization.modules.ChartVisualization import ChartModule
from mesa.visualization.modules.PieChartVisualization import PieChartModule
from mesa.visualization.UserParam import UserSettableParameter

from model import CashierAgent, CustomerAgent, MapSize, ObstacleAgent, QueueType, SupermarketModel
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
            "sprite": "images/characters/cashier" if agent.open else None,
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

map_size = UserSettableParameter('choice', 'Map Size', value=MapSize.SMALL.name,
                                 choices=[MapSize.SMALL.name, MapSize.MEDIUM.name, MapSize.LARGE.name])

with open(os.path.join(os.getcwd(), '..', 'resources', 'map3-snake.txt')) as f:
    capacity, lane_switch_boundary = map(int, f.readline().strip().split(' '))
    terrain_map_name = f.readline().strip()
    world = [list(c) for c in f.read().split('\n') if c]

width = len(world[0])
height = len(world)
tile_size = 24

grid = CanvasGridWithTerrain(agent_portrayal, width, height, terrain_map_name, width * tile_size, height * tile_size)

# Label MUST match with value of model variables added to data collector.
piechart_agents_num_element = PieChartModule([{"Label": "Queued",
                                               "Color": "#AA0000"},
                                              {"Label": "Shopping",
                                                  "Color": "#FF8040"},
                                              {"Label": "Paying",
                                                  "Color": "#800000"}
                                              ], 300, 300, data_collector_name='datacollector')

agent_in_queue_chart = ChartModule([{"Label": "Queued", "Color": "#AA0000"},
                                    {"Label": "Queued (AVG)", "Color": "#0000A0"}
                                    ], data_collector_name='datacollector')

avg_time_agent_in_queue_chart = ChartModule([{"Label": "Queued Time (AVG)", "Color": "#408080"}],
                                            data_collector_name='datacollector')

server = CustomModularServer(
    SupermarketModel,
    [grid, piechart_agents_num_element, agent_in_queue_chart, avg_time_agent_in_queue_chart],
    "Supermarket Model",
    {
        "capacity": capacity,
        "boundary": lane_switch_boundary,
        "world": world,
        "width": width,
        "height": height,
        "type": queue_type,
        "terrain_map_name": terrain_map_name
    }
)

print(server.settings)
server.port = 8521  # The default
server.launch()
