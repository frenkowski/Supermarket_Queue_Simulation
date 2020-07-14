from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules.ChartVisualization import ChartModule
from mesa.visualization.modules.TextVisualization import TextElement

import numpy as np
from scipy.spatial import distance
from enum import Enum

class ObstacleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass


class CashierAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.open = self.random.random() > 0.5
        if self.open:
            model.open_cashier += 1

    def step(self):
        pass


class CustomerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.n_prod = self.random.randint(1, 51)

        self.shopping_time = Counter(self.n_prod)
        self.paying_time = Counter(self.n_prod)
        self.phase = AgentPhase.SHOPPING
        self.objective = None
        
        # Dictionary for counting number of step in each phase of Supermarket.
        self.step_for_phase = {
            AgentPhase.PAYING: 0,
            AgentPhase.SHOPPING: 1,
            AgentPhase.IN_QUEUE: 0
        }

    def step(self):
        # print('Phase AgentPhase.{}'.format(self.phase.name))

        if self.phase == AgentPhase.SHOPPING:
            if not self.shopping_time.is_expired():
                self.shopping_time.decrement()
            elif self.decide_queue():
                self.phase = AgentPhase.IN_QUEUE

        elif self.phase == AgentPhase.IN_QUEUE:
            dest_x, dest_y  = self.decide_destination()
            if self.model.grid[dest_x][dest_y] != None:
                return

            self.model.grid.move_agent(self, (dest_x, dest_y))
            x, y = self.pos

            if (x + 1, y) in self.model.cash_registers.values():
                self.phase = AgentPhase.PAYING

        elif self.phase == AgentPhase.PAYING:
            if not self.paying_time.is_expired():
                self.paying_time.decrement()
            else:
                self.model.queues[self.objective].remove(self.unique_id)
                self.objective = None
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)

        self.print_agent_info()
        print(self.model.queues)
        print(self.step_for_phase)
        
        # Update should go at head or tail of atw method ???
        self.step_for_phase[self.phase] += 1

    def decide_queue(self):
        coin = self.random.randint(0, 4)

        x, y, _ = self.model.entry_points[coin]
        if self.model.grid.is_cell_empty((x, y)):
            self.model.grid.place_agent(self, (x, y))
            return True

        return False

    def decide_destination(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True
        )

        destinations = []
        for destination, floor_field in self.model.floor_fields.items():
            if not self.is_cash_register_open(destination):
                continue

            candidates = [floor_field[y, x] for x, y in possible_steps]

            x, y = possible_steps[np.argmin(candidates)]
            destinations.append({
                'destination': destination,
                'move': (x, y),
                'cost': floor_field[y, x]
            })

        if self.pos[1] <= (self.model.height - self.model.lane_switch_boundary):
            for destination in destinations:
                destination['cost'] += len(self.model.queues[destination['destination']])
                if self.objective == destination['destination']:
                    destination['cost'] -= 1

        selected_move = min(destinations, key=lambda x: x['cost'])
        self.update_objective(selected_move['destination'])

        obj_x, obj_y = self.model.cash_registers[self.objective]
        queue_start_y = obj_y - len(self.model.queues[self.objective])
        obj_x -= 1
        x, y = self.pos
        x_direction = -1 if x > obj_x else 1
        y_direction = -1 if y > queue_start_y else 1

        if x != obj_x and y != queue_start_y:
            return (x + x_direction, y + y_direction)

        if x != obj_x:
            return (x + x_direction, y)

        return (x, y + 1)

    def is_cash_register_open(self, destination):
        cash_register_y, cash_register_x = self.model.cash_registers[destination]
        return self.model.grid[cash_register_y][cash_register_x].open

    def update_objective(self, target):
        old_destination = self.objective

        if not self.objective:
            self.model.queues[target].add(self.unique_id)

        if old_destination and target != old_destination:
            self.model.queues[old_destination].remove(self.unique_id)
            self.model.queues[target].add(self.unique_id)

        self.objective = target

    def print_agent_info(self):
        print("Hi, I am agent " + str(self.unique_id) +
              " - PRODS: " + str(self.shopping_time) +
              " - POSITION: " + str(self.pos) +
              " - DESTINATION: " + str(self.objective) + ".")

class SupermarketModel(Model):
    def __init__(self, N, B, world, width, height):
        self.world = world
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, True)
        self.schedule = BaseScheduler(self)
        self.capacity = N
        self.lane_switch_boundary = B
        self.running = True

        self.queues = {}
        self.entry_points = []
        self.queue_entry_points = {}
        self.cash_registers = {}
        self.agents_count = 0
        self.open_cashier = 0

        # Populate grid from world
        for y, row in enumerate(self.world):
            for x, cell in enumerate(row):
                if (cell == 'X'):
                    self.grid[x][y] = ObstacleAgent(str(y)+str(x), self)
                if (cell in ['1', '2', '3', '4', '5']):
                    self.grid[x][y] = CashierAgent(str(y)+str(x), self)
                    self.cash_registers[cell] = (x, y)
                    self.queues[cell] = set()
                    self.queue_entry_points[cell] = (x, y - self.lane_switch_boundary)
                if (cell in ['A', 'B', 'C', 'D', 'E']):
                    self.entry_points.append((x, y, cell))

        worldMatrix = np.matrix(self.world)
        self.distanceMatrix = np.zeros((self.height, self.width))
        self.distanceMatrix[worldMatrix == 'X'] = np.inf
        self.distanceMatrix[worldMatrix == '1'] = np.inf
        self.distanceMatrix[worldMatrix == '2'] = np.inf
        self.distanceMatrix[worldMatrix == '3'] = np.inf
        self.distanceMatrix[worldMatrix == '4'] = np.inf
        self.distanceMatrix[worldMatrix == '5'] = np.inf

        self.floor_fields = {}
        for dest_label, (dest_y, dest_x) in self.queue_entry_points.items():
            self.floor_fields[dest_label] = self.calculate_floor_field((dest_x, dest_y - 1))
            
        self.datacollector = DataCollector(
            model_reporters={"Agent in supermarket": agents_in_supermarket,
                             "Agent that SHOPPING": agents_in_shopping,
                             "Agent IN_QUEUE": agents_in_queue_and_paying,
                             "Avg. number of agent IN_QUEUE": agents_in_queue_avg,
                             "Avg. time spent in queue": agent_in_queue_avg_time}
        )

    def step(self):
        print("STEP - " + str(len(self.schedule.agents)))
        if len(self.schedule.agents) < self.capacity:
            self.schedule.add(self.create_agent())
            
        self.datacollector.collect(self)
        self.schedule.step()
        #print("AGENTS IN_QUEUE_AVG_STEPS: " + str(agent_in_queue_avg_time(self)))

    def create_agent(self):
        agent = CustomerAgent(self.agents_count, self)
        self.agents_count += 1
        return agent

    def calculate_floor_field(self, target):
        field = self.distanceMatrix.copy()

        for x1 in range(len(field)):
            for y1 in range(len(field[x1])):
                if not np.isinf(field[x1, y1]):
                    field[x1, y1] = distance.euclidean([x1, y1], target)

        return field

class AgentPhase(Enum):
    SHOPPING = 0
    IN_QUEUE = 1
    PAYING = 2

class Counter():
    def __init__(self, start):
        self.count = start

    def __repr__(self):
        return str(self.count)

    def is_expired(self):
        return self.count == 0

    def decrement(self):
        self.count -= 1
        return self.count

def agents_in_queue_and_paying(model):
    # Count number of agents IN_QUEUE and PAYING state.   
    agents_in_queue = [agent for agent in model.schedule.agents 
                       if isinstance(agent, CustomerAgent) and 
                       agent.phase in [AgentPhase.IN_QUEUE, AgentPhase.PAYING]]
    return len(agents_in_queue)

def agents_in_supermarket(model):
    # Return number of agents in supermarket.
    agents = [agent for agent in model.schedule.agents]
    return len(agents)

def agents_in_shopping(model):
    # Return number of agents that shopping.
    agents = [agent for agent in model.schedule.agents if isinstance(
        agent, CustomerAgent) and agent.phase in [AgentPhase.SHOPPING]]
    return len(agents)

def agents_in_queue_avg(model):
    # Return number avg num of agent in queue.
    agents = [agent for agent in model.schedule.agents
              if isinstance(agent, CustomerAgent) and
              agent.phase in [AgentPhase.IN_QUEUE, AgentPhase.PAYING]]
    return len(agents) / model.open_cashier

def agent_in_queue_avg_time(model):
    # Count avg number of steps IN_QUEUE and PAYING.
    agents = agents_in_queue_and_paying(model)
    agents_time = [agent.step_for_phase[AgentPhase.IN_QUEUE] + 
                   agent.step_for_phase[AgentPhase.PAYING] 
                   for agent in model.schedule.agents]
    return round(sum(agents_time) / agents, 2) if agents != 0 else 0
    
