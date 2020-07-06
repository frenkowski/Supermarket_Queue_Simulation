from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import SingleGrid

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
        self.open = self.random.random() > 0.2

    def step(self):
        pass


class CustomerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.n_prod = self.random.randint(1, 51)

        self.shopping_time = Counter(self.n_prod)
        self.paying_time = Counter(self.n_prod)
        self.phase = AgentPhase.SHOPPING

    def step(self):
        print("Hi, I am agent " + str(self.unique_id) +
              " - PRODS: " + str(self.shopping_time) +
              " - POSITION: " + str(self.pos) + ".")

        if self.phase == AgentPhase.SHOPPING:
            if not self.shopping_time.is_expired():
                self.shopping_time.decrement()
            elif self.decide_queue():
                self.phase = AgentPhase.IN_QUEUE

        elif self.phase == AgentPhase.IN_QUEUE:
            self.model.grid.move_agent(self, self.decide_destination())
            x, y = self.pos
            if (x + 1, y) in self.model.cash_registers.values():
                self.phase = AgentPhase.PAYING

        elif self.phase == AgentPhase.PAYING:
            if not self.paying_time.is_expired():
                self.paying_time.decrement()
            else:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)


    def decide_queue(self):
        coin = self.random.randint(0, 4)

        x, y, _ = self.model.entry_points[coin]
        if (self.model.grid.is_cell_empty((x, y))):
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

        selected_move = min(destinations, key=lambda x: x['cost'])
        return selected_move['move']

    def is_cash_register_open(self, destination):
        cash_register_y, cash_register_x = self.model.cash_registers[destination]
        return self.model.grid[cash_register_y][cash_register_x].open

class SupermarketModel(Model):
    def __init__(self, N, world, width, height):
        self.world = world
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, True)
        self.schedule = BaseScheduler(self)
        self.capacity = N
        self.running = True

        self.cash_registers = {}
        self.entry_points = list()
        self.agents_count = 0

        # Populate grid from world
        for i, row in enumerate(self.world):
            for j, cell in enumerate(row):
                if (cell == 'X'):
                    self.grid[j][i] = ObstacleAgent(str(i)+str(j), self)
                if (cell in ['1', '2', '3', '4', '5']):
                    self.grid[j][i] = CashierAgent(str(i)+str(j), self)
                    self.cash_registers[cell] = (j, i)
                if (cell in ['A', 'B', 'C', 'D', 'E']):
                    self.entry_points.append((j, i, cell))

        worldMatrix = np.matrix(self.world)
        self.distanceMatrix = np.zeros((self.height, self.width))
        self.distanceMatrix[worldMatrix == 'X'] = np.inf
        self.distanceMatrix[worldMatrix == '1'] = np.inf
        self.distanceMatrix[worldMatrix == '2'] = np.inf
        self.distanceMatrix[worldMatrix == '3'] = np.inf
        self.distanceMatrix[worldMatrix == '4'] = np.inf
        self.distanceMatrix[worldMatrix == '5'] = np.inf

        # self.distanceMatrix[worldMatrix == 'A'] = 1
        # self.distanceMatrix[worldMatrix == 'B'] = 1
        # self.distanceMatrix[worldMatrix == 'C'] = 1
        # self.distanceMatrix[worldMatrix == 'D'] = 1
        # self.distanceMatrix[worldMatrix == 'E'] = 1

        # print(self.distanceMatrix)

        print(self.cash_registers)

        self.floor_fields = {}
        for dest_label, (dest_y, dest_x) in self.cash_registers.items():
            self.floor_fields[dest_label] = self.calculate_floor_field((dest_x, dest_y - 1))

    def step(self):
        print("STEP - " + str(len(self.schedule.agents)))
        if (len(self.schedule.agents) < self.capacity):
            self.schedule.add(self.create_agent())

        self.schedule.step()

    def create_agent(self):
        a = CustomerAgent(self.agents_count, self)
        self.agents_count += 1
        return a

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
