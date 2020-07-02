from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid

import numpy as np


class ObstacleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass


class CashierAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.open = self.random.random() > 0.5

    def step(self):
        pass


class CustomerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.permanence_time = self.random.randint(0, 4)

    def step(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print("Hi, I am agent " + str(self.unique_id) +
              " - TIME: " + str(self.permanence_time) + ".")


class SupermarketModel(Model):
    def __init__(self, N, world, width, height):
        self.world = world
        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.capacity = N
        self.running = True

        self.entry_point = list()
        self.agents_count = 0

        # Populate grid from world
        for i, row in enumerate(self.world):
            for j, cell in enumerate(row):
                if (cell == 'X'):
                    self.grid[j][i] = ObstacleAgent(str(i)+str(j), self)
                if (cell in ['1', '2', '3', '4', '5']):
                    self.grid[j][i] = CashierAgent(str(i)+str(j), self)
                if (cell in ['A', 'B', 'C', 'D', 'E']):
                    self.entry_point.append((j, i))

        # # Create agents
        # for i in range(self.num_agents):
        #     a = CustomerAgent(i, self)
        #     self.schedule.add(a)
        #     # Add the agent to a random grid cell
        #     x = self.random.randrange(self.grid.width)
        #     y = self.random.randrange(self.grid.height)
        #     while not self.grid.is_cell_empty((x, y)):
        #         x = self.random.randrange(self.grid.width)
        #         y = self.random.randrange(self.grid.height)
        #     self.grid.place_agent(a, (x, y))

        worldMatrix = np.matrix(self.world)
        worldMatrix[worldMatrix == 'A'] = np.inf
        worldMatrix[worldMatrix == '1'] = 0
        print(worldMatrix)

    def step(self):
        print("STEP - " + str(len(self.schedule.agents)))
        if (len(self.schedule.agents) < self.capacity):
            coin = self.random.randint(0, 4)
            if(self.grid.is_cell_empty(self.entry_point[coin])):
                a = self.createAgent()
                self.schedule.add(a)
                self.grid.place_agent(a, self.entry_point[coin])

        self.schedule.step()

    def createAgent(self):
        a = CustomerAgent(self.agents_count, self)
        self.agents_count += 1
        return a
