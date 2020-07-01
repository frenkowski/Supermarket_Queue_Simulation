from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid

import random


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
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.shop_time = random.randint(1, 6)

    def step(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print("Hi, I am agent " + str(self.unique_id) + " - Time: " + str(self.shop_time) + ".")


class SupermarketModel(Model):
    def __init__(self, N, world, width, height):
        self.world = world
        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)

        self.entry_point = (0, 0)
        self.capacity = N
        self.agents_count = 0
        
        self.running = True
        
        # Populate grid from world
        for i, row in enumerate(self.world):
            for j, cell in enumerate(row):
                if (cell == 'X'):
                    self.grid[j][i] = ObstacleAgent(str(i)+str(j), self)
                if (cell == 'C'):
                    self.grid[j][i] = CashierAgent(str(i)+str(j), self)
                if (cell == 'E'):
                    self.entry_point = (j, i)

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

    def step(self):
        print("STEP - " + str(len(self.schedule.agents)))
        if (len(self.schedule.agents) < self.capacity):
            a = self.createAgent()
            if(self.grid.is_cell_empty(self.entry_point)):
                self.schedule.add(a)
                self.grid.place_agent(a, self.entry_point)

        self.schedule.step()

    def createAgent(self):
        a = CustomerAgent(self.agents_count, self)
        self.agents_count += 1
        return a
