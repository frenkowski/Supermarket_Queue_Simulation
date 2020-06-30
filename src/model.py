from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid


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

    def step(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print ("Hi, I am agent " + str(self.unique_id) +".")


class SupermarketModel(Model):
    def __init__(self, N, world, width, height):
        self.num_agents = N
        self.world = world
        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Populate grid from world
        for i, row in enumerate(self.world):
            for j, cell in enumerate(row):
                if (cell == 'X'):
                    self.grid[j][i] = ObstacleAgent(str(i)+str(j), self)
                if (cell == 'C'):
                    self.grid[j][i] = CashierAgent(str(i)+str(j), self)

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
        self.schedule.step()
