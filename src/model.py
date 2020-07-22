import math
import os

from colorama import Fore, Back
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.spatial import distance
from scipy.stats import truncnorm
import seaborn as sns

from enums import AgentPhase, QueueType
from strategies import ClassicStepStrategy, SnakeStepStrategy


class Counter():
    def __init__(self, start):
        self.count = math.floor(start)

    def __repr__(self):
        return str(int(self.count))

    def is_expired(self):
        return self.count <= 0

    def decrement(self):
        self.count -= 1
        return self.count


class ObstacleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class CashierAgent(Agent):
    def __init__(self, unique_id, model, pos, cash_register):
        super().__init__(unique_id, model)
        self.pos = pos
        self.cash_register = cash_register

    @property
    def working(self):
        col, row = self.pos
        return self.cash_register.open or self.model.grid[col - 2][row] is not None or self.cash_register.is_busy


class CashRegisterAgent(Agent):
    def __init__(self, unique_id, model, pos=None):
        super().__init__(unique_id, model)
        self.set_life(0)
        self.pos = pos
        self.is_busy = False

    def __repr__(self):
        return 'Cashier: {} - Position: {}'.format(self.unique_id, self.pos)

    def set_life(self, life=180):
        self.remaining_life = life
        self.open = life > 0
        if self.open:
            self.model.open_cashier.add(self.unique_id)

    def step(self):
        if self.remaining_life > 0:
            self.remaining_life -= 1
        elif self.remaining_life < 0:
            self.remaining_life += 1


class CustomerAgent(Agent):
    def __init__(self, unique_id, model, sprite):
        super().__init__(unique_id, model)
        self.sprite = sprite

        self.products_count = math.floor(get_truncated_normal(mean=self.model.capacity / 2, sd=self.model.capacity / 4, upp=self.model.capacity))
        self.shopping_time = Counter(3 + self.products_count * 0.75)
        self.paying_time = Counter(1 + self.products_count * 0.25)

        self.phase = AgentPhase.SHOPPING
        self.remaining_objective_updates = 2
        self.objective = None
        self.destination = None
        self.previous_positions = None
        self.is_stuck = False

        if self.model.queue_type == QueueType.CLASSIC:
            self.strategy = ClassicStepStrategy(self, self.model)
        elif self.model.queue_type == QueueType.SNAKE:
            self.strategy = SnakeStepStrategy(self, self.model)

        # Dictionary for counting number of step in each phase of Supermarket.
        self.step_for_phase = {
            AgentPhase.SHOPPING: 1,
            AgentPhase.REACHING_QUEUE: 0,
            AgentPhase.IN_QUEUE: 0,
            AgentPhase.SNAKE_REACHING_CASHIER: 0,
            AgentPhase.PAYING: 0,
        }

    def step(self):
        print(Fore.CYAN + str(self))
        self.step_for_phase[self.phase] += 1

        self.strategy.step()

        if self.pos is not None:
            col, row = self.pos
            self.model.heatmap[row, col] += 1

        # print(self.model.queues)
        # print(self.step_for_phase)

    def __repr__(self):
        return 'AGENT: {}/{} - PROD COUNT: {} - SHOP_TIME: {} - POS: {} - OBJ: {} - DEST: {}'.format(
            self.unique_id,
            self.phase.name,
            self.products_count,
            self.shopping_time,
            self.pos,
            self.objective,
            self.destination
        )


class SupermarketModel(Model):
    def __init__(self, type=QueueType.CLASSIC):
        # Mesa internals
        self.running = True
        self.steps_in_day = 1440

        # World related
        self.queue_type = QueueType[type]
        self.terrain_map_name = 'map' if self.queue_type == QueueType.CLASSIC else 'map_snake'
        with open(os.path.join(os.getcwd(), '..', 'resources', '{}.txt'.format(self.terrain_map_name))) as f:
            self.width, self.height = map(int, f.readline().strip().split(' '))
            self.capacity = int(f.readline().strip())
            self.world = [list(c) for c in f.read().split('\n') if c]

        self.grid = SingleGrid(self.width, self.height, True)

        # Agent related
        self.generated_customers_count = 0
        self.schedule = BaseScheduler(self)

        self.entry_points = []
        self.queues = {}
        self.queue_length_limit = 6
        self.cashiers = {}
        # TODO: Merge position (cash_registers) and open
        # attribute (open_cashier) with cashiers dict
        self.cash_registers = {}
        self.open_cashier = set()

        # Pathfinding
        self.finder = AStarFinder()
        self.snake_entry = None
        self.snake_exit = None

        # Populate grid from world
        for col, line in enumerate(self.world):
            for row, cell in enumerate(line):
                if cell == 'X':
                    self.grid[row][col] = ObstacleAgent('{}:{}'.format(col, row), self)
                elif cell == 'S':
                    self.snake_entry = (row, col)
                elif cell == 'Z':
                    self.snake_exit = (row, col)
                elif cell in ['1', '2', '3', '4', '5']:
                    cash_register = CashRegisterAgent(cell, self, (row, col))
                    self.cashier_row = col
                    self.cashiers[cell] = self.grid[row][col] = cash_register
                    self.cash_registers[cell] = (row, col)
                    self.queues[cell] = set()
                    # TODO: Add (remove) only upon cashier opening (closing)
                    self.schedule.add(cash_register)

                    cashier = CashierAgent('Y{}'.format(cell), self, (row + 1, col), cash_register)
                    self.grid[row + 1][col] = cashier
                elif cell in ['A', 'B', 'C', 'D', 'E']:
                    self.entry_points.append((row, col, cell))
                    self.spawn_row = col

        self.lane_switch_boundary = math.ceil((self.cashier_row - self.spawn_row) * 3 / 4)

        self.heatmap = np.zeros((self.height, self.width))

        world_matrix = np.matrix(self.world)
        self.distance_matrix = np.zeros((self.height, self.width))
        self.distance_matrix[world_matrix == 'X'] = np.inf
        self.distance_matrix[world_matrix == '1'] = np.inf
        self.distance_matrix[world_matrix == '2'] = np.inf
        self.distance_matrix[world_matrix == '3'] = np.inf
        self.distance_matrix[world_matrix == '4'] = np.inf
        self.distance_matrix[world_matrix == '5'] = np.inf
        self.distance_matrix[world_matrix == 'Y'] = np.inf

        self.floor_fields = {}
        for dest_label, (dest_col, dest_row) in self.cash_registers.items():
            self.floor_fields[dest_label] = self.calculate_floor_field((dest_row, dest_col - 1))

        self.datacollector = DataCollector(
            model_reporters={"Total": get_total_agents,
                             "Shopping": get_shopping_agents,
                             "Queued": get_queued_agents,
                             "Queued (AVG)": get_avg_queued_agents,
                             "Queued Time (AVG)": get_avg_queued_steps,
                             "Paying": get_paying_agents})

        if self.queue_type == QueueType.SNAKE:
            self.distance_matrix[world_matrix == 'X'] = 1
            self.distance_matrix[world_matrix == '1'] = 1
            self.distance_matrix[world_matrix == '2'] = 1
            self.distance_matrix[world_matrix == '3'] = 1
            self.distance_matrix[world_matrix == '4'] = 1
            self.distance_matrix[world_matrix == '5'] = 1
            self.distance_matrix[world_matrix == 'Y'] = 1
            self.movement_grid = Grid(matrix=self.distance_matrix, inverse=True)

        coin = self.random.randint(1, len(self.cashiers) - 1)
        self.cashiers[str(coin)].set_life()

    def step(self):
        self.current_agents = len(self.schedule.agents) - len(self.cashiers.items())
        if self.current_agents < self.capacity and self.should_spawn_agent():
            self.schedule.add(self.create_agent())

        self.datacollector.collect(self)
        self.schedule.step()

        self.adjust_cashiers()
        if self.queue_type == QueueType.SNAKE:
            self.assign_cash_register_to_customer()

        if self.schedule.steps % (self.steps_in_day / 2) == 0:
            self.store_heatmap()

    def adjust_cashiers(self):
        opened, closed = self.partition(self.cashiers.values(), lambda c: c.open)
        if len(closed) > 0:
            # if len(get_agents_in_phase(self, [AgentPhase.REACHING_QUEUE, AgentPhase.IN_QUEUE])) / len(opened) >= self.queue_length_limit:
            if len(opened) < self.ideal_number_of_cashier(self.schedule.steps) and self.current_agents > (len(opened) + 1) * self.capacity / self.queue_length_limit:
                coin = self.random.randint(0, len(closed) - 1)
                cashier = closed[coin]
                # if cashier.remaining_life == 0:
                cashier.set_life()
                self.open_cashier.add(cashier.unique_id)
                print(Back.WHITE + Fore.GREEN + 'OPENING NEW CASH_REGISTER: {}'.format(cashier.unique_id))

        if len(opened) > 1:
            if self.queue_type == QueueType.CLASSIC:
                for cashier in opened:
                    in_queue = len(self.queues[cashier.unique_id])
                    # if in_queue > 1 and in_queue <= math.floor(self.queue_length_limit / 2) and (cashier.remaining_life == 0 and self.current_agents < (self.capacity / 3)):
                    if in_queue > 1 and len(opened) > self.ideal_number_of_cashier(self.schedule.steps) and self.current_agents < (len(opened) + 1) * self.capacity / self.queue_length_limit:
                        self.close_cashier(cashier)
                        opened.remove(cashier)
                        # cashier.set_life(-90)
                        # [cashier.set_life(cashier.remaining_life + 25) for cashier in opened]
                        break

            elif self.queue_type == QueueType.SNAKE:
                if get_avg_queued_agents(self) < math.floor(self.queue_length_limit / 2):
                    to_close = [c for c in opened if c.remaining_life == 0 and self.current_agents < (self.capacity / 3)]
                    if len(to_close) > 0:
                        coin = self.random.randint(0, len(to_close) - 1)
                        cashier = to_close[coin]

                        self.close_cashier(cashier)
                        opened.remove(cashier)
                        # cashier.set_life(-90)
                        [cashier.set_life(cashier.remaining_life + 25) for cashier in opened]

    def assign_cash_register_to_customer(self):
        available, busy = self.partition(self.cashiers.values(), lambda c: c.open and not c.is_busy)
        if (not self.grid.is_cell_empty(self.snake_exit)) and len(available) > 0:
            customer = self.grid.get_cell_list_contents(self.snake_exit)[0]

            coin = self.random.randint(0, len(available) - 1)
            cashier = available[coin]

            customer.objective = cashier.unique_id
            dest_col, dest_row = cashier.pos
            customer.destination = (dest_col - 1, dest_row)
            cashier.is_busy = True
            print(Back.WHITE + Fore.BLACK + 'ASSIGNING CASH_REGISTER {} TO CUSTOMER {}'.format(coin, customer.unique_id))

    def store_heatmap(self):
        sns.heatmap(self.heatmap)
        plt.savefig(os.path.join('..', 'output', 'heatmap{}.png'.format('' if self.queue_type == QueueType.CLASSIC else '-snake')))
        plt.close()

    def close_cashier(self, cashier):
        cashier.open = False
        self.open_cashier.remove(cashier.unique_id)
        print(Back.WHITE + Fore.RED + 'CLOSING CASH_REGISTER: {}'.format(cashier.unique_id))

    def partition(self, elements, predicate):
        left, right = [], []
        for e in elements:
            (left if predicate(e) else right).append(e)

        return left, right

    def create_agent(self):
        agent = CustomerAgent(self.generated_customers_count, self, self.random_sprite())
        self.generated_customers_count += 1
        return agent

    def should_spawn_agent(self):
        # Current: -\frac{\cos\left(\frac{t\pi}{1200}\right)}{2}+\frac{1}{2}
        # Attempt: \frac{1}{16\cos^{2}\left(\pi x\right)+1}
        relative_time = self.schedule.steps % self.steps_in_day
        prob = (-math.cos(relative_time * np.pi / (self.steps_in_day / 2)) + 1) / 2
        return self.random.random() <= prob

    def ideal_number_of_cashier(self, step):
        prob = (step % self.steps_in_day) / self.steps_in_day

        if prob <= 0.18 or prob >= 0.90:
            return 1
        if prob <= 0.25 or prob >= 0.83:
            return 2
        if prob <= 0.30 or prob >= 0.72:
            return 3
        if prob <= 0.41 or prob >= 0.62:
            return 4

        # if prob <= 0.45 or prob >= 0.55:
        return 5

    def random_sprite(self):
        sprites = [
            'images/characters/grandpa3',
            'images/characters/man5',
            'images/characters/man8',
            'images/characters/girl',
            'images/characters/girl3',
            'images/characters/girl9',
        ]

        return sprites[self.random.randint(0, len(sprites) - 1)]

    def calculate_floor_field(self, destination):
        field = self.distance_matrix.copy()

        for row in range(len(field)):
            for col in range(len(field[row])):
                if not np.isinf(field[row, col]):
                    field[row, col] = distance.euclidean([row, col], destination)

        return field


def get_agents_in_phase(model, phases):
    """ Retrieve agents in the specified phase(phases) """
    return [agent for agent in model.schedule.agents
            if isinstance(agent, CustomerAgent) and agent.phase in phases]


def get_queued_agents(model):
    """ Count number of agents IN_QUEUE state. """
    agents_in_queue = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return len(agents_in_queue)


def get_avg_queued_agents(model):
    """ Return number avg num of Queued. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return math.ceil(len(agents) / len(model.open_cashier))


def get_total_agents(model):
    """ Return number of agents in supermarket. """
    return model.current_agents


def get_shopping_agents(model):
    """ Return number of agents in SHOPPING state. """
    return len(get_agents_in_phase(model, [AgentPhase.SHOPPING]))


def get_paying_agents(model):
    """ Count number of agents in PAYING state. """
    return len(get_agents_in_phase(model, [AgentPhase.PAYING]))


def get_avg_queued_steps(model):
    """ Count avg number of steps IN_QUEUE. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    agents_steps = [agent.step_for_phase[AgentPhase.IN_QUEUE] + agent.step_for_phase[AgentPhase.SNAKE_REACHING_CASHIER]
                    for agent in agents]
    return math.ceil(sum(agents_steps) / len(agents)) if len(agents) != 0 else 0


def get_truncated_normal(mean=0, sd=1, low=1, upp=np.inf):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()
