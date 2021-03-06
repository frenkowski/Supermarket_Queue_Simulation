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
from pathfinding.finder.ida_star import IDAStarFinder
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
        self.empty_since = 0

    def __repr__(self):
        return 'Cashier: {} - Position: {}'.format(self.unique_id, self.pos)

    def set_life(self, life=180):
        self.remaining_life = life
        self.open = life > 0
        if self.open:
            self.model.open_cashier.add(self.unique_id)

    @property
    def serving(self):
        col, row = self.pos
        return self.model.grid[col - 1][row] is not None or self.open

    def step(self):
        if not self.serving:
            self.empty_since += 1
        else:
            self.empty_since = 0

        if self.remaining_life > 0:
            self.remaining_life -= 1
        elif self.remaining_life < 0:
            self.remaining_life += 1


class CustomerAgent(Agent):
    def __init__(self, unique_id, model, sprite):
        super().__init__(unique_id, model)
        self.sprite = sprite

        self.products_count = math.floor(get_truncated_normal(mean=self.model.capacity / 2, sd=self.model.capacity / 4, upp=self.model.capacity))
        self.shopping_time = Counter((3 + self.products_count * 0.75) * 60)
        self.paying_time = Counter((1 + self.products_count * 0.25) * 60)

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
    def __init__(self, type=QueueType.CLASSIC, seed=None):
        np.random.seed(seed)

        # Mesa internals
        self.running = True
        self.steps_in_day = 7200

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
        self.queue_length_limit = 5
        self.cashiers = {}
        # TODO: Merge position (cash_registers) and open
        # attribute (open_cashier) with cashiers dict
        self.cash_registers = {}
        self.open_cashier = set()

        # Pathfinding
        self.finder = IDAStarFinder()
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
            floor_field = self.calculate_floor_field((dest_row, dest_col - 1))

            self.floor_fields[dest_label] = floor_field.copy()

            # Save floor field heatmap into file
            # floor_field[floor_field == np.inf] = -np.inf
            # plt.figure(figsize=(14, 14))
            # sns.heatmap(floor_field, vmin=0, fmt='.1f', vmax=np.max(floor_field), annot=True, cbar=False, square=True, cmap='mako', xticklabels=False, yticklabels=False)
            # plt.tight_layout()
            # plt.savefig(os.path.join('..', 'output', 'ff-heatmap{}.png'.format(dest_label)))
            # plt.close()

        self.datacollector = DataCollector(
            model_reporters={"Total": get_total_agents,
                             "Shopping": get_shopping_agents,
                             "Queued": get_queued_agents,
                             "Queued (AVG)": get_avg_queued_agents,
                             "Queued Time (AVG)": get_avg_queued_steps,
                             "Total Time (AVG)": get_avg_total_steps,
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

        coin = self.random.randint(1, len(self.cashiers))
        self.cashiers[str(coin)].set_life()

        coin = self.random.randint(1, len(self.cashiers))
        while self.cashiers[str(coin)].open:
            coin = self.random.randint(1, len(self.cashiers))

        self.cashiers[str(coin)].set_life()

    def step(self):
        self.current_agents = len(self.schedule.agents) - len(self.cashiers.items())

        if self.schedule.steps > self.steps_in_day and get_total_agents(self) == 0 and (self.schedule.steps - 3) % 250:
            self.store_heatmap()
            self.running = False
            return

        if self.schedule.steps < self.steps_in_day and self.steps_in_day and self.schedule.steps % 25 == 0 and self.current_agents < self.capacity and self.should_spawn_agent():
            self.schedule.add(self.create_agent())

        self.datacollector.collect(self)
        self.schedule.step()

        self.adjust_cashiers()
        if self.queue_type == QueueType.SNAKE:
            self.assign_cash_register_to_customer()

    def adjust_cashiers(self):
        # self.current_agents > (len(opened) + 1) * self.capacity / 7
        # (len(opened) + 1) * self.queue_length_limit / self.current_agents

        opened, closed = self.partition(self.cashiers.values(), lambda c: c.open)
        if len(closed) > 0:
            if len(opened) < self.ideal_number_of_cashier(self.schedule.steps) and self.current_agents > (len(opened) + 1) * self.queue_length_limit:
                coin = self.random.randint(0, len(closed) - 1)
                cashier = closed[coin]
                cashier.set_life()
                self.open_cashier.add(cashier.unique_id)
                print(Back.WHITE + Fore.GREEN + 'OPENING NEW CASH_REGISTER: {}'.format(cashier.unique_id))

        if len(opened) > 2:
            np.random.shuffle(opened)
            if self.queue_type == QueueType.CLASSIC:
                for cashier in opened:
                    in_queue = len(self.queues[cashier.unique_id])
                    if (in_queue > 1 or in_queue == 0) and len(opened) > self.ideal_number_of_cashier(self.schedule.steps) and self.current_agents < (len(opened) + 1) * self.queue_length_limit:
                        self.close_cashier(cashier)
                        opened.remove(cashier)
                        break

            elif self.queue_type == QueueType.SNAKE:
                if len(opened) > self.ideal_number_of_cashier(self.schedule.steps) and self.current_agents < (len(opened) + 1) * self.queue_length_limit:
                    to_close = opened
                    if len(to_close) > 0:
                        coin = self.random.randint(0, len(to_close) - 1)
                        cashier = to_close[coin]

                        self.close_cashier(cashier)
                        opened.remove(cashier)
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
        self.heatmap /= np.max(self.heatmap)
        sns.heatmap(self.heatmap, vmin=0, vmax=1)
        plt.savefig(os.path.join('..', 'output', 'heatmap{}.png'.format('' if self.queue_type == QueueType.CLASSIC else '-snake')))
        plt.close()

    def close_cashier(self, cashier):
        cashier.open = False
        cashier.empty_since = 0
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
        relative_time = self.schedule.steps % self.steps_in_day
        prob = (-math.cos(relative_time * np.pi / (self.steps_in_day / 2) + 1) + 1) / 2
        return self.random.random() <= 0.85 if self.random.random() <= prob else False

    def ideal_number_of_cashier(self, step):
        prob = (step % self.steps_in_day) / self.steps_in_day

        # if prob <= 0.125 or prob >= 0.875:
        #     return 2
        # if prob <= 0.25 or prob >= 0.75:
        #     return 3
        # if prob <= 0.375 or prob >= 0.625:
        #     return 4
        # if prob <= 0.75 or prob >= 0.25:
        #     return 5

        if prob <= 0.265 or prob >= 0.88:
            return 2
        if prob <= 0.36 or prob >= 0.765:
            return 3
        if prob <= 0.44 or prob >= 0.66:
            return 4

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
    """ Count number of agents IN_QUEUE state.

    Used only on piechart
    """
    agents_in_queue = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return len(agents_in_queue)


def get_working_cashiers(model):
    return [agent for agent in model.schedule.agents if isinstance(agent, CashRegisterAgent) and agent.serving]


def get_avg_queued_agents(model):
    """ Return number avg num of Queued. """
    working_cashiers = get_working_cashiers(model)
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return math.ceil(len(agents) / len(working_cashiers))


def get_total_agents(model):
    """ Return number of agents in supermarket. """
    return model.current_agents


def get_shopping_agents(model):
    """ Return number of agents in SHOPPING state. """
    return len(get_agents_in_phase(model, [AgentPhase.SHOPPING]))


def get_paying_agents(model):
    """ Count number of agents in PAYING state. """
    return len(get_agents_in_phase(model, [AgentPhase.PAYING]))


# def get_avg_queued_steps(model):
#     """ Count avg number of steps IN_QUEUE. """
#     queue_removed_time = model.removed_agents_steps[AgentPhase.IN_QUEUE] + model.removed_agents_steps[AgentPhase.REACHING_QUEUE]
#     return math.ceil(queue_removed_time / model.removed_number / 60) if model.removed_number != 0 else 0


# def get_avg_total_steps(model):
#     """ Count avg number of steps IN_QUEUE. """
#     return (sum(model.removed_agents_steps.values()) / 60) / model.removed_number if model.removed_number != 0 else 0


def get_avg_queued_steps(model):
    """ Count avg number of steps IN_QUEUE. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.REACHING_QUEUE])
    agents_steps = [agent.step_for_phase[AgentPhase.IN_QUEUE] + agent.step_for_phase[AgentPhase.REACHING_QUEUE]
                    for agent in agents]
    return sum(agents_steps) / len(agents) / 60 if len(agents) != 0 else 0


def get_avg_total_steps(model):
    """ Count avg number of steps IN_QUEUE. """
    agents = get_agents_in_phase(model, [AgentPhase.SHOPPING, AgentPhase.REACHING_QUEUE, AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER, AgentPhase.PAYING])
    agents_steps = [agent.step_for_phase[AgentPhase.IN_QUEUE]
                    + agent.step_for_phase[AgentPhase.SNAKE_REACHING_CASHIER]
                    + agent.step_for_phase[AgentPhase.SHOPPING]
                    + agent.step_for_phase[AgentPhase.REACHING_QUEUE]
                    + agent.step_for_phase[AgentPhase.PAYING]
                    for agent in agents]
    return sum(agents_steps) / len(agents) / 60 if len(agents) != 0 else 0


def get_truncated_normal(mean=0, sd=1, low=1, upp=np.inf):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()
