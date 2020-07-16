import math

from enum import Enum
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.spatial import distance


class QueueType(Enum):
    CLASSIC = 0
    SNAKE = 1


class AgentPhase(Enum):
    SHOPPING = 0
    REACHING_QUEUE = 1
    IN_QUEUE = 2
    SNAKE_REACHING_CASHIER = 3
    PAYING = 4


class Counter():
    def __init__(self, start):
        self.count = start

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

    def step(self):
        pass


class CashierAgent(Agent):
    def __init__(self, unique_id, model, remaining_life=300, pos=None):
        super().__init__(unique_id, model)
        self.set_life(remaining_life)
        self.pos = pos
        self.is_busy = False

    def __repr__(self):
        return 'Cashier: {} - Position: {}'.format(self.unique_id, self.pos)

    def set_life(self, remaining_life):
        self.remaining_life = remaining_life
        self.open = remaining_life != 0
        if self.open:
            self.model.open_cashier.add(self.unique_id)

    def step(self):
        if self.remaining_life > 0:
            self.remaining_life -= 1


class CustomerAgent(Agent):
    def __init__(self, unique_id, model, sprite):
        super().__init__(unique_id, model)
        self.sprite = sprite

        self.products_count = np.random.normal(self.model.capacity / 2, self.model.capacity / 4)
        self.shopping_time = Counter(3 + self.products_count * 0.75)
        self.paying_time = Counter(1 + self.products_count * 0.25)

        self.phase = AgentPhase.SHOPPING
        self.objective = None
        self.destination = None

        # Dictionary for counting number of step in each phase of Supermarket.
        self.step_for_phase = {
            AgentPhase.SHOPPING: 1,
            AgentPhase.REACHING_QUEUE: 0,
            AgentPhase.IN_QUEUE: 0,
            AgentPhase.SNAKE_REACHING_CASHIER: 0,
            AgentPhase.PAYING: 0,
        }

    def step(self):
        print(self)
        self.step_for_phase[self.phase] += 1

        if self.phase == AgentPhase.SHOPPING:
            if not self.shopping_time.is_expired():
                self.shopping_time.decrement()
            elif self.random_spawn_point():
                self.phase = AgentPhase.REACHING_QUEUE

        elif self.phase == AgentPhase.REACHING_QUEUE:
            if self.model.queue_type == QueueType.CLASSIC:
                # Pick destination cash_register
                dest_x, dest_y = self.decide_destination()
                if self.model.grid[dest_x][dest_y] is not None:
                    return

                # Try to reach queue
                self.model.grid.move_agent(self, (dest_x, dest_y))
                if self.has_reached_destination():
                    self.enqueue()
                    self.phase = AgentPhase.IN_QUEUE

            elif self.model.queue_type == QueueType.SNAKE:
                self.destination = self.model.snake_entry

                self.model.movement_grid.cleanup()
                start = self.model.movement_grid.node(*self.pos)
                end = self.model.movement_grid.node(*self.destination)
                path, _ = self.model.finder.find_path(start, end, self.model.movement_grid)

                dest_x, dest_y = path[1]
                if self.model.grid[dest_x][dest_y] is not None:
                    return

                # Try to reach queue
                self.model.grid.move_agent(self, path[1])
                if self.has_reached_destination():
                    self.phase = AgentPhase.IN_QUEUE

        elif self.phase == AgentPhase.IN_QUEUE:
            if self.model.queue_type == QueueType.CLASSIC:
                x, y = self.pos

                # Use y+1 because we moved agent after reading position
                if (x + 1, y) in self.model.cash_registers.values():
                    self.phase = AgentPhase.PAYING
                    return

                # Move vertically in queue
                if self.model.grid[x][y + 1] is None:
                    self.model.grid.move_agent(self, (x, y + 1))

            if self.model.queue_type == QueueType.SNAKE:
                self.destination = self.model.snake_exit

                self.model.movement_grid.cleanup()
                start = self.model.movement_grid.node(*self.pos)
                end = self.model.movement_grid.node(*self.destination)
                path, _ = self.model.finder.find_path(start, end, self.model.movement_grid)

                dest_x, dest_y = path[1]
                if self.model.grid[dest_x][dest_y] is not None:
                    return

                # Try to reach queue
                self.step_for_phase[self.phase] -= 1
                self.model.grid.move_agent(self, path[1])
                if self.has_reached_destination():
                    self.phase = AgentPhase.SNAKE_REACHING_CASHIER

        elif self.phase == AgentPhase.SNAKE_REACHING_CASHIER:
            if self.pos == self.destination:
                return

            self.step_for_phase[self.phase] -= 1
            self.model.movement_grid.cleanup()
            start = self.model.movement_grid.node(*self.pos)
            end = self.model.movement_grid.node(*self.destination)
            path, _ = self.model.finder.find_path(start, end, self.model.movement_grid)

            dest_x, dest_y = path[1]
            if self.model.grid[dest_x][dest_y] is not None:
                return

            # Try to reach queue
            self.model.grid.move_agent(self, path[1])
            if self.has_reached_destination():
                self.phase = AgentPhase.PAYING

        elif self.phase == AgentPhase.PAYING:
            if not self.paying_time.is_expired():
                self.paying_time.decrement()
            else:
                if self.model.queue_type == QueueType.CLASSIC:
                    self.dequeue()
                elif self.model.queue_type == QueueType.SNAKE:
                    self.model.cashiers[self.objective].is_busy = False

                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)

        # print(self.model.queues)
        # print(self.step_for_phase)

    def enqueue(self):
        self.model.queues[self.objective].add(self.unique_id)

    def dequeue(self):
        self.model.queues[self.objective].remove(self.unique_id)

    def random_spawn_point(self):
        coin = self.random.randint(0, len(self.model.entry_points) - 1)

        x, y, _ = self.model.entry_points[coin]
        if self.model.grid.is_cell_empty((x, y)):
            self.model.grid.place_agent(self, (x, y))
            return True

        return False

    def decide_destination(self):
        neighbour_cells = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True
        )
        selected_move = self.find_best_move(neighbour_cells)

        self.update_objective(selected_move['objective'])
        self.destination = self.find_queue_start_position()

        return self.next_move()

    def find_best_move(self, cells):
        destinations = []
        for objective, floor_field in self.model.floor_fields.items():
            if self.is_cash_register_open(objective):
                candidates = [floor_field[row, col] for col, row in cells]

                x, y = cells[np.argmin(candidates)]
                destinations.append({
                    'objective': objective,
                    'cost': floor_field[y, x]
                })

        if self.pos[1] <= (self.model.height - self.model.lane_switch_boundary):
            for destination in destinations:
                destination['cost'] += len(self.model.queues[destination['objective']])

        return min(destinations, key=lambda x: x['cost'])

    def update_objective(self, objective):
        old_objective = self.objective

        if old_objective is not None and objective != old_objective:
            print("changing destination", old_objective, objective)
            if self.unique_id in self.model.queues[old_objective]:
                self.dequeue()

        self.objective = objective

    def next_move(self):
        dest_col, dest_row = self.destination

        col, row = self.pos
        h_direction = -1 if col > dest_col else 1
        v_direction = -1 if row > dest_row else 1

        if col != dest_col:
            return (col + h_direction, row)

        return (col, row + v_direction)

    def is_cash_register_open(self, destination):
        col, row = self.model.cash_registers[destination]
        return self.model.grid[col][row].open

    def has_reached_destination(self):
        return self.pos == self.destination

    def find_queue_start_position(self):
        col, row = self.model.cash_registers[self.objective]
        col -= 1
        while (not self.model.grid.is_cell_empty((col, row))) and self.model.grid[col][row] != self:
            row -= 1

        return (col, row)

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
    def __init__(self, capacity, boundary, world, width, height, terrain_map_name, type=QueueType.CLASSIC):
        # Mesa internals
        self.running = True

        # World related
        self.world = world
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, True)
        self.queue_type = QueueType[type]
        self.terrain_map_name = 'map3' if self.queue_type == QueueType.CLASSIC else 'map3-snake'
        self.lane_switch_boundary = boundary

        # Agent related
        self.agents_count = 0
        self.schedule = BaseScheduler(self)
        self.capacity = capacity

        self.entry_points = []
        self.queues = {}
        self.queue_entry_points = {}
        self.queue_length_limit = 7
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
                    agent = CashierAgent(cell, self, 0, (row, col))
                    self.cashiers[cell] = self.grid[row][col] = agent
                    self.cash_registers[cell] = (row, col)
                    self.queues[cell] = set()
                    self.queue_entry_points[cell] = (row, col - self.lane_switch_boundary)
                    # TODO: Add (remove) only upon cashier opening (closing)
                    self.schedule.add(agent)
                elif cell in ['A', 'B', 'C', 'D', 'E']:
                    self.entry_points.append((row, col, cell))

        world_matrix = np.matrix(self.world)
        self.distance_matrix = np.zeros((self.height, self.width))
        self.distance_matrix[world_matrix == 'X'] = np.inf
        self.distance_matrix[world_matrix == '1'] = np.inf
        self.distance_matrix[world_matrix == '2'] = np.inf
        self.distance_matrix[world_matrix == '3'] = np.inf
        self.distance_matrix[world_matrix == '4'] = np.inf
        self.distance_matrix[world_matrix == '5'] = np.inf

        self.floor_fields = {}
        for dest_label, (dest_y, dest_x) in self.cash_registers.items():
            self.floor_fields[dest_label] = self.calculate_floor_field((dest_x, dest_y - 1))

        self.datacollector = DataCollector(
            model_reporters={"Agent in supermarket": agents_in_supermarket,
                             "Agent that shopping": agents_in_shopping,
                             "Agent in queue": agents_in_queue,
                             "Avg. number of agent in queue": agents_in_queue_avg,
                             "Avg. time spent in queue": agent_in_queue_avg_time,
                             "Agent in payment": agents_in_paying})

        if self.queue_type == QueueType.SNAKE:
            self.distance_matrix[world_matrix == 'X'] = 1
            self.distance_matrix[world_matrix == '1'] = 1
            self.distance_matrix[world_matrix == '2'] = 1
            self.distance_matrix[world_matrix == '3'] = 1
            self.distance_matrix[world_matrix == '4'] = 1
            self.distance_matrix[world_matrix == '5'] = 1
            self.movement_grid = Grid(matrix=self.distance_matrix, inverse=True)

        coin = self.random.randint(1, len(self.cashiers) - 1)
        self.cashiers[str(coin)].set_life(300)

    def step(self):
        print('Customers: {} - {}'.format(self.agents_count, len(self.schedule.agents)))
        if len(self.schedule.agents) - 4 < self.capacity and self.should_spawn_agent():
            self.schedule.add(self.create_agent())

        available, busy = self.partition(self.cashiers.values(), lambda c: c.open and not c.is_busy)
        if self.queue_type == QueueType.SNAKE:
            if (not self.grid.is_cell_empty(self.snake_exit)) and len(available) > 0:
                customer = self.grid.get_cell_list_contents(self.snake_exit)[0]

                coin = self.random.randint(0, len(available) - 1)
                cashier = available[coin]

                customer.objective = cashier.unique_id
                dest_x, dest_y = cashier.pos
                customer.destination = (dest_x - 1, dest_y)
                cashier.is_busy = True
                print('Assigning cash {} to customer {}'.format(coin, customer))
        # print("AGENTS IN_QUEUE_AVG_STEPS: " + str(agent_in_queue_avg_time(self)))

        self.datacollector.collect(self)
        self.schedule.step()

        serving, closed = self.partition(self.cashiers.values(), lambda c: c.open)
        if len(serving) > 0 and len(closed) > 0:
            if agents_in_queue_avg(self) >= self.queue_length_limit:
                coin = self.random.randint(0, len(closed) - 1)
                cashier = closed[coin]
                cashier.set_life(300)
                self.open_cashier.add(cashier.unique_id)
                print('Opening new cash register: {}'.format(cashier.unique_id))

        if self.queue_type == QueueType.CLASSIC:
            if len(serving) > 1:
                for cashier in serving:
                    in_queue = len(self.queues[cashier.unique_id])
                    if in_queue < 3 and in_queue > 1 and (cashier.remaining_life == 0 or self.agents_count < (self.capacity / 3)):
                        cashier.open = False
                        self.open_cashier.remove(cashier.unique_id)
                        print('Closing cash register: {}'.format(cashier.unique_id))
        elif self.queue_type == QueueType.SNAKE:
            avg_agents_in_queue = agents_in_queue_avg(self)
            if len(serving) > 1 and avg_agents_in_queue < 3:
                to_close = [c for c in serving if c.remaining_life == 0 or self.agents_count < (self.capacity / 3)]
                if len(to_close) > 0:
                    coin = self.random.randint(0, len(to_close) - 1)
                    self.close_cashier(to_close[coin])

    def close_cashier(self, cashier):
        cashier.open = False
        self.open_cashier.remove(cashier.unique_id)
        print('Closing cash register: {}'.format(cashier.unique_id))

    def partition(self, elements, predicate):
        left, right = [], []
        for e in elements:
            (left if predicate(e) else right).append(e)

        return left, right

    def create_agent(self):
        agent = CustomerAgent(self.agents_count, self, self.random_sprite())
        self.agents_count += 1
        return agent

    def should_spawn_agent(self):
        # Current: -\frac{\cos\left(\frac{t\pi}{1200}\right)}{2}+\frac{1}{2}
        # Attempt: \frac{1}{16\cos^{2}\left(\pi x\right)+1}
        relative_time = self.schedule.steps % 1440
        prob = (-math.cos(relative_time * np.pi / 720) + 1) / 2
        return self.random.random() <= prob

    def random_sprite(self):
        sprites = [
            'images/characters/grandpa',
            'images/characters/grandpa2',
            'images/characters/grandpa3',
            'images/characters/man',
            'images/characters/man2',
            'images/characters/man3',
            'images/characters/man4',
            'images/characters/man5',
            'images/characters/man6',
            'images/characters/man7',
            'images/characters/man8',
            'images/characters/man9',
            'images/characters/girl',
            'images/characters/girl2',
            'images/characters/girl3',
            'images/characters/girl4',
            'images/characters/girl5',
            'images/characters/girl6',
            'images/characters/girl7',
            'images/characters/girl8',
            'images/characters/girl9',
            'images/characters/child-male',
            'images/characters/child-female',
        ]

        return sprites[self.random.randint(0, len(sprites) - 1)]

    def calculate_floor_field(self, target):
        field = self.distance_matrix.copy()

        for x1 in range(len(field)):
            for y1 in range(len(field[x1])):
                if not np.isinf(field[x1, y1]):
                    field[x1, y1] = distance.euclidean([x1, y1], target)

        return field


def get_agents_in_phase(model, phase):
    return [agent for agent in model.schedule.agents
            if isinstance(agent, CustomerAgent) and agent.phase in phase]


def agents_in_queue(model):
    """ Count number of agents IN_QUEUE state. """
    agents_in_queue = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return len(agents_in_queue)


def agents_in_queue_avg(model):
    """ Return number avg num of agent in queue. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    return round(len(agents) / len(model.open_cashier), 2)


def agents_in_supermarket(model):
    """ Return number of agents in supermarket. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE,
                                         AgentPhase.PAYING,
                                         AgentPhase.SHOPPING,
                                         AgentPhase.REACHING_QUEUE,
                                         AgentPhase.SNAKE_REACHING_CASHIER])
    return len(agents)


def agents_in_shopping(model):
    """ Return number of agents that SHOPPING. """
    agents = get_agents_in_phase(model, [AgentPhase.SHOPPING])
    return len(agents)


def agents_in_paying(model):
    """ Count number of agents in PAYING state. """
    agents_in_queue = get_agents_in_phase(model, [AgentPhase.PAYING])
    return len(agents_in_queue)


def agent_in_queue_avg_time(model):
    """ Count avg number of steps IN_QUEUE. """
    agents = get_agents_in_phase(model, [AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER])
    agents_time = [agent.step_for_phase[AgentPhase.IN_QUEUE] + agent.step_for_phase[AgentPhase.SNAKE_REACHING_CASHIER]
                   for agent in agents]
    return round(sum(agents_time) / len(agents), 2) if len(agents) != 0 else 0
