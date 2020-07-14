import math

from enum import Enum
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import SingleGrid
import numpy as np
from scipy.spatial import distance


class QueueType(Enum):
    CLASSIC = 0
    SNAKE = 1


class AgentPhase(Enum):
    SHOPPING = 0
    REACHING_QUEUE = 1
    IN_QUEUE = 2
    SNAKE_END = 3
    PAYING = 4


class Counter():
    def __init__(self, start):
        self.count = start

    def __repr__(self):
        return str(self.count)

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
    def __init__(self, unique_id, model, remaining_life=300):
        super().__init__(unique_id, model)
        self.extend_life(remaining_life)

    def extend_life(self, remaining_life):
        self.remaining_life = remaining_life
        self.open = remaining_life is not 0

    def step(self):
        if self.remaining_life > 0:
            self.remaining_life -= 1


class CustomerAgent(Agent):
    def __init__(self, unique_id, model, sprite):
        super().__init__(unique_id, model)
        self.sprite = sprite
        self.n_prod = np.random.normal(self.model.capacity / 2, self.model.capacity / 4)

        self.shopping_time = Counter(3 + self.n_prod * 0.75)
        self.paying_time = Counter(1 + self.n_prod * 0.25)
        self.phase = AgentPhase.SHOPPING
        self.destination = None
        self.objective = None

    def step(self):
        # self.print_agent_info()

        if self.phase == AgentPhase.SHOPPING:
            if not self.shopping_time.is_expired():
                self.shopping_time.decrement()
            elif self.decide_spawn_point():
                self.phase = AgentPhase.REACHING_QUEUE

        elif self.phase == AgentPhase.REACHING_QUEUE:
            if self.model.queue_type == QueueType.CLASSIC:
                # Pick destination cash_register
                dest_x, dest_y  = self.decide_destination()
                if self.model.grid[dest_x][dest_y] != None:
                    return

                # Try to reach queue
                self.model.grid.move_agent(self, (dest_x, dest_y))
                if self.has_reached_destination():
                    self.model.queues[self.objective].add(self.unique_id)
                    self.phase = AgentPhase.IN_QUEUE

            elif self.model.queue_type == QueueType.SNAKE:
                self.destination = self.model.snake_entry
                self.move_towards_cell()

                # Try to reach queue
                self.model.grid.move_agent(self, (dest_x, dest_y))
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
                if self.model.grid[x][y + 1] != None:
                    return

                self.model.grid.move_agent(self, (x, y + 1))
            if self.model.queue_type == QueueType.SNAKE:
                print('Snake entry reached, now what!?')

        elif self.phase == AgentPhase.PAYING:
            if not self.paying_time.is_expired():
                self.paying_time.decrement()
            else:
                self.model.queues[self.objective].remove(self.unique_id)
                self.objective = None
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)

        # print(self.model.queues)

    def decide_spawn_point(self):
        coin = self.random.randint(0, len(self.model.entry_points) - 1)

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

        selected_move = self.find_best_move(possible_steps)
        self.update_objective(selected_move['destination'])
        self.destination = self.find_queue_start_position()

        return self.move_towards_destination()

    def move_towards_destination(self):
        dest_col, dest_row = self.destination

        x, y = self.pos
        x_direction = -1 if x > dest_col else 1
        y_direction = -1 if y > dest_row else 1

        if x != dest_col:
            return (x + x_direction, y)

        return (x, y + y_direction)

    def find_best_move(self, possible_steps):
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

        return min(destinations, key=lambda x: x['cost'])

    def is_cash_register_open(self, destination):
        cash_register_y, cash_register_x = self.model.cash_registers[destination]
        return self.model.grid[cash_register_y][cash_register_x].open

    def has_reached_destination(self):
        return self.pos == self.destination

    def has_reached_cash_register_queue(self, pos):
        queue_col, queue_row = self.find_queue_start_position()
        return pos == (queue_col, queue_row)

    def update_objective(self, target):
        old_destination = self.objective

        # if not self.objective:
        #     self.model.queues[target].add(self.unique_id)

        if old_destination and target != old_destination:
            print ("changing destination", old_destination, target)
            if self.unique_id in self.model.queues[old_destination]:
                self.model.queues[old_destination].remove(self.unique_id)

        self.objective = target

    def find_queue_start_position(self):
        col, row = self.model.cash_registers[self.objective]
        col -= 1
        while (not self.model.grid.is_cell_empty((col,row))) and self.model.grid[col][row] != self:
            row -= 1

        return (col,row)

    def print_agent_info(self):
        print('AGENT: {}/{} - PRODS: {} - SHOP_TIME: {} - CURRENT: {} - DEST: {}'.format(
            self.unique_id,
            self.phase.name,
            self.n_prod,
            self.shopping_time,
            self.pos,
            self.objective
        ))

class SupermarketModel(Model):
    def __init__(self, N, B, world, width, height, queue_type=QueueType.CLASSIC):
        self.world = world
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, True)
        self.schedule = BaseScheduler(self)
        self.capacity = N
        self.lane_switch_boundary = B
        self.running = True

        self.queue_type = queue_type
        self.snake_entry = None
        self.snake_exit = None

        self.queues = {}
        self.entry_points = []
        self.queue_entry_points = {}
        self.cashiers = {}
        self.cash_registers = {}
        self.agents_count = 0

        # Populate grid from world
        for y, row in enumerate(self.world):
            for x, cell in enumerate(row):
                if cell == 'X':
                    self.grid[x][y] = ObstacleAgent(str(y)+str(x), self)
                elif cell == 'S':
                    self.snake_entry = (x,y)
                elif cell == 'Z':
                    self.snake_exit = (x,y)
                elif cell in ['1', '2', '3', '4', '5']:
                    agent = CashierAgent(cell, self, 0)
                    self.cashiers[cell] = self.grid[x][y] = agent
                    self.schedule.add(agent)
                    self.cash_registers[cell] = (x, y)
                    self.queues[cell] = set()
                    self.queue_entry_points[cell] = (x, y - self.lane_switch_boundary)
                elif cell in ['A', 'B', 'C', 'D', 'E']:
                    self.entry_points.append((x, y, cell))

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

        coin = self.random.randint(1, len(self.cashiers) - 1)
        self.cashiers[str(coin)].extend_life(300)

    def step(self):
        if len(self.schedule.agents) < self.capacity and self.should_spawn_agent():
            self.schedule.add(self.create_agent())

        self.schedule.step()

        serving, closed = self.partition(self.cashiers.values(), lambda c: c.open)
        if len(serving) > 0 and len(closed) > 0:
            ciq = 0
            for cashier in serving:
                ciq += len(self.queues[cashier.unique_id])

            cpq = ciq / len(serving)
            if cpq >= 8:
                coin = self.random.randint(0, len(closed) - 1)
                closed[coin].extend_life(300)
                print('Opening new cash register: {}'.format(cashier.unique_id))

        if len(serving) > 1:
            for cashier in serving:
                in_queue = len(self.queues[cashier.unique_id])
                if in_queue < 3 and in_queue > 1 and (cashier.remaining_life == 0 or self.agents_count < (self.capacity / 3)):
                    cashier.open = False
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
        # -\frac{\cos\left(\frac{t\pi}{1200}\right)}{2}+\frac{1}{2}
        relative_time = self.schedule.steps % 1440
        prob = (-math.cos(relative_time * np.pi / 720) + 1) / 2
        return self.random.random() < prob

        # if self.random.random() < prob:
            # return self.random.random() > 0.85

        # return self.random.random() > 0.95

    def random_sprite(self):
        sprites = [
            'images/characters/scout',
            'images/characters/old-man',
        ]

        return sprites[self.random.randint(0, len(sprites) - 1)]

    def calculate_floor_field(self, target):
        field = self.distance_matrix.copy()

        for x1 in range(len(field)):
            for y1 in range(len(field[x1])):
                if not np.isinf(field[x1, y1]):
                    field[x1, y1] = distance.euclidean([x1, y1], target)

        return field
