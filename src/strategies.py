from abc import ABC, abstractmethod

from colorama import Fore, Back
import numpy as np

from enums import AgentPhase


class StepStrategy(ABC):
    def __init__(self, agent, model):
        self.agent = agent
        self.model = model

    @abstractmethod
    def step(self) -> None:
        pass

    def enqueue(self):
        self.model.queues[self.agent.objective].add(self.agent.unique_id)

    def dequeue(self):
        self.model.queues[self.agent.objective].remove(self.agent.unique_id)

    def random_spawn_point(self):
        # coin = self.random.randint(0, len(self.model.entry_points) - 1)
        coin = np.random.choice(len(self.model.entry_points), 1, p=[0.35, 0.25, 0.2, 0.1, 0.1])[0]

        x, y, _ = self.model.entry_points[coin]
        if self.model.grid.is_cell_empty((x, y)):
            self.model.grid.place_agent(self.agent, (x, y))
            return True

        return False

    def decide_destination(self):
        neighbour_cells = self.model.grid.get_neighborhood(
            self.agent.pos,
            moore=True,
            include_center=True
        )
        selected_move = self.find_best_move(neighbour_cells)

        self.update_objective(selected_move['objective'])
        self.agent.destination = self.find_queue_start_position()

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

        if self.agent.pos[1] <= (self.model.height - self.model.lane_switch_boundary):
            for destination in destinations:
                destination['cost'] += len(self.model.queues[destination['objective']])

        return min(destinations, key=lambda x: x['cost'])

    def update_objective(self, objective):
        old_objective = self.agent.objective

        # if self.agent.is_stuck:
        #     self.agent.remaining_objective_updates = 2
        #     self.agent.is_stuck = False

        if old_objective is None:
            self.agent.objective = objective
        elif (objective != old_objective and self.model.random.random() <= 0.25 and self.agent.remaining_objective_updates > 0):
            print(Back.WHITE + Fore.MAGENTA + 'CHANGING DESTINATION FROM {} TO {}'.format(old_objective, objective))
            self.agent.remaining_objective_updates -= 1
            self.agent.objective = objective
            if self.agent.unique_id in self.model.queues[old_objective]:
                self.dequeue()

    def next_move(self):
        dest_col, dest_row = self.agent.destination

        col, row = self.agent.pos
        h_direction = -1 if col > dest_col else 1
        v_direction = -1 if row > dest_row else 1

        if col != dest_col:
            return (col + h_direction, row)

        return (col, row + v_direction)

    def is_cash_register_open(self, destination):
        col, row = self.model.cash_registers[destination]
        return self.model.grid[col][row].open

    def has_reached_destination(self):
        return self.agent.pos == self.agent.destination

    def find_queue_start_position(self):
        col, row = self.model.cash_registers[self.agent.objective]
        col -= 1
        while (not self.model.grid.is_cell_empty((col, row))) and self.model.grid[col][row] != self.agent:
            row -= 1

        return (col, row)


class ClassicStepStrategy(StepStrategy):
    def step(self):
        if self.agent.phase == AgentPhase.SHOPPING:
            if not self.agent.shopping_time.is_expired():
                self.agent.shopping_time.decrement()
            elif self.random_spawn_point():
                self.agent.phase = AgentPhase.REACHING_QUEUE

        elif self.agent.phase == AgentPhase.REACHING_QUEUE:
            if self.agent.previous_positions == self.agent.pos:
                self.agent.is_stuck = True

            # Pick destination cash_register
            dest_x, dest_y = self.decide_destination()

            if self.agent.is_stuck:
                print(Back.WHITE + Fore.CYAN + 'AGENT IS STUCK AT {}'.format(self.agent.pos))
                dest_y += 1
                self.agent.is_stuck = False

            self.agent.previous_positions = self.agent.pos

            if self.model.grid[dest_x][dest_y] is not None:
                return

            # Try to reach queue
            self.model.grid.move_agent(self.agent, (dest_x, dest_y))
            if self.has_reached_destination():
                self.enqueue()
                self.agent.phase = AgentPhase.IN_QUEUE

        elif self.agent.phase == AgentPhase.IN_QUEUE:
            x, y = self.agent.pos

            # Use y+1 because we moved agent after reading position
            if (x + 1, y) in self.model.cash_registers.values():
                self.agent.phase = AgentPhase.PAYING
                return

            # Move vertically in queue
            if self.model.grid[x][y + 1] is None:
                self.model.grid.move_agent(self.agent, (x, y + 1))

        elif self.agent.phase == AgentPhase.PAYING:
            if not self.agent.paying_time.is_expired():
                self.agent.paying_time.decrement()
            else:
                self.dequeue()
                self.model.grid.remove_agent(self.agent)
                self.model.schedule.remove(self.agent)


class SnakeStepStrategy(StepStrategy):
    def step(self):
        if self.agent.phase == AgentPhase.SHOPPING:
            if not self.agent.shopping_time.is_expired():
                self.agent.shopping_time.decrement()
            elif self.random_spawn_point():
                self.agent.phase = AgentPhase.REACHING_QUEUE

        elif self.agent.phase in [AgentPhase.REACHING_QUEUE, AgentPhase.IN_QUEUE, AgentPhase.SNAKE_REACHING_CASHIER]:
            if self.agent.phase == AgentPhase.REACHING_QUEUE:
                self.agent.destination = self.model.snake_entry
            elif self.agent.phase == AgentPhase.IN_QUEUE:
                self.agent.destination = self.model.snake_exit
            elif self.agent.phase == AgentPhase.SNAKE_REACHING_CASHIER and self.agent.pos == self.agent.destination:
                return

            # if self.agent.phase == AgentPhase.SNAKE_REACHING_CASHIER:
                # self.agent.step_for_phase[self.agent.phase] -= 1

            self.model.movement_grid.cleanup()
            start = self.model.movement_grid.node(*self.agent.pos)
            end = self.model.movement_grid.node(*self.agent.destination)
            path, _ = self.model.finder.find_path(start, end, self.model.movement_grid)

            dest_x, dest_y = path[1]
            if self.model.grid[dest_x][dest_y] is not None:
                return

            if self.agent.phase == AgentPhase.IN_QUEUE:
                self.agent.step_for_phase[self.agent.phase] -= 1

            # Try to reach queue
            self.model.grid.move_agent(self.agent, path[1])
            if self.has_reached_destination():
                if self.agent.phase == AgentPhase.REACHING_QUEUE:
                    self.agent.phase = AgentPhase.IN_QUEUE
                elif self.agent.phase == AgentPhase.IN_QUEUE:
                    self.agent.phase = AgentPhase.SNAKE_REACHING_CASHIER
                elif self.agent.phase == AgentPhase.SNAKE_REACHING_CASHIER:
                    self.agent.phase = AgentPhase.PAYING

        elif self.agent.phase == AgentPhase.PAYING:
            if not self.agent.paying_time.is_expired():
                self.agent.paying_time.decrement()
            else:
                self.model.cashiers[self.agent.objective].is_busy = False
                self.model.grid.remove_agent(self.agent)
                self.model.schedule.remove(self.agent)
