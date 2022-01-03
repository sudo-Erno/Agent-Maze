import numpy as np
from random import randint

class Agent:

    def __init__(self, x = 0, y = 0, lr=1e-3, gamma = 1e-4, reward_for_leaving_limits = -0.75):
        # Up: 0, Right: 1, Down: 2, Left: 3
        self.actions = {0: "up", 1: "right", 2: "down", 3: "left"}

        # Set initial coordinates
        self.initial_x = x
        self.initial_y = y

        # Set actual coordinates of the agent
        self.actual_coords_x = x
        self.actual_coords_y = y

        # Initialize the gamma for the Q-function
        self.gamma = gamma

        # Initialize the learning rate for the Q-function
        self.learning_rate = lr

        self.reward_for_leaving_limits = reward_for_leaving_limits

        # Initialize the reward
        self.reward = 0
    
    def set_environmet(self, env, location_blocks, final):
        # Save the Maze
        self.environment = env
        self.location_blocks = location_blocks
        
        # Save the coordinates of the exit
        self.final = final

        self.QValues = np.zeros_like(env)

    def instant_reward(self, new_coordinates):
        d = ((new_coordinates[0] - self.final[0])**2 + (new_coordinates[1] - self.final[1])**2)**0.5
        if d != 0:
            return 1 / d, d
        else:
            return 9, d
    
    def available_movements(self):
        movements = ["up", "right", "down", "left"]

        if self.actual_coords_x - 1 < 0:
            movements.remove("left")
        elif self.actual_coords_x + 1 > self.final[0]:
            movements.remove("right")
        
        if self.actual_coords_y - 1 < 0:
            movements.remove("up")
        elif self.actual_coords_y + 1 > self.final[1]:
            movements.remove("down")

        return movements
    
    def plot_qtable(self):
        q_table = np.array(self.QValues, dtype=np.str)
        q_table[self.final] = "F"
        q_table[self.initial_y][self.initial_x] = "A"
        print(q_table)

    def move_throught_environment(self):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        get_to_final = False
        
        # Check the values on the QValue table for next action
        next_states = [
            [self.actual_coords_x, self.actual_coords_y - 1], # UP
            [self.actual_coords_x + 1, self.actual_coords_y], # RIGHT
            [self.actual_coords_x, self.actual_coords_y + 1], # DOWN
            [self.actual_coords_x - 1, self.actual_coords_y] # LEFT
        ]

        # Calculate distances
        distances = []
        for state in next_states:
            _, dis = self.instant_reward(state)
            distances.append(dis)
        
        min_distance_index = distances.index(min(distances))

        states_possibilities = []

        for i in range(len(next_states)):
            state = list(self.actions.values())
            state = state[i]

            if i == min_distance_index:
                for j in range(70):
                    states_possibilities.append(state)
            else:
                for j in range(10):
                    states_possibilities.append(state)

        print(states_possibilities)

        # print("\t\t")
        # print(self.QValues)
        # print("\t\t")
        
        # self.plot_qtable()

        return get_to_final