import numpy as np
from random import randint

class Agent:

    def __init__(self, x = 0, y = 0, lr=1e-3, gamma = 1e-4, reward_for_leaving_limits = -0.75):
        # Up: 0, Right: 1, Down: 2, Left: 3
        self.actions = ["up", "right", "down", "left"]

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

        self.QValues = np.zeros_like(env)
        self.QValues[1][0] = 1.0

        # Save the coordinates of the exit
        self.final = final

    def instant_reward(self, new_coordinates):
        return 1 / ((new_coordinates[0] - self.final[0])**2 + (new_coordinates[1] - self.final[1])**2)**0.5
    
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
        
    def move_throught_environment(self):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """

        # Check the values on the QValue table for next action
        next_states = [
            self.QValues[self.actual_coords_x][self.actual_coords_y - 1], # UP
            self.QValues[self.actual_coords_x + 1][self.actual_coords_y], # RIGHT
            self.QValues[self.actual_coords_x][self.actual_coords_y + 1], # DOWN
            self.QValues[self.actual_coords_x - 1][self.actual_coords_y] # LEFT
        ]
        
        # Get the max value and index of it
        max_value = max(next_states)
        max_index = next_states.index(max_value)

        # Executes movements based on the values of the states
        future_state = [self.actual_coords_x, self.actual_coords_y]

        if max_index == 0:
            future_state[1] -= 1
        
        elif max_index == 1:
            future_state[0] += 1
        
        elif max_index == 2:
            future_state[1] += 1
        
        elif max_index == 3:
            future_state[0] -= 1

        # Get the new reward value
        self.reward += self.instant_reward((self.actual_coords_x, self.actual_coords_y))

        # Check if its outside the limits
        if future_state[0] < 0 or future_state[0] > self.final[0]:
            self.reward += self.reward_for_leaving_limits
            return -1
        
        elif future_state[1] < 0 or future_state[1] > self.final[1]:
            self.reward += self.reward_for_leaving_limits
            return -1
        
        # Check if it has arrive at the exit
        if future_state[0] == self.final[0] and future_state[1] == self.final[1]:
            self.reward += self.reward_reaching_final
            return 1

        self.reward += self.environment[future_state[0], future_state[1]]

        # Update Q-Value
        self.QValues[self.actual_coords_x, self.actual_coords_y] = self.reward + self.gamma * self.QValues[future_state[0], future_state[1]]

        # Update the coordinates
        self.actual_coords_x = future_state[0]
        self.actual_coords_y = future_state[1]
        
        return 0