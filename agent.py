import numpy as np
from random import randint

class Agent:

    def __init__(self, x = 0, y = 0, lr=1e-3, gamma = 1e-4, reward_for_leaving_limits = -0.75):
        # Up: 0, Right: 1, Down: 2, Left: 3
        self.actions = ["up", "right", "down", "left"]

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
            return 1 / d
        else:
            return 9
    
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
        next_states = []
        next_states_values = []
        
        movements = self.available_movements()

        # Save the coordinates and values of the posible next states
        if "up" in movements:
            next_states_values.append(self.QValues[self.actual_coords_x, self.actual_coords_y - 1])
            next_states.append([self.actual_coords_x, self.actual_coords_y - 1])
        
        if "right" in movements:
            next_states_values.append(self.QValues[self.actual_coords_x + 1, self.actual_coords_y])
            next_states.append([self.actual_coords_x + 1, self.actual_coords_y])
        
        if "down" in movements:
            next_states_values.append(self.QValues[self.actual_coords_x, self.actual_coords_y + 1])
            next_states.append([self.actual_coords_x, self.actual_coords_y + 1])
        
        if "left" in movements:
            next_states_values.append(self.QValues[self.actual_coords_x - 1, self.actual_coords_y])
            next_states.append([self.actual_coords_x - 1, self.actual_coords_y])

        # Getting the state with the highest value
        max_value_states = max(next_states_values)

        # Getting the index of the state with the highest value
        max_value_index = max(next_states)

        # Get the reward for the next state
        self.reward += self.instant_reward((max_value_index[0], max_value_index[1]))

        """
        # Check if its outside the limits
        if future_state[0] < 0 or future_state[0] > self.final[0]:
            self.reward += self.reward_for_leaving_limits
            return -1
        
        elif future_state[1] < 0 or future_state[1] > self.final[1]:
            self.reward += self.reward_for_leaving_limits
            return -1
        """

        # Check if it has arrive at the exit
        if max_value_index[0] == self.final[0] and max_value_index[1] == self.final[1]:
            self.reward += 1.0
            get_to_final = True

        # Bellman equation
        self.QValues[self.actual_coords_y][self.actual_coords_x] = self.reward + self.gamma * max_value_states

        # Update position of the agent
        self.actual_coords_y, self.actual_coords_x = max_value_index

        print("\t\t")
        print(self.QValues)
        print("\t\t")
        
        # self.plot_qtable()

        return get_to_final