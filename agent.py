import numpy as np
import random

class Agent:

    def __init__(self, x = 0, y = 0, lr = 1e-3, gamma = 1e-4, epsilon = 0.9, reward_for_leaving_limits = -0.75):
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

        self.epsilon = epsilon

        # Initialize the learning rate for the Q-function
        self.learning_rate = lr

        self.reward_for_leaving_limits = reward_for_leaving_limits

    
    def set_position(self, y, x):
        self.actual_coords_y = y
        self.actual_coords_x = x

    def set_environmet(self, env, location_blocks, final):
        # Save the Maze
        self.environment = env
        self.location_blocks = location_blocks
        
        # Save the coordinates of the exit
        self.final = final

        self.QValues = np.zeros_like(self.environment)

    def instant_reward(self, new_coordinates):
        d = ((new_coordinates[0] - self.final[0])**2 + (new_coordinates[1] - self.final[1])**2)**0.5
        if d != 0:
            return 1 / d, d
        else:
            return 9, d
    
    def is_inside_maze(self, state):
        if state[0] < 0 or state[0] > len(self.environment[0]) - 1:
            return False

        if state[1] < 0 or state[1] > len(self.environment[1]) - 1:
            return False

        return True
    
    def arrived_final(self, state):
        if state[0] == self.final[0] and state[1] == self.final[1]:
            return True

        return False

    def choose_action(self, next_states, min_distance_index, exploration_prob, b_prob=0.7):
        """
        Returns list with the probabilities of the executing the actions
        """

        if exploration_prob < self.epsilon:
            # Choose the one with highest state value
            next_states_max_value = 0
            next_states_max_index = 1

            for i in range(len(next_states)):
                state_value = self.QValues[next_states[i][0], next_states[i][1]]

                if self.is_inside_maze(next_states[i]) and state_value >= next_states_max_value:
                    next_states_max_value = state_value
                    next_states_max_index = i
            
            return next_states_max_index
            
        else:
            return random.randint(0, len(self.actions.keys()) - 1)
        
    def update_state_values(self, actual_state, next_state):

        inside_maze = self.is_inside_maze(next_state)
        
        if inside_maze:
            self.QValues[actual_state[0], actual_state[1]] = (1 - self.learning_rate) * self.QValues[actual_state[0], actual_state[1]] + self.gamma * self.QValues[next_state[0], next_state[1]]
        else:
            self.QValues[actual_state[0], actual_state[1]] = (1 - self.learning_rate) * self.QValues[actual_state[0], actual_state[1]] + self.gamma * self.reward_for_leaving_limits

    def move_throught_environment(self):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        game_state = 0
        
        # Check the values on the QValue table for next action
        posible_next_states = [
            [self.actual_coords_y - 1, self.actual_coords_x], # UP
            [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
            [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
            [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
        ]

        # Calculate distances
        distances = []
        for state in posible_next_states:
            _, dis = self.instant_reward(state)
            distances.append(dis)

        # action = self.choose_action(posible_next_states, distances.index(min(distances)), 0.1)
        action = self.choose_action(posible_next_states, distances.index(min(distances)), random.random())

        next_state = posible_next_states[action]

        while not self.is_inside_maze(next_state):
            next_state = posible_next_states[action]

        self.update_state_values((self.actual_coords_y, self.actual_coords_x), next_state)

        self.actual_coords_y = next_state[0]
        self.actual_coords_x = next_state[1]

        print(f"{self.actual_coords_y = } {self.actual_coords_x = }")

        print(self.QValues)
        print("\n")

        return game_state