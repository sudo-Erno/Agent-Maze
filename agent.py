import numpy as np
import random

from numpy.testing._private.utils import rand

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

        # Initialize the reward
        self.reward = 0
    
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

    def update_q_values(self, next_states):
        states_values = []

        for i in range(len(next_states)):
            
            state = next_states[i]

            print(f"{self.environment[state[0], state[1]] = }")

            if self.is_inside_maze(state):
                states_values.append(self.environment[state[0], state[1]])
            else:
                states_values.append(self.reward_for_leaving_limits)
        
        print("\n")

        prob_times_value = 0
        for i in range(len(states_values)):
            prob_times_value += 0.25 * states_values[i]

        self.QValues[self.actual_coords_y, self.actual_coords_x] = self.environment[self.actual_coords_y, self.actual_coords_x] + self.gamma * prob_times_value

    def move_throught_environment(self, steps=5):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        for i in range(steps):
            game_state = 0
            
            # Check the values on the QValue table for next action
            posibles_next_states = [
                [self.actual_coords_y - 1, self.actual_coords_x], # UP
                [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
                [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
                [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
            ]

            rnd = random.randint(0, 3)
            next_state = posibles_next_states[rnd]

            self.update_q_values(posibles_next_states)

            while not self.is_inside_maze(next_state):
                rnd = random.randint(0, 3)
                next_state = posibles_next_states[rnd]
            
            # Update position
            self.actual_coords_x = next_state[1]
            self.actual_coords_y = next_state[0]

            # Check if it has reached the final
            if self.arrived_final((self.actual_coords_y, self.actual_coords_x)):
                game_state = 1
                # Restart from random position
                self.actual_coords_y = self.initial_y
                self.actual_coords_x = self.initial_x

        return game_state, self.QValues

        # for i in range(steps):
        #     game_state = 0
            
        #     while game_state == 0:
        #         # Check the values on the QValue table for next action
        #         posibles_next_states = [
        #             [self.actual_coords_y - 1, self.actual_coords_x], # UP
        #             [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
        #             [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
        #             [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
        #         ]

        #         rnd = random.randint(0, 3)
        #         next_state = posibles_next_states[rnd]

        #         self.update_q_values(posibles_next_states)

        #         while not self.is_inside_maze(next_state):
        #             rnd = random.randint(0, 3)
        #             next_state = posibles_next_states[rnd]
                
        #         # Update position
        #         self.actual_coords_x = next_state[1]
        #         self.actual_coords_y = next_state[0]

        #         # Check if it has reached the final
        #         if self.arrived_final((self.actual_coords_y, self.actual_coords_x)):
        #             game_state = 1
        #             # Restart from random position
        #             self.actual_coords_y = self.initial_y
        #             self.actual_coords_x = self.initial_x

        # return game_state, self.QValues