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

        # Initialize the reward
        self.reward = 0

        # Save all the states where the agent was
        self.past_states = []
        self.is_first_action = True
    
    def set_position(self, y, x):
        self.actual_coords_y = y
        self.actual_coords_x = x

    def set_environmet(self, env, location_blocks, final):
        # Save the Maze
        self.environment = env
        self.location_blocks = location_blocks
        
        # Save the coordinates of the exit
        self.final = final

        self.QValues = np.zeros((len(self.actions), self.environment.shape[0], self.environment.shape[1]))
        self.Q_action_state = self.QValues

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

    def action_probability(self, next_states, min_distance_index, b_prob=70, s_prob=10, epsilon=0.7):
        """
        Returns a list with the probability of each action
        """
        states_possibilities = []

        if epsilon < self.epsilon:
            # Give the majority of the probability of moving to the state where the distance to objective is shorter

            for i in range(len(next_states)):
                
                state = list(self.actions.keys())
                state = state[i]

                if i == min_distance_index:
                    states_possibilities.append(b_prob / 100)
                
                else:
                    states_possibilities.append(s_prob / 100)
        else:
            for i in range(len(next_states)):
                states_possibilities.append(1 / len(next_states))
        
        return states_possibilities
        
    # TODO: La parte de probabilidad deberia ir a la Q-Function, no en la V-Function... --> Nose, despues pienso
    def update_state_values(self, next_states, probabilities):

        for i in range(len(probabilities)):
            next_state = next_states[i]
            
            # Check if the next state is outside the maze...
            inside_maze = self.is_inside_maze(next_state)
            
            if inside_maze:
                self.QValues[i, self.actual_coords_y, self.actual_coords_x] = probabilities[i] * (self.environment[next_state[0], next_state[1]] + self.gamma * self.QValues[i, next_state[0], next_state[1]])
            else:
                self.QValues[i, self.actual_coords_y, self.actual_coords_x] = probabilities[i] * self.reward_for_leaving_limits
        
        self.past_states.append([self.actual_coords_y, self.actual_coords_x]) # Only save the x and y coordinates because I already take into account all the actions

        if not self.is_first_action:
            
            for i in range(len(self.past_states[:-1])):
                
                row = self.past_states[i][0]
                col = self.past_states[i][1]

                next_row = self.past_states[i+1][0]
                next_col = self.past_states[i+1][1]

                for i in range(len(list(self.actions.keys()))):
                
                    if self.is_inside_maze([row, col]):
                        self.QValues[i, row, col] = probabilities[i] * (self.environment[next_row, next_col] + self.gamma * self.QValues[i, next_row, next_col])
                    else:
                        self.QValues[i, row, col] = probabilities[i] * self.reward_for_leaving_limits

    def update_state_actions_values(self, next_states, probabilities):
        
        max_probability_action_index = 0

        if len(set(probabilities)) != 1:
            max_probability_action_index = probabilities.index(max(probabilities))
        else:
            max_probability_action_index = random.randint(0, 3)
        
        # Moves to the state which it's values is the max of all options
        self.actual_coords_y = next_states[max_probability_action_index][0]
        self.actual_coords_x = next_states[max_probability_action_index][1]

    def move_throught_environment(self):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        get_to_final = False
        
        # Check the values on the QValue table for next action
        next_states = [
            [self.actual_coords_y - 1, self.actual_coords_x], # UP
            [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
            [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
            [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
        ]

        # Calculate distances
        distances = []
        for state in next_states:
            _, dis = self.instant_reward(state)
            distances.append(dis)

        actions_probabilities = self.action_probability(next_states, distances.index(min(distances)), epsilon = random.random()) # Returns list with the probability of moving to different states.

        self.update_state_values(next_states, actions_probabilities)

        self.is_first_action = False

        self.update_state_actions_values(next_states, actions_probabilities)

        # Check if it has arrived to the end
        if self.actual_coords_y == self.final[0] and self.actual_coords_x == self.final[1]:
            get_to_final = True
        
        print("\tQ-VALUES\t")
        print(self.QValues)
        print("\t\t")

        return get_to_final