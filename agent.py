import numpy as np
import random

w = 1

class Agent:

    def __init__(self, x = 0, y = 0, lr = 1e-3, gamma = 0.8, epsilon = 0.9, reward_for_leaving_limits = -0.75):
        # Up: 0, Right: 1, Down: 2, Left: 3
        self.actions = {0: "up", 1: "right", 2: "down", 3: "left"}

        # Set actual coordinates of the agent
        self.actual_coords_x = x
        self.actual_coords_y = y

        # Save coordinates for updating the state_values
        self.initial_coords = y, x
        
        # List to store all the states the agent has been on
        self.states_history = []

        # Discount factor
        self.gamma = gamma

        # Probability of doing something random
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

        self.state_values = np.zeros((self.environment.shape[0], self.environment.shape[1]))

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

    def plot_qtable(self):
        q_table = np.array(self.state_values, dtype=np.str)
        q_table[self.final] = "F"
        q_table[self.initial_y][self.initial_x] = "A"
        print(q_table)

    def choose_action(self, next_states, min_distance_index, exploration_prob, b_prob=0.7):
        """
        Returns list with the probabilities of the executing the actions
        """
        states_possibilities = []

        s_prob = 1 - b_prob
        s_prob /= (len(next_states) - 1)

        if exploration_prob < self.epsilon:
            # Choose the one closest to the objective
            
            for i in range(len(next_states)):
                if i == min_distance_index:
                    states_possibilities.append(b_prob)
                else:
                    states_possibilities.append(s_prob)
        else:
            rnd_prob = 1 / len(next_states)
            states_possibilities = [rnd_prob] * len(next_states)
        
        next_action_index = 0

        if len(set(states_possibilities)) == 1:
            next_action_index = random.randint(0, len(self.actions.keys()) - 1)
        else:
            next_action_index = states_possibilities.index(max(states_possibilities))

        return next_action_index
    
    def update_state_values(self):
        rew = 0
        number_states = len(self.states_history) - 1

        for i in range(number_states, -1, -1):
            state_row, state_col = self.states_history[i]

            # Check if its outside the maze
            if not self.is_inside_maze((state_row, state_col)):
                rew += self.gamma ** (number_states - i) * self.reward_for_leaving_limits
            else:
                rew += self.gamma ** (number_states - i) * self.environment[state_row][state_col]

        self.state_values[self.initial_coords[0], self.initial_coords[1]] = rew

    def change_state(self, next_states, exploration_prob):
        next_states_values = []

        for state in next_states:
            row, col = state
            if self.is_inside_maze(state):
                next_states_values.append(self.state_values[row, col])
            else:
                next_states_values.append(self.reward_for_leaving_limits)

        index = 0

        if len(set(next_states_values)) != 1 and exploration_prob < self.epsilon:
            index = next_states_values.index(max(next_states_values))

        else:
            index = random.randint(0, len(next_states) - 1)
        
        return next_states[index]
        
    def move_throught_environment(self):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        # -1 --> Left the maze
        # 0 --> Is inside maze
        # 1 --> Reached objective
        game_state = 0
        
        # Check the values on the QValue table for next action
        next_states_available = [
            [self.actual_coords_y - 1, self.actual_coords_x], # UP
            [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
            [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
            [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
        ]

        # Calculate distances
        distances = []

        for state in next_states_available:
            _, dis = self.instant_reward(state)
            distances.append(dis)

        next_action_index = self.choose_action(next_states_available, distances.index(min(distances)), random.random())

        next_states = next_states_available[next_action_index]

        # Save the current state in history list
        self.states_history.append([self.actual_coords_y, self.actual_coords_x])

        # Update position
        self.actual_coords_y = next_states[0]
        self.actual_coords_x = next_states[1]

        if not self.is_inside_maze((self.actual_coords_y, self.actual_coords_x)) or self.arrived_final((self.actual_coords_y, self.actual_coords_x)):
            self.update_state_values()
            
            # Select next state which has the biggest state value with some probability
            new_state = self.change_state(next_states_available, random.random())
            
            self.initial_coords = new_state[0], new_state[1]
            
            self.actual_coords_y = new_state[0]
            self.actual_coords_x = new_state[1]

        return game_state, self.state_values