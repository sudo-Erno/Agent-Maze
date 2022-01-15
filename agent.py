import numpy as np
import random

class Agent:

    def __init__(self, x = 0, y = 0, lr = 1e-3, discount_rate = 0.99, epsilon = 0.9, reward_for_leaving_limits = -0.75):
        # Up: 0, Right: 1, Down: 2, Left: 3
        self.actions = {0: "up", 1: "right", 2: "down", 3: "left"}

        # Set initial coordinates
        self.initial_x = x
        self.initial_y = y

        # Set actual coordinates of the agent
        self.actual_coords_x = x
        self.actual_coords_y = y

        # Initialize the gamma for the Q-function
        self.discount_rate = discount_rate

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
        self.QValues[self.final] = 1.0

    def instant_reward(self, new_coordinates):
        d = ((new_coordinates[0] - self.final[0])**2 + (new_coordinates[1] - self.final[1])**2)**0.5
        if d != 0:
            return 1 / d
        else:
            return 9

    def calculate_distances(self, states):
        dist = list()

        for state in states:
            dist.append(((state[0] - self.final[0])**2 + (state[1] - self.final[1])**2)**0.5)

        return dist
    
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

    def draw_final(self):

        qvalues_map = np.zeros_like(self.environment, dtype=np.str)

        for row in range(self.environment.shape[0]):
            
            for col in range(self.environment.shape[1]):

                values = list()

                next_states = [
                    [row - 1, col], # UP
                    [row, col + 1], # RIGHT
                    [row + 1, col], # DOWN
                    [row, col - 1] # LEFT
                ]

                for state in next_states:
                    i, j = state
                    
                    if self.is_inside_maze(state):
                    
                        values.append(self.QValues[i][j])
                    
                    else:
                        values.append(float("-inf")) # Low value so it's not selected

                max_state_value_index = values.index(max(values))

                if self.arrived_final((row, col)):
                    qvalues_map[row][col] = "F"

                elif max_state_value_index == 0:
                    qvalues_map[row][col] = "UP"
                
                elif max_state_value_index == 1:
                    qvalues_map[row][col] = "RIGHT"
                
                elif max_state_value_index == 2:
                    qvalues_map[row][col] = "DOWN"
                
                else:
                    qvalues_map[row][col] = "LEFT"

        return qvalues_map


    def move_throught_environment(self, steps=5):
        """
        Returns 1 if it has arrived to destiny, 0 if it has not arrived and -1 if it has left the maze.
        """
        for i in range(steps):
            
            game_state = 0

            while game_state != 1:
                
                # Check the values on the QValue table for next action
                posibles_next_states = [
                    [self.actual_coords_y - 1, self.actual_coords_x], # UP
                    [self.actual_coords_y, self.actual_coords_x + 1], # RIGHT
                    [self.actual_coords_y + 1, self.actual_coords_x], # DOWN
                    [self.actual_coords_y, self.actual_coords_x - 1] # LEFT
                ]

                index_next_state = 0

                if random.random() < self.epsilon:
                    
                    next_states_values = list()

                    for state in posibles_next_states:
                        
                        r, c = state

                        if self.is_inside_maze(state):
                            
                            next_states_values.append(self.QValues[r][c])
                        
                        else:

                            next_states_values.append(self.reward_for_leaving_limits)

                    # Check if they are the same
                    if len(set(next_states_values)) == 1:
                        
                        rnd = random.randint(0, len(self.actions.keys()) - 1)
                        
                        index_next_state = posibles_next_states[rnd]
                    
                    else:

                        index_next_state = next_states_values.index(max(next_states_values))
                        print()
                    
                    # next_states = self.calculate_distances(posibles_next_states)

                    # next_state = next_states.index(min(next_states))
                    
                    print(f"{index_next_state = }")

                    next_state = posibles_next_states[index_next_state]
                
                else:
                    
                    rnd = random.randint(0, len(self.actions.keys()) - 1)

                    next_state = posibles_next_states[rnd]

                while not self.is_inside_maze(next_state):
                    
                        rnd = random.randint(0, len(self.actions.keys()) - 1)
                        
                        next_state = posibles_next_states[rnd]

                next_row, next_col = next_state

                self.QValues[self.actual_coords_y][self.actual_coords_x] = self.epsilon * (self.environment[next_row][next_col] + self.discount_rate * self.QValues[next_row][next_col])

                # Update position
                self.actual_coords_y = next_row
                self.actual_coords_x = next_col

                if self.arrived_final((self.actual_coords_y, self.actual_coords_x)):
                    game_state = 1

                    # Start in random place
                    self.actual_coords_y = random.randint(0, self.environment.shape[0] - 1)
                    self.actual_coords_x = random.randint(0, self.environment.shape[1] - 1)

        qvalues_map = self.draw_final()

        return game_state, self.QValues, qvalues_map