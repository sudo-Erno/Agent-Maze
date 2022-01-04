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
    
    def set_position(self, y, x):
        self.actual_coords_y = y
        self.actual_coords_x = x

    def set_environmet(self, env, location_blocks, final):
        # Save the Maze
        self.environment = env
        self.location_blocks = location_blocks
        
        # Save the coordinates of the exit
        self.final = final

        self.QValues = np.zeros((self.environment.shape[0], self.environment.shape[1], len(self.actions)))
        # self.QValues[0, 0, 2] = 1.0

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

    def plot_qtable(self):
        q_table = np.array(self.QValues, dtype=np.str)
        q_table[self.final] = "F"
        q_table[self.initial_y][self.initial_x] = "A"
        print(q_table)

    def choose_action(self, next_states, min_distance_index, b_prob=70, s_prob=10):
        """
        Returns action number (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        """
        states_possibilities = []

        for i in range(len(next_states)):
            
            state = list(self.actions.keys())
            state = state[i]

            if i == min_distance_index:
                for j in range(b_prob):
                    states_possibilities.append(state)
            
            else:
                for j in range(s_prob):
                    states_possibilities.append(state)

        return random.choice(states_possibilities)

    def change_state(self, prob, next_states):
        """
        Greedy method
        """
        # Calculate distances
        distances = []

        for state in next_states:
            _, dis = self.instant_reward(state)
            distances.append(dis)

        if prob < self.epsilon:
            min_distance_index = distances.index(min(distances))
            next_state = next_states[min_distance_index]

            max_value = self.QValues[next_state[0], next_state[1], min_distance_index]
            max_index = min_distance_index
            # max_index = list(self.QValues[self.actual_coords_y, self.actual_coords_x]).index(max_value)
            
            return max_value, max_index
        
        random_action_index = random.randint(0, 3)
        next_state = next_states[random_action_index]

        while not self.is_inside_maze(next_state):
            random_action_index = random.randint(0, 3)
            next_state = next_states[random_action_index]
        
        return self.QValues[next_state[0], next_state[1], random_action_index], random_action_index
        

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

        future_state_q_value, action_number = self.change_state(random.random(), next_states)
        
        # print(f"{future_state_q_value = } {action_number = }")

        self.QValues[self.actual_coords_y, self.actual_coords_x, action_number] = self.environment[self.actual_coords_y, self.actual_coords_x] + self.gamma * future_state_q_value - self.QValues[self.actual_coords_y, self.actual_coords_x, action_number]

        self.actual_coords_y = next_states[action_number][0]
        self.actual_coords_x = next_states[action_number][1]
        
        """
        min_distance_index = distances.index(min(distances))

        action_number = self.choose_action(next_states, min_distance_index, 70, 10)

        new_state = next_states[action_number]

        inside_maze = self.is_inside_maze(new_state)
        
        sum_probabilities = 0
        # for i in range(len(self.actions)):
        #     if i == action_number:
        #         self.QValues[self.actual_coords_y, self.actual_coords_x, i] = 


        # TODO: Make the agent stay in its place
        if inside_maze:
            # Update state of the agent
            self.actual_coords_y = new_state[0]
            self.actual_coords_x = new_state[1]

        """
        print("\tQ-VALUES\t")
        print(self.QValues)
        print("\t\t")
        
        # self.plot_qtable()

        return get_to_final