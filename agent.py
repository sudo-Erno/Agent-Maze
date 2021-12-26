import numpy as np

class Agent:

    def __init__(self, reward_per_step, reward_reaching_final, x = 0, y = 0, lr=1e-3, gamma = 1e-4, reward_for_leaving_limits = -0.75):
        self.actions = ["up", "right", "down", "left"]

        # Set actual coordinates of the agent
        self.actual_coords_x = x
        self.actual_coords_y = y

        # Initialize the gamma for the Q-function
        self.gamma = gamma

        # Initialize the learning rate for the Q-function
        self.learning_rate = lr

        # Set the reward per step and the reward for reaching the exit (final)
        self.reward_per_step = reward_per_step
        self.reward_reaching_final = reward_reaching_final
        self.reward_for_leaving_limits = reward_for_leaving_limits

        # Initialize the reward
        self.reward = 0
    
    def set_environmet(self, env, final):
        self.environment = env # Save the Maze

        # Save the coordinates of the exit
        self.final = np.where(self.environment == "f")
        self.final = self.final[0][0], self.final[1][0]

    def calculate_reward(self, action):
        """
        Given and action, it will update the coordinates and calculate the reward.
        """
        
        if action == "up":
            self.actual_coords_y -= 1
        elif action == "right":
            self.actual_coords_x += 1
        elif action == "down":
            self.actual_coords_y += 1
        elif action == "left":
            self.actual_coords_x -= 1
        
        # Check if its outside the limits
        if self.actual_coords_x < 0 or self.actual_coords_x > self.final[0]:
            return self.reward_for_leaving_limits
        
        elif self.actual_coords_y < 0 or self.actual_coords_y > self.final[1]:
            return self.reward_for_leaving_limits
        
        # Check if it has arrive at the exit
        if self.actual_coords_x == self.final[0] and self.actual_coords_y == self.final[1]:
            return self.reward_reaching_final
        
        return self.reward_per_step

    def distance_to_objective(self, new_coordinates):
        return ((new_coordinates[0] - self.final_coords_x)**2 + (new_coordinates[1] - self.final_coords_y)**2)**0.5

    def move_throught_environment(self, action):
        """
        Given an action, it will estimate the reward        
        """
        self.reward = self.calculate_reward(action)
        
        print(self.reward)