import numpy as np

class Agent:

    def __init__(self, x = 0, y = 0, lr=1e-3, gamma = 1e-4, reward_for_leaving_limits = -0.75):
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

        # Save the coordinates of the exit
        self.final = final

    def distance_to_objective(self, new_coordinates):
        return ((new_coordinates[0] - self.final_coords_x)**2 + (new_coordinates[1] - self.final_coords_y)**2)**0.5

    def move_throught_environment(self, action):
        """
        Given an action, it will calculate the reward.

        Returns True if it has arrived to destiny.   
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
            self.reward += self.reward_for_leaving_limits
        
        elif self.actual_coords_y < 0 or self.actual_coords_y > self.final[1]:
            self.reward += self.reward_for_leaving_limits
        
        # Check if it has arrive at the exit
        if self.actual_coords_x == self.final[0] and self.actual_coords_y == self.final[1]:
            self.reward += self.reward_reaching_final
            return True

        self.reward += self.environment[self.actual_coords_x, self.actual_coords_y]

        # print(self.reward)
        return False