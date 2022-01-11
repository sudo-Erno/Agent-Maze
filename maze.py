import numpy as np
from random import randint

from numpy.lib.function_base import disp

class Maze:

    def create_maze(self, final, rows=5, cols=5, initial=(0, 0), reward_per_step=-0.04, reward_reaching_final=1.0):
        """
        Given the size and the initial and final coordinates, it will generate a maze.
        """
        self.maze = np.ones((rows, cols)) * reward_per_step

        self.location_blocks = [] # Save the coordinates of the blocks at the maze

        # Save the coordinates of the initial and final position
        self.initial = initial
        self.final = final

        self.reward_per_step = reward_per_step

        self.maze[final[0], final[1]] = reward_reaching_final
        self.maze[initial[0], initial[1]] = 0

        return self.maze
    
    def observe_maze(self):
        """
        Returns maze and the location of the blocks.
        """
        return self.maze, self.location_blocks
    
    def generate_blocks(self, number_blocks):
        
        for i in range(number_blocks):
            bx = randint(int(self.initial[0]) + 1, int(self.final[0]) - 1)
            by = randint(int(self.initial[1]) + 1, int(self.final[1]) - 1)

            self.location_blocks.append((bx, by))

        return self.location_blocks

    def plot_maze(self):

        display_maze = self.maze.astype("str")
        display_maze[self.final] = "f"
        display_maze[self.initial] = "A"
        display_maze[[block for block in self.location_blocks]] = "X"
        display_maze[display_maze == str(self.reward_per_step)] = ""
        
        print(display_maze)
