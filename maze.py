import numpy as np
from random import randint

class Maze:

    def create_maze(self, final, rows=5, cols=5, initial=(0, 0)):

        self.maze = np.ones((rows, cols), dtype=np.str)

        self.initial = initial
        self.final = final

        self.maze[initial] = 'A'
        self.maze[final] = 'f'

        return self.maze
    
    def generate_blocks(self, number_blocks):
        
        for i in range(number_blocks + 1):
            bx = randint(int(self.initial[0]) + 1, int(self.final[0]) - 1)
            by = randint(int(self.initial[1]) + 1, int(self.final[1]) - 1)

            self.maze[bx, by] = 'X'

        return self.maze