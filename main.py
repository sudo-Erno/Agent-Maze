import numpy as np
from maze import Maze
from agent import Agent

# Maze parameters
rows = 5
cols = 5
final = rows - 1, cols - 1

# Agent parameters
reward_per_step = -0.04
reward_reaching_final = 1

maze = Maze()

environment = maze.create_maze(final, rows, cols)
environment = maze.generate_blocks(2)

agent = Agent(reward_per_step, reward_reaching_final)
agent.set_environmet(maze)

print(environment)

# agent.move_throught_maze(environment, )