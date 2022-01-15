import numpy as np
from maze import Maze
from agent import Agent
import matplotlib.pyplot as plt

# Maze parameters
rows = 6
cols = 5
final = rows - 2, cols - 2
reward_per_step = -0.04
reward_reaching_final = 1.0

# Agent parameters
discount_rate = 0.95

maze = Maze()

environment = maze.create_maze(final, rows, cols, reward_per_step=reward_per_step, reward_reaching_final=reward_reaching_final)
location_blocks = maze.generate_blocks(2)

agent = Agent(x = 0, y = 0, discount_rate = discount_rate)
agent.set_environmet(environment, location_blocks, final)

position_state = False

game_state, qvalues, qvalues_map = agent.move_throught_environment(50)

print(qvalues)
print("\n")
print(qvalues_map)
