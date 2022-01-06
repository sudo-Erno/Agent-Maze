import numpy as np
from maze import Maze
from agent import Agent
import matplotlib.pyplot as plt

# Maze parameters
rows = 5
cols = 5
final = rows - 1, cols - 1
reward_per_step = -0.04
reward_reaching_final = 1.0

# Agent parameters
gamma = 0.80
epsilon = 0.1

maze = Maze()

environment = maze.create_maze(final, rows, cols, reward_per_step=reward_per_step, reward_reaching_final=reward_reaching_final)
location_blocks = maze.generate_blocks(2)

agent = Agent(x=0, y=0, gamma=gamma, epsilon=epsilon)
agent.set_environmet(environment, location_blocks, final)

game_state = False
state_actions = 0

for i in range(3):
    game_state, state_actions = agent.move_throught_environment()

print("\n")
print(state_actions)

# for i in range(5):
#     while game_state == False:
#         game_state = agent.move_throught_environment()
#     agent.set_position(0, 0)

# maze.plot_maze()