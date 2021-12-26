class Agent:

    def __init__(self, reward_per_step, reward_reaching_final, x = 0, y = 0, lr=1e-3, gamma = 1e-4):
        self.actions = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3
        }

        self.x = x
        self.y = y

        self.gamma = gamma

        self.learning_rate = lr

        self.reward_per_step = reward_per_step
        self.reward_reaching_final = reward_reaching_final
    
    def set_environmet(self, env):
        self.environment = env

    def move_throught_environment(self, action):
        pass

    def distance_to_objective(self, final_coords):
        return ((self.x - final_coords[0])**2 + (self.y - final_coords[1])**2)**0.5

    def reward(self, action, final_coords):
        reward = 1 / self.distance_to_objective(final_coords)

