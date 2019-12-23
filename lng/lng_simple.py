import numpy as np
from lng_gym import SingleLngEnv
import time


class SimpleLngSolver(object):

    def solve(self, env):
        env.seed(3722)
        env.verbose = True
        states = env.reset()
        env.render()
        fuel_cost = env.fuel_cost

        while True:
            distances = np.linalg.norm(states[:, :2], axis=1)
            if env._cost is None:
                profits = -states[:, 2] - distances * fuel_cost
            else:
                profits = states[:, 2] - distances * fuel_cost
            action = np.argmax(profits)
            states, _, done, _ = env.step(action)
            env.render()
            if done:
                break
            time.sleep(0.01)


env = SingleLngEnv(
    n_loc=10,
    n_steps=1000,
    fuel_cost=0.1,
    price_sigma=0.1,
    price_daily_vol=0.02,
    price_theta=0.01,
    max_distance=30.0
)
solver = SimpleLngSolver()
solver.solve(env)