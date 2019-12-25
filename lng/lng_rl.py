import numpy as np
from lng_gym import SingleLngEnv
import time


class DdpgLngSolver(object):
    def __init__(self, env, agent, seed):
        self._seed = seed
        self.env = env
        self.agent = agent

    def solve(self):
        env.seed(self._seed)
        env.verbose = True
        states = env.reset()
        env.render()

        while True:
            action = self.agent.act(states)
            choice = np.argmax(action)
            states, _, done, _ = env.step(choice)
            env.render()
            if done:
                break
            time.sleep(0.01)

seed = 3721
env = SingleLngEnv(
    n_loc=10,
    n_steps=1000,
    fuel_cost=0.1,
    price_sigma=0.1,
    price_daily_vol=0.02,
    price_theta=0.01,
    max_distance=30.0,
    normalize=True
)

# from ddpg import Agent
from dqn import Agent
agent = Agent(state_size=env.n_loc * 3 + 1, action_size=env.n_loc, random_seed=seed)
agent.train(env, 1000, 1000)

solver = DdpgLngSolver(env, agent, 3721)
solver.solve()
