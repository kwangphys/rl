import numpy as np
from gym import Env
import gym.spaces as spaces
from gym.utils import seeding
from pyglet.gl import *

class SingleLngEnv(Env):

    reward_range = (-float('inf'), float('inf'))
    metadata = {
        'render.modes': ['human'], #, 'rgb array'],
        'video.frames_per_second': 4
    }

    def __init__(self, n_loc, n_steps, fuel_cost, price_sigma, price_daily_vol, price_theta, max_distance):
        """ Vehicle speed is always 1
        :param n_loc: number of locations
        :param n_steps: number of steps for each run
        :param fuel_cost: fuel cost for travelling unit distance
        :param price_sigma: sigma for price distribution
        :param price_daily_vol: volatility for daily prices
        :param price_theta: mean reversion strength for daily prices
        :param max_distance: max_distance of the region
        """
        self.base_price = 100.0
        self.log_base_price = np.log(self.base_price)
        self.action_space = spaces.Discrete(n_loc)
        self.lows = np.array([-max_distance, -max_distance, self.base_price * np.exp(-price_sigma * 3.5)])
        self.highs = np.array([max_distance, max_distance, self.base_price * np.exp(price_sigma * 3.5)])
        # Observation space is a 1-d array consisting of n_loc relative xs, n_loc relative ys, n_loc prices, and a state of -1/0/1
        self.observation_space = spaces.Box(low=np.tile(self.lows, (n_loc, 1)), high=np.tile(self.highs, (n_loc, 1)), dtype=np.float32)
        self.fuel_cost = fuel_cost
        self.n_loc = n_loc
        self.n_steps = n_steps
        self.price_sigma = price_sigma
        self.max_distance = max_distance
        self.price_vol = price_daily_vol
        self.price_theta = price_theta
        self.long_term_vol = self.price_vol / np.sqrt(2.0 * self.price_theta)
        self.ethetadt = np.exp(-self.price_theta)
        self.one_step_vol = self.long_term_vol * np.sqrt(1.0 - self.ethetadt ** 2)
        self.verbose = False

        self._locations = None
        self._prices = None
        self._logprices = None

        self._prve_action = None
        self._istep = 0
        self._pos = None
        self._cost = None
        self._profit = 0.0
        self._mtm = 0.0
        self._relative_locations = None
        self._curr_state = -1
        self._prve_state = -1

        self.seed()
        self.window = None

    def normalize_states(self, states):
        normed_states = states.T.flatten()
        normed_states[:self.n_loc * 2] /= self.highs[0] * 2.0
        normed_states[self.n_loc * 2:] = (np.log(normed_states[self.n_loc * 2:]) - self.log_base_price) / self.long_term_vol / np.sqrt(self.n_steps)
        normed_states = np.concatenate([normed_states, np.array([self._curr_state])])
        return normed_states

    def step(self, action):
        self._logprices = self._logprices * self.ethetadt + self.log_base_price * (1.0 - self.ethetadt) + self.np_random.normal(0.0, self.one_step_vol, size=self.n_loc)
        self._prices = np.exp(self._logprices)
        # self._prices *= np.exp(self.np_random.normal(0.0, self.price_vol, size=self.n_loc))
        self._prev_action = action
        diff = self._locations[action] - self._pos
        distance = np.linalg.norm(diff)
        reward = -self.fuel_cost * min(distance, 1.0)
        self._prve_state = self._curr_state
        if self._cost is None: # After dump, before load
            mtm = -self._prices[action]
            if self._mtm > 0: # right after dump
                reward += mtm
            else:
                reward += mtm - self._mtm
        else:
            mtm = self._prices[action]
            if self._mtm < 0: # right after load
                reward += mtm
            else:
                reward += mtm - self._mtm
        self._mtm = mtm

        if distance <= 1.0:  # Arrived
            self._pos = np.array(self._locations[action])
            if self._cost is None: # Load from this port
                self._cost = self._prices[action]
                self._curr_state = 1
            else: # Dump to this port
                # reward += self._prices[action] - self._cost
                self._cost = None
                self._curr_state = -1
        else:
            self._pos += diff / distance
            self._curr_state = 0

        self._profit += reward
        self._relative_locations = self._locations - self._pos
        # profits = -self._prices if self._cost is None else self._prices - self._cost
        states = np.concatenate([self._relative_locations, self._prices.reshape(-1, 1)], axis=1)
        self._istep += 1
        if self.verbose:
            norm_states = self.normalize_states(states)
            print('Step', self._istep, 'State:', norm_states[-1], 'Action:', action, 'Reward:', reward, 'Position:', self._pos, 'Profit:', self._profit)
        return states, reward, self._istep >= self.n_steps, {}

    def reset(self):
        self._locations = self.np_random.uniform(0.0, self.max_distance, size=(self.n_loc, 2))
        self._prices = self.base_price * np.exp(self.np_random.normal(0.0, self.price_sigma, size=self.n_loc))
        self._prices = np.minimum(np.maximum(self._prices, self.lows[2]), self.highs[2])
        self._logprices = np.log(self._prices)
        self._istep = 0
        icurr = self.np_random.randint(self.n_loc)
        self._pos = np.array(self._locations[icurr])
        self._cost = None
        self._prev_action = None
        self._profit = 0.0
        self._mtm = 0.0
        self._curr_state = -1
        self._prev_state = -1
        self._relative_locations = self._locations - self._pos
        if self.verbose:
            print('Locations:')
            for i in range(self.n_loc):
                print(i, self._locations[i], self._prices[i])

        # profits = -self._prices if self._cost is None else self._prices - self._cost
        return np.concatenate([self._relative_locations, self._prices.reshape(-1, 1)], axis=1)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.highs[0]
        scale = screen_width / world_width

        if self.window is None:
            self.window = pyglet.window.Window(width=screen_width + 40, height=screen_height + 50, display=None)
            pyglet.gl.glClearColor(255, 255, 255, 255)
            pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
            pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
            self._lng_img = pyglet.image.load(r'lng.png')
            self._port_img = pyglet.image.load(r'port.png')

        centers = (self._locations * scale).astype(np.int)
        ship_loc = (self._pos * scale).astype(np.int)
        price_labels = [pyglet.text.Label("{0:.2f}".format(self._prices[i]), font_size=16, x=centers[i, 0],
                                          y=max(centers[i, 1] - 18, 0), color=(255, 0, 255, 255)) for i in range(self.n_loc)]
        cost_label = None if self._cost is None else pyglet.text.Label("{0:.2f}".format(self._cost), font_size=16, x=ship_loc[0],
                                                                       y=min(ship_loc[1] + 40, screen_height - 1), color=(255, 0, 0, 255))
        profit_label = pyglet.text.Label("Profits: {0:.2f}".format(self._profit), font_size=20, x=0, y=screen_height+ 25, color=(0, 0, 255, 255))

        @self.window.event
        def on_draw():
            self.window.clear()
            for i in range(self.n_loc):
                self._port_img.blit(centers[i, 0], centers[i, 1])
                price_labels[i].draw()
            self._lng_img.blit(ship_loc[0], ship_loc[1])
            if cost_label:
                cost_label.draw()
            profit_label.draw()

        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()
        return True

    def close(self):
        if self.window:
            self.window.exit()
            self.window = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]