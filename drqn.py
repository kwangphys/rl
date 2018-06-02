import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LSTM, TimeDistributed, Dot
import keras.optimizers as optimizers
from keras.utils import to_categorical
import gym


class Environment:
    def __init__(self):
        pass

    def is_end(self):
        """ returns true if is end step
        """
        pass

    def reset(self):
        """ reset internal states and returns initial state
        """
        pass

    def step(self, action):
        """ returns (new_state, reward, is_end)
        """

class Memory:
    def __init__(self, max_memory_size, trace_size):
        self.max_memory_size = max_memory_size
        self.trace_size = trace_size
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.episode_sizes = []
        self.total_size = 0

    def clear(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.episode_sizes = []
        self.total_size = 0

    def remember_episode(self, states, actions, rewards):
        """ An episode is a time series of memories.
            Each of format (state, action, reward, new_state, is_end)
        """
        self.state_memory.append(states)
        self.action_memory.append(actions)
        self.reward_memory.append(rewards)
        len_e = actions.shape[0] - self.trace_size + 1
        self.episode_sizes.append(len_e)
        self.total_size += len_e
        istart = 0
        while self.total_size > self.max_memory_size:
            self.total_size -= self.episode_sizes[istart]
            istart += 1
        self.episode_sizes = self.episode_sizes[istart:]
        self.state_memory = self.state_memory[istart:]
        self.action_memory = self.action_memory[istart:]
        self.reward_memory = self.reward_memory[istart:]

    def replay(self, batch_size):
        ez = np.array(self.episode_sizes)
        if np.min(ez) <= 0:
            print(ez)
        episodes = np.random.choice(len(self.episode_sizes), size=batch_size, p=ez / self.total_size)
        sizes = ez[episodes]
        indices = (np.random.random(size=batch_size) * sizes).astype(np.int)
        # indices = (np.array(sizes) - 1).astype(np.int)

        state_traces = [self.state_memory[episodes[i]][indices[i]:indices[i] + self.trace_size + 1] for i in range(batch_size)]
        action_traces = [self.action_memory[episodes[i]][indices[i]:indices[i] + self.trace_size] for i in range(batch_size)]
        reward_traces = [self.reward_memory[episodes[i]][indices[i]:indices[i] + self.trace_size] for i in range(batch_size)]
        end_traces = [[j == sizes[i] + self.trace_size - 2 for j in range(indices[i], indices[i] + self.trace_size)] for i in range(batch_size)]
        # if np.any(end_traces):
        #     print('End')
        if np.any(np.array(end_traces)[:, :-1]):
            print('Wrong!')
        return np.array(state_traces), np.array(action_traces), np.array(reward_traces), np.array(end_traces)


class DeepRecurrentQNetwork:
    def __init__(
        self,
        env,
        hidden_sizes,
        max_episode_size,
        skip_front,
        trace_size,
        memory_size,
        discount_rate=1.0,
        learning_rate=0.001
    ):
        self.env = env
        self.hidden_sizes = hidden_sizes
        self.memory = Memory(memory_size, trace_size)
        self.max_episode_size = max_episode_size
        self.skip_front = skip_front
        self.trace_size = trace_size
        self.gamma = discount_rate    # discount rate
        self.learning_rate = learning_rate
        # TODO: Here assumes action space is 1-d discrete
        self.action_size = self.env.action_space.n
        self.main_model = self._build_model(1)

    def _build_model(self, batch_size, include_train_model=False):
        # TODO: Here assumes state space is 1-d
        state_size = self.env.observation_space.shape[0]

        input = Input(batch_shape=(batch_size, None, state_size), name='input')
        lstm_out = LSTM(self.hidden_sizes[0], stateful=True, return_sequences=True, name='lstm')(input)
        qout = TimeDistributed(Dense(self.action_size, activation=None), name='qout')(lstm_out)
        model = Model(inputs=[input], outputs=[qout])
        model.compile(loss={'qout': 'logcosh'}, optimizer=optimizers.adam(lr=self.learning_rate))
        if not include_train_model:
            return model

        action_input = Input(batch_shape=(batch_size, None, self.action_size), name='action')
        q = Dot(axes=-1)([qout, action_input])
        model2 = Model(inputs=[input, action_input], outputs=[q])
        model2.compile(loss='logcosh', optimizer=optimizers.adam(lr=self.learning_rate))
        return model, model2

    def run(self):
        env = self.env
        self.main_model.reset_states()
        input_state = env.reset()
        is_end = False
        while not is_end:
            env.render()
            qs = self.main_model.predict(input_state.reshape(1, 1, -1), batch_size=1)
            action = np.argmax(qs.ravel(), axis=-1)
            input_state, reward, is_end, info = env.step(action)

    def train(
            self,
            update_freq=10,
            batch_size=4,
            n_epochs=100,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
    ):
        target_model = self._build_model(batch_size)
        q_model, q_train_model = self._build_model(batch_size, include_train_model=True)
        q_model.set_weights(self.main_model.get_weights())
        target_model.set_weights(self.main_model.get_weights())
        env = self.env

        self.memory.clear()
        e = epsilon
        state_memory = [None for _ in range(self.max_episode_size)]
        action_memory = [None for _ in range(self.max_episode_size)]
        reward_memory = [None for _ in range(self.max_episode_size)]
        itotal = 0
        # all_rewards = np.zeros(len(envs))
        for i in range(n_epochs):
            ienv = 0
            self.main_model.reset_states()
            input_state = env.reset()
            is_end = False
            ie = 0
            state_memory[0] = input_state
            while not is_end:
                env.render()
                q0s = self.main_model.predict(input_state.reshape(1, 1, -1), batch_size=1)
                if np.random.rand() <= epsilon:
                    action = np.random.randint(self.action_size)
                else:
                    action = np.argmax(q0s.ravel(), axis=-1)
                new_input_state, reward, is_end, info = env.step(action)
                if ie + 1 >= len(state_memory):
                    print('Out of range!')
                state_memory[ie + 1] = new_input_state
                action_memory[ie] = action
                reward_memory[ie] = reward
                input_state = new_input_state
                ie += 1
                itotal += 1

                if itotal >= self.skip_front:
                    if e > epsilon_min:
                        e *= epsilon_decay

                    if itotal % update_freq == 0:
                        states, actions, rewards, ends = self.memory.replay(batch_size)
                        next_inputs_batch = states[:, 1:]
                        qs = q_model.predict(next_inputs_batch, batch_size=batch_size)
                        next_actions = np.argmax(qs, axis=-1)
                        q2s = target_model.predict(next_inputs_batch, batch_size=batch_size)
                        next_targets = q2s.reshape(-1, self.action_size)[np.arange(batch_size*self.trace_size), next_actions.ravel()].reshape(batch_size, self.trace_size)
                        # next_targets = q2s[..., actions]
                        # rewards = np.vstack([v[3] for v in train_batch])
                        inputs = states[:, :-1]
                        # ends = np.zeros(rewards.shape)
                        # ends[..., -1] = 1.0
                        # Bellman equation:
                        #   Q(t) = R(t) + gamma * Q(t+1)
                        targets = (rewards + (1 - ends) * next_targets * self.gamma).reshape(batch_size, self.trace_size, -1)
                        actions = to_categorical(actions, num_classes=self.action_size).reshape(batch_size, self.trace_size, -1)

                        target_model.set_weights(self.main_model.get_weights())
                        q_model.reset_states()
                        q_train_model.fit(
                            [inputs.reshape(batch_size, self.trace_size, -1), actions],
                            targets,
                            batch_size=batch_size,
                            epochs=1,
                            shuffle=False,
                            verbose=0
                        )
                        self.main_model.set_weights(q_model.get_weights())
                        target_model.reset_states()
                        q_model.reset_states()
            # all_rewards[ienv] = np.mean(reward_memory[:ie])
            # print('Env #' + str(ienv) + ':', all_rewards[ienv] * 10000, 'E:', e)
            ienv += 1
            self.memory.remember_episode(np.array(state_memory[:ie + 1]), np.array(action_memory[:ie]), np.array(reward_memory[:ie]))
            print('Episode #' + str(i) + ':', ie, e, np.max(q0s))


class IntradayPriceEnvironment(Environment):
    def __init__(self, inputs, returns, slippage_rate=0):
        super().__init__()
        self._istep = 0
        self._position = 0
        self._inputs = inputs
        self._returns = returns
        self._slippage_rate = slippage_rate

    def is_end(self):
        """ returns true if is end step
        """
        return self._istep >= self._inputs.shape[0] - 1

    def reset(self):
        self._istep = 0
        self._position = 0
        return np.concatenate([self._inputs[0], [0]])

    def step(self, action):
        """ returns (new_state, reward, is_end)
        """
        new_position = min([max([-1, self._position + action - 1]), 1])
        pnl = new_position * self._returns[self._istep]
        slippage = abs(new_position - self._position) * self._slippage_rate
        reward = pnl - slippage
        self._istep += 1
        self._position = new_position
        new_state = np.concatenate([self._inputs[self._istep], [new_position]])
        return new_state, reward, self.is_end()


if __name__ == "__main__":
    import os
    from tensorflow import set_random_seed
    np.random.seed(3721)
    set_random_seed(3721)

    # from lstm_daily import *
    # dataset = pickle.load(open('/home/kaiwang/Documents/data/experiments/Kaggle_Daily/processed_dataset.pkl', 'rb'))
    # dataset = filter_and_split_dataset(
    #     dataset,
    #     datetime.date(2015, 1, 1),
    #     datetime.date(2016, 1, 1),
    #     start_date=datetime.date(2006, 1, 1),
    #     n_skip_front=30,
    #     min_train_sequence_length=250
    # )
    # input_scalers = [
    #     pp.MinMaxScaler(feature_range=(-1, 1)),
    #     pp.MinMaxScaler(feature_range=(-1, 1)),
    #     pp.MinMaxScaler(feature_range=(-1, 1)),
    #     pp.RobustScaler(),
    #     pp.RobustScaler(),
    #     pp.RobustScaler(),
    #     pp.RobustScaler(),
    #     pp.RobustScaler(),
    #     None
    # ]
    # output_scalers = [pp.RobustScaler()]
    # scale_dataset(dataset, input_scalers, output_scalers)
    # train_envs = []
    # valid_envs = []
    # test_envs = []
    # for tag, data in dataset.items():
    #     train_envs.append(IntradayPriceEnvironment(data['train inputs'], data['train outputs'].ravel()))
    #     valid_envs.append(IntradayPriceEnvironment(data['valid inputs'], data['valid outputs'].ravel()))
    #     test_envs.append(IntradayPriceEnvironment(data['test inputs'], data['test outputs'].ravel()))
    # dataset = None
    # max_episode_size = max([env._inputs.shape[0] for env in train_envs])
    #
    # drqn = DeepRecurrentQNetwork(
    #     9,
    #     3,
    #     [128],
    #     max_episode_size,
    #     10000,
    #     16,
    #     10000,
    #     discount_rate=0.98,
    #     learning_rate=0.001
    # )
    # drqn.train(
    #     train_envs[:25],
    #     update_freq=10,
    #     batch_size=64,
    #     n_epochs=100,
    #     epsilon=1.0,
    #     epsilon_min=0.1,
    #     epsilon_decay=0.999999,
    # )


    env = gym.make('CartPole-v0')
    drqn = DeepRecurrentQNetwork(
        env,
        [64],
        1000,
        1000,
        8,
        50000,
        discount_rate=0.9,
        learning_rate=0.001
    )
    drqn.train(
        update_freq=10,
        batch_size=128,
        n_epochs=10000,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9999,
    )
    drqn.run()
    print('All Done!')
# from gridworld import gameEnv
# env = gameEnv(partial=True, size=9)
# drqn = DeepRecurrentQNetwork(
#     9,
#     3,
#     [128],
#     max_episode_size,
#     10000,
#     16,
#     10000,
#     discount_rate=0.98,
#     learning_rate=0.001
# )
# drqn.train(
#     train_envs,
#     update_freq=10,
#     batch_size=64,
#     n_epochs=100,
#     epsilon=1.0,
#     epsilon_min=0.1,
#     epsilon_decay=0.9999,
# )