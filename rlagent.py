import numpy as np
import warnings
from collections import deque
import random
import time

class DummyTransformer:
    def fit(self, x, y=None):
        return self

    def fit_transform(self, x, y=None):
        return x

    def transform(self, x, y=None):
        return x


class SarsaAgent:
    """Model-free on-policy reinforcement learning algorithm that solves the
    control problem through trial-and-error learning.

    The algorithm estimates the action-value function `Q-pi` of the behavior
    policy `pi`, and uses an exploration strategy to improve `pi` while
    increasing the policy's greediness.

    SarsaAgent can be fitted on gym environments, and its output is an action
    for a given state.

    Arguments:
    --------------
    gamma   : float, default 1.0
        Discount factor
    alpha   : float, default 0.25
        Learning rate
    epsilon : float or None, default None
        Stochastic exploration move percentage. Model's greediness can be
        defined by 1 - epsilon.
    seed    : int or None, default None
        Seed value to be passed on to environment and numpy
    epsilon_shrink : float, default 1.0
        Multiply epsilon by this value at the end of each episode
    alpha_shrink : float, default 1.0
        Multiply alpha (learning rate) by this value at the end of each episode
    max_episode_len : int or None, default None
        Maximum length of an episode. Defaults to infinity.

    Notes:
    ---------------
    SARSA is model-free because, unlike with value iteration and policy
    iteration, it does not need or use an MDP. It is on-policy because it learns
    about the same policy that generates behaviors.

    Implementation of the algorithm is per Sarsa (on-policy TD control) for
    estimating Q ≈ q* (Sutton, 2018, p. 130) Sutton, R. S., Barto, A. G.
    (2018). Reinforcement Learning: An Introduction,  2nd Edition
    """
    def __init__(self, gamma=1.0, alpha=0.25, epsilon=None, seed=None, epsilon_shrink=1.0, alpha_shrink=1.0, max_episode_len=None):
        self.gamma = gamma
        self.alpha = alpha
        self.env_ = None
        self.Qsa = None
        self.transition_freq = None
        self.exist_freq = None
        if epsilon is None:
            epsilon = 0.0
        self.epsilon = epsilon
        self.seed = seed
        self.epsilon_shrink = epsilon_shrink
        self.alpha_shrink = alpha_shrink
        self._abs_update_mean = []
        self._episode_time = []
        self._policy_diff = []
        self._rewards = []
        self._creation_time = time.time()
        if max_episode_len is None:
            max_episode_len = float('inf')
        self.max_episode_len = max_episode_len

    def _greedy_move(self, state):
        """Make a greedy move.

        Selects the action having the highest value for that state.
        """
        return self.Qsa[state].argmax()

    def _explore_move(self, state=None):
        """Make a random move.

        Randomly selects an action to explore environment.
        """
        return np.random.randint(self.env_.nA)

    def move(self, state):
        """Make a move.

        Depending on `epsilon` parameter, a move type is chosen and an action
        is selected.
        """
        is_greedy = np.random.random() >= self.epsilon
        if is_greedy:
            action = self._greedy_move(state)
        else:
            action = self._explore_move(state)
        return action

    def _Q_update_func(self, state, action, reward, state_p, done, action_p=None):
        """Calculate new Q function value for state-action pair."""
        Q_p = self.Q(state_p, action_p) if not done else 0.0
        return self.Q(state, action) \
               + self.alpha * (reward
                               + self.gamma * Q_p
                               - self.Q(state, action)
                               )

    def _fit_episode(self):
        """Update values for one iteration.

        An iteration is defined by a sequence of steps from starting point to a
        terminal state.
        """
        state = self.env_.reset()
        action = self.move(state)
        done = False
        old_Qtable = self.Qsa.copy()
        steps = 0
        episode_rewards = 0.
        while not done and steps <= self.max_episode_len:
            state_p, reward, done, _ = self.env_.step(action)
            episode_rewards += reward
            action_p = self.move(state_p)

            self.Qsa[state, action] = self._Q_update_func(state, action, reward, state_p, done, action_p)

            state = state_p
            action = action_p
            steps += 1
        self.epsilon *= self.epsilon_shrink
        self.alpha *= self.alpha_shrink
        episode_abs_mean = np.abs(old_Qtable - self.Qsa).max()
        self._abs_update_mean.append(episode_abs_mean)
        self._episode_time.append(time.time() - self._creation_time)
        self._policy_diff.append((old_Qtable.argmax(axis=1) != self.Qsa.argmax(axis=1)).sum())
        self._rewards.append(episode_rewards)

    def set_seed(self):
        """Set seed for numpy and environment."""
        if self.seed is not None:
            self.env_.seed(self.seed)
            np.random.seed(self.seed)

    def _init_Q(self, env):
        """Initiate Q value table."""
        if self.Qsa is None:
            self.Qsa = Qsa = np.zeros((env.nS, env.nA))
        if self.transition_freq is None:
            self.transition_freq = np.ones((env.nA, env.nS, env.nS), dtype=int)

    def _parse_env(self, env):
        """Parse environment parameters and initiate Q table."""
        self.env_ = env
        self._init_Q(env)

    def fit(self, env, iters=1000):
        """Fit agent to environment.

        Q table is reinitiated each time `fit` method is called.
        """
        self._parse_env(env)
        self.set_seed()

        for iter_ in range(iters):
            self._fit_episode()

    def partial_fit(self, env=None, iters=None):
        """Perform a one-step partial fit to environment.

        Q table is not reinitiated for partial fit.
        """
        if env is None:
            env = self.env_
        else:
            self._parse_env(env)

        if iters is None:
            iters = 1

        for iter_ in range(iters):
            self._fit_episode()

    def Q(self, state, action):
        """Get value of Q function for given state-action pair."""
        return self.Qsa[state, action]

    def policy(self, state):
        """Get action for given state."""
        return self.Qsa[state].argmax()

    @property
    def policy_map(self):
        return [self.policy(state) for state in range(self.env_.nS)]


class ExpectedSarsa(SarsaAgent):
    def _init_Q(self, env):
        """Initiate Q value table."""
        if self.Qsa is None:
            self.Qsa = Qsa = np.zeros((env.nS, env.nA))
        if self.transition_freq is None:
            self.transition_freq = np.ones((env.nA, env.nS, env.nS), dtype=int)

    def _fit_episode(self):
        """Update values for one iteration.

        An iteration is defined by a sequence of steps from starting point to a
        terminal state.
        """
        state = self.env_.reset()
        action = self.move(state)
        done = False
        old_Qtable = self.Qsa.copy()
        steps = 0
        episode_rewards = 0.
        while not done and steps <= self.max_episode_len:
            state_p, reward, done, _ = self.env_.step(action)
            self.transition_freq[action, state, state_p] += 1
            episode_rewards += reward
            action_p = self.move(state_p)

            self.Qsa[state, action] = self._Q_update_func(state, action, reward, state_p, done, action_p)

            state = state_p
            action = action_p
            steps += 1
        self.epsilon *= self.epsilon_shrink
        self.alpha *= self.alpha_shrink
        episode_abs_mean = np.abs(old_Qtable - self.Qsa).max()
        self._abs_update_mean.append(episode_abs_mean)
        self._episode_time.append(time.time() - self._creation_time)
        self._policy_diff.append((old_Qtable.argmax(axis=1) != self.Qsa.argmax(axis=1)).sum())
        self._rewards.append(episode_rewards)

    def _Q_update_func(self, state, action, reward, state_p, done, action_p=None):
        """Calculate new Q function value for state-action pair."""
        Q_p = (self.Qsa.max(axis=1) * self.transition_freq[action, state]).sum() / self.transition_freq[action, state].sum() if not done else 0.0
        return self.Q(state, action) \
               + self.alpha * (reward
                               + self.gamma * Q_p
                               - self.Q(state, action)
                               )


class QLAgent(SarsaAgent):
    """QLAgent is an off-policy TD control algorithm implementation, also known
    as Q-Learning (Watkins, 1989)

    QLAgent can be fitted on gym environments, and its output is an action
    for a given state.

    Arguments:
    --------------
    gamma   : float, default 1.0
        Discount factor
    alpha   : float, default 0.25
        Learning rate
    epsilon : float or None, default None
        Stochastic exploration move percentage. Model's greediness can be
        defined by 1 - epsilon.
    seed    : int or None, default None
        Seed value to be passed on to environment and numpy
    epsilon_shrink : float, default 1.0
        Multiply epsilon by this value at the end of each episode
    alpha_shrink : float, default 1.0
        Multiply alpha (learning rate) by this value at the end of each episode
    max_episode_len : int or None, default None
        Maximum length of an episode. Defaults to infinity.

    Notes:
    --------------
    Q-Learning algorithm implementation is adapted from Q-learning
    (off-policy TD control) for estimating π ≈ π* (Sutton, 2018)
    Sutton, R. S., Barto, A. G.  (2018).
    """
    def _Q_update_func(self, state, action, reward, state_p, done, action_p=None):
        """Calculate new Q value for state-action pair."""
        Q_p = self.Qsa[state_p].max() if not done else 0.0
        return self.Q(state, action) \
            + self.alpha * (reward
                            + self.gamma * Q_p
                            - self.Q(state, action)
                            )

    def _fit_episode(self, env=None):
        """Update values for one iteration.

        An iteration is defined by a sequence of steps from starting point to a
        terminal state.
        """
        done = False
        if env is None:
            env = self.env_

        episode_rewards = 0.
        state = env.reset()
        old_Qtable = self.Qsa.copy()
        steps = 0
        while not done and steps <= self.max_episode_len:
            action = self.move(state)
            state_p, reward, done, _ = env.step(action)
            episode_rewards += reward
            self.Qsa[state, action] = self._Q_update_func(state, action, reward, state_p, done)
            self.transition_freq[action, state, state_p] += 1
            state = state_p
            steps += 1
        self.epsilon *= self.epsilon_shrink
        self.alpha *= self.alpha_shrink
        episode_abs_mean = np.abs(old_Qtable - self.Qsa).mean()
        self._abs_update_mean.append(episode_abs_mean)
        self._episode_time.append(time.time() - self._creation_time)
        self._policy_diff.append((old_Qtable.argmax(axis=1) != self.Qsa.argmax(axis=1)).sum())
        self._rewards.append(episode_rewards)





def shape_map(amap):
    """Create an nxn matrix from given environment map."""
    edge_length = int(np.sqrt(len(amap)))
    amap_square = np.array(list(amap)).reshape(edge_length, edge_length)
    return amap_square

# gmc = GMCAgent(lr=0.01, init_epsilon=0.8, max_steps=800, gamma=0.99, threshold=0.05,
#                transformer=None)
# gmc.fit(env, render_train=False, verbose=True, episodes=500)
class GMCAgent:
    def __init__(self, lr=0.01, gamma=0.99, init_epsilon=0.995, threshold=0.1,
                 max_steps=3000, transformer=None, decay_lr=True):
        self.alpha = lr
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.thresh = threshold
        self.w = None
        self.history = {'rewards': [], 'weights':[], 'gradients':[]}
        self.max_steps = max_steps
        if transformer is None:
            transformer = DummyTransformer()
        self.transformer = transformer
        self.transformer.fit(np.random.randn(1, 8))
        self.decay_lr = decay_lr

    def transform(self, s):
        return self.transformer.transform(s.reshape(-1,8))

    def Q(self, s, a):
        return self.transform(s.reshape(1,-1)) @ self.w[:, a == np.arange(4)]

    def V(self, s):
        return self.transform(s.reshape(1,-1)) @ self.w

    def policy(self, s):
        return (self.transform(s.reshape(1,-1)) @ self.w).argmax()

    def move(self, s):
        epsilon = max(self.epsilon, self.thresh)
        if np.random.random() < epsilon:
    #         if np.random.random() > 0.5:
    #             return 0
            return np.random.randint(self.w.shape[1])
        else:
            return self.policy(s)

    def _init_weights(self, env):
        self.w = np.zeros((self.transform(np.random.randn(1, env.observation_space.shape[0])).shape[1], env.action_space.n))

    def fit(self, env, episodes=1000, verbose=1, render_train=False):
        if self.w is None:
            self._init_weights(env)
        for episode in range(episodes):
            if verbose:
                print('Episode:', episode)
            self.fit_episode(env, verbose, render_train)
            self.epsilon = self.epsilon * 0.99
            # if self.decay_lr:
            #     self.alpha *= 0.9995

    def update_weights(self, S, A, R):
        for i in range(len(S) - 1):
            gradient = \
            (R[i]*(A[i] == np.arange(4))
            + self.gamma * self.V(S[i + 1])
            - self.V(S[i])).reshape(1,-1) \
            * self.transform(S[i]).reshape(-1,1)
            # gradient = np.clip(gradient, -0.5, 0.5)
            self.w += self.alpha * gradient
            self.history['gradients'].append(np.abs(gradient).sum())

    def fit_episode(self, env, verbose, render_train):
        S = [env.reset()]
        a = self.move(S[0])
        A = [a]
        R = []
        done = False
        i = 0
        while not done and i < self.max_steps:
            St, Rt, done, _ = env.step(a)
            S.append(St)
            R.append(Rt)
            a = self.move(St)
            if render_train:
                env.render()
            A.append(a)
            i += 1
        R.append(0)
        self.update_weights(S, A, R)
        sumr = sum(R)
        self.history['rewards'].append(sumr)
        if verbose:
            print('Total reward:', sumr)

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = w

    def land(self, env, render=True, verbose=True):
        eps, thr = self.epsilon, self.thresh
        self.epsilon, self.thresh = 0., 0.
        done = False

        s = env.reset()
        S = []
        A = []
        if verbose:
            print('Initial state:')
            print(*s.round(2))
        i = 0
        rew = 0
        while not done and i < 2000:
            if verbose:
                print('State:')
                print(*s.round(2))
            a = self.move(s)
            S.append(s)
            A.append(a)
            if verbose:
                print('Action taken:', a)
            s, r, done, _ = env.step(a)
            if verbose:
                print('Reward:', round(r, 2))
            rew += r
            if render:
                env.render()
            i += 1
        self.epsilon, self.thresh= eps, thr
        return rew, r


# class DQNAgent(GMCAgent):
#     def __init__(self, lr=0.01, gamma=0.99, init_epsilon=0.9, threshold=0.1,
#                  max_steps=3000, transformer=None):
#         GMCAgent.__init__(self, lr=lr, gamma=gamma, init_epsilon=init_epsilon, threshold=threshold,
#                           max_steps=max_steps, transformer=transformer)
#         self.decision = None
#         self.memory = deque(maxlen=100000)
#
#     def Q(self, s, a):
#         return self.decision.predict(self.transform(s))[a]
#
#     def V(self, s):
#         return self.decision.predict(self.transform(s))
#
#     def policy(self, s):
#         return self.decision.predict(self.transform(s)).argmax()
#
#     def move(self, s):
#         epsilon = max(self.epsilon, self.thresh)
#         if np.random.random() < epsilon:
#             return np.random.randint(self.w.shape[1])
#         else:
#             return self.policy(s)
#
#     def _init_weights(self, env):
#         input_shape = self.transform(np.random.randn(1, env.observation_space.shape[0])).shape[1]
#         output_shape = env.action_space.n
#         self.w = np.random.randn(input_shape, output_shape)
#         model = Sequential()
#         model.add(Dense(150, input_shape=(input_shape,), activation='relu'))
#         model.add(Dense(100, activation='relu'))
#         # model.add(Dense(100, activation='relu'))
#         model.add(Dense(output_shape, activation=None))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
#         self.decision = model
#         model.fit(self.transform(np.random.randn(1, 8)), np.random.randn(1, 4), verbose=0)
#
#     def fit(self, env, episodes=1000, verbose=1, render_train=False):
#         if self.decision is None:
#             self._init_weights(env)
#         for episode in range(episodes):
#             if verbose:
#                 print('Episode:', episode)
#             self.fit_episode(env, verbose, render_train)
#             self.epsilon = self.epsilon * 0.98
#             if np.mean(self.history['rewards'][-100:]) > 200:
#                 break
#
#     def update_weights(self):
#         # import pdb; pdb.set_trace()
#         if not self.memory:
#             return
#         sars = random.sample(self.memory, min(32, len(self.memory)))
#
#         S = np.vstack([np.array(episode[0]) for episode in sars])
#         A = np.concatenate([episode[1] for episode in sars])
#         R = np.concatenate([episode[2] for episode in sars])
#         St = np.vstack([np.array(episode[3]) for episode in sars])
#
#         s_ind = np.random.choice(range(S.shape[0]), size=64)
#
#         S = S[s_ind]
#         A = A[s_ind]
#         R = R[s_ind]
#         St = St[s_ind]
#
#         rews = (np.reshape(R, (-1,1))*(np.reshape(A, (-1, 1)) == np.arange(4)))
#         y = self.gamma * self.V(np.array(St))
#         y[rews != 0] = rews[rews != 0]
#         X = self.transform(np.array(S))
#         self.decision.fit(X, y, verbose=0)
#
#     # def update_weights(self, S, A, R):
#     #     y = self.gamma * self.V(np.array(S[1:]))
#     #     rews = (np.reshape(R[:-1], (-1,1))*(np.reshape(A[:-1], (-1, 1)) == np.arange(4)))
#     #     y[rews != 0] = rews[rews != 0]
#     #     X = self.transform(np.array(S[:-1]))
#     #
#     #     self.decision.fit(X, y, verbose=0)
#
#
#         # tf = self.V(S[:-1])
#         # for i in range(1,8):
#         #     y = (np.reshape(R[i:], (-1,1))*(np.reshape(A[i:], (-1, 1)) == np.arange(4))) * self.gamma ** (i - 1) + (self.gamma ** i) * self.V(np.array(S[i:]))
#         #     X = self.transform(np.array(S[:-i]))
#         #     # import pdb; pdb.set_trace()
#         #     self.decision.fit(X, y, verbose=0)
#
#     def fit_episode(self, env, verbose, render_train):
#         S = [env.reset()]
#         a = self.move(S[0])
#         A = [a]
#         R = []
#         done = False
#         i = 0
#         while not done and i <= self.max_steps:
#             St, Rt, done, _ = env.step(a)
#             S.append(St)
#             R.append(Rt)
#             a = self.move(St)
#             if render_train:
#                 env.render()
#             A.append(a)
#             i += 1
#             # for k in range(1):
#             self.update_weights()
#         # if not done:
#         #     R[-1] = -600
#         R.append(R[-1])
#         # self.update_weights(S, A, R)
#         # import pdb; pdb.set_trace()
#         self.memory.append([S[:-1], A[:-1], R[:-1], S[1:]])
#         sumr = sum(R)
#         self.history['rewards'].append(sumr)
#         if verbose:
#             print('Total reward:', sumr)
#
#     def get_weights(self):
#         return self.decision.get_weights()
#
#     def set_weights(self, w):
#         self.decision.set_weights(w)


class SemiGradientAgent(GMCAgent):
    def update_weights(self, s, a, r, st):
        gradient = ((r)*(a == np.arange(4)) + self.gamma * self.V(st) - self.V(s)).reshape(1,-1) * self.transform(s).reshape(-1,1)
        gradient = np.clip(gradient, -70, 70)
        self.w += self.alpha * gradient
        self.history['gradients'].append(np.abs(gradient).sum())

    def fit_episode(self, env, verbose, render_train):
        s = env.reset()
        done = False
        i = 0
        total_rewards = 0
        while not done and i < self.max_steps:
            a = self.policy(s)
            st, r, done, _ = env.step(a)
            total_rewards += r
            if render_train:
                env.render()
            self.update_weights(s, a, r, st)
            s = st
            i += 1
        if verbose:
            print('Total rewards:', total_rewards)
        self.history['rewards'].append(total_rewards)


class NStepSemiGradientAgent(GMCAgent):
    def __init__(self, n, lr=0.01, gamma=0.99, init_epsilon=0.9, threshold=0.1,
                 max_steps=3000):
        GMCAgent.__init__(self, lr, gamma, init_epsilon, threshold, max_steps)
        self.n = n

    def update_weights(self, s, a, r, st):
        pass

    # def _init_weights(self, env):
    #     self.w = np.random.randn(8, 4)

    def fit_episode(self, env, verbose, render_train):
        s = env.reset()
        S = [s]
        R = []
        A = []
        T = self.max_steps
        t = 0

        while True:
            if t < T:
                a = self.policy(s)
                s, r, done, _ = env.step(a)
                if render_train:
                    env.render()
                S.append(s)
                R.append(r)
                if done:
                    T = t + 1
            tau = t - self.n - 1
            if tau >= 0:
                G = sum([self.gamma ** (i - tau - 1) * R[i]
                         for i in range(tau + 1, min(tau + self.n, T))])
                if tau + self.n < T:
                    # import pdb; pdb.set_trace()
                    G = (a == np.arange(4)) * G + self.gamma ** self.n * self.V(S[tau + self.n])
                    gradient = (G - self.V(S[tau])).reshape(1, -1) * S[tau].reshape(-1, 1)
                    self.w += self.alpha * gradient
            if tau == (T - 1):
                break
            t += 1
        sumr = sum(R)
        if verbose:
            print("Total rewards:", sumr)
        self.history['rewards'].append(sumr)




class EpisodicSemiGradientSarsa(GMCAgent):
    def update_weights(self, s, a, r, st):
        self.w += self.alpha \
            * (r*(a == np.arange(4)) + self.gamma * self.V(st) - self.V(s)).reshape(1,-1) \
            * self.transform(s).reshape(-1,1)

    def fit_episode(self, env, verbose, render_train):
        s = env.reset()
        done = False
        i = 0
        total_rewards = 0
        while not done and i < self.max_steps:
            a = self.policy(s)
            st, r, done, _ = env.step(a)
            if done:
                self.w += self.alpha * ((r*(a == np.arange(4)) - self.V(s))) * self.transform(s).reshape(-1,1)
                break

            total_rewards += r
            if render_train:
                env.render()
            self.update_weights(s, a, r, st)
            s = st
            i += 1
        if verbose:
            print('Total rewards:', total_rewards)
        self.history['rewards'].append(total_rewards)
