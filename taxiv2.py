from gym.envs.toy_text import TaxiEnv
import gym.envs.toy_text.discrete as discrete
import numpy as np
import json

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class TaxiEnvTr(TaxiEnv):
    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        # assert 0 <= i < 5
        return reversed(out)

    def __init__(self, rowreps=1, colreps=1):
        self.desc = np.asarray(MAP, dtype='c')

        self.desc = np.concatenate([self.desc[:,:3], np.repeat(self.desc[:, 3:5], colreps, axis=1), self.desc[:,5:]], axis=1)
        self.desc = np.concatenate([self.desc[:2,:], np.repeat(self.desc[2:3, :], rowreps, axis=0), self.desc[3:,:]], axis=0)

        self.rowreps = rowreps
        num_rows = self.desc.shape[0] - 2
        num_columns = self.desc.shape[1] - 6 - (colreps - 1) * 2

        self.num_rows = num_rows
        self.num_cols = num_columns

        num_states = num_rows * num_columns * 5 * 4
        max_row = num_rows - 1
        self.locs = locs = [(0, 0), (0, 4), (max_row, 0), (max_row, 3)]
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            try:
                                self.desc[1 + row, 2 * col + 2]
                            except IndexError as e:
                                print(2 * col + 2)
                                raise e
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if (pass_idx < 4 and taxi_loc == locs[pass_idx]):
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx)
                            P[state][action].append(
                                (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    @property
    def transition(self):
        tr = np.zeros((self.nA, self.nS, self.nS))

        for a in range(6):
            for s in range(self.nS):
                self.s = s
                sp, reward, done, proba = self.step(a)
                tr[a, s, sp] = proba['prob']
        self.reset()
        return tr

    @property
    def rewards(self):
        rw = np.zeros((self.nA, self.nS, self.nS))

        for a in range(6):
            for s in range(self.nS):
                self.s = s
                sp, reward, done, proba = self.step(a)
                rw[a, s, sp] = reward
        self.reset()
        return rw

    def export_config(self, filename):
        with open(filename, 'w') as f:
            json.dump({'rowreps': self.rowreps}, f)

    @classmethod
    def from_config(cls, filename):
        with open(filename) as f:
            params = json.load(f)
        return cls(**params)
