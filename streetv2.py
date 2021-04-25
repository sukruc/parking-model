import numpy as np
import pandas as pd
from attr import dataclass
# import mdptoolbox
import json
# import matplotlib.pyplot as plt


@dataclass
class StreetModel:
    street_length: int
    park_probas: dict
    park_costs: dict
    walk_cost: float = 0.03
    drive_cost: float = 0.05
    katoto_cost: float = 10.0
    random_accident_proba: float = 0.003
    random_accident_cost: float = 90.0
    allow_goback: bool = True
    pkatoto_full: float = 0.0


class StreetParking(StreetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.park_probas = ParkProbas(self.park_probas)
        self.park_costs = ParkCosts(self.park_costs)
        if sorted(self.park_probas) != sorted(self.park_costs):
            raise ValueError("Parking probability and cost dictionary keys do not match.")
        self._num_states = self.street_length * len(self.park_costs) * 2 + 3
        self._num_park_states = len(self.park_costs) * 2
        self._num_actions = 2
        self.state = None
        self._state_length = None
        self.done = None
        self.reset()
        self._tr = self.transition
        self._rew = self.rewards

    def reset(self):
        self.done = False
        # parks, probas = np.array([[k, v['pexist']] for k, v in sorted(self.park_probas.items())]).T.tolist()
        # self.state = np.random.choice()
        # self.state = np.random.randint(0, self._num_park_states + 1)
        self.state = self._select_random_parking_state(init=True)
        self._state_length = 0
        return self.state

    def _select_random_parking_state(self, init=False):
        if init:
            random_accident_proba = 0.0
        else:
            random_accident_proba = self.random_accident_proba
        one_row = []
        for park_type, probas in self.park_probas.items():
            pexist, poccupied = probas['pexist'], probas['poccupied']
            # one_row += list(probas)
            one_row += [
                (1. - random_accident_proba) * pexist * poccupied,
                (1. - random_accident_proba) * pexist * (1. - poccupied),
                ]

        return np.random.choice(np.arange(self._num_park_states), p=one_row)

    def _move_step(self):
        disaster = np.random.random() <= self.random_accident_proba
        if disaster:
            self.done = True
            reward = -self.random_accident_cost
            self.state = self._num_states - 1
        elif self._state_length == (self.street_length - 1):
            if not self.allow_goback:
                self.done = True
                reward = -self.katoto_cost - self.drive_cost
                self.state = self._num_states - 3
            else:
                if np.random.random() < self.pkatoto_full:
                    self.done = False
                    reward = -self.drive_cost * self.street_length
                    self.state = self.reset()
                else:
                    self.done = True
                    reward = -self.katoto_cost - self.drive_cost
                    self.state = self._num_states - 3
        else:
            self._state_length += 1
            reward = -self.drive_cost
            self.state = (self._state_length * self._num_park_states) + self._select_random_parking_state()
        return self.state, reward, self.done

    def step(self, action):
        occupied = not bool((self.state % self._num_states) % 2)
        if self.done:
            return self.state, 0.0, self.done
        if action == 0:
            self.state, reward, self.done = self._move_step()
        elif action == 1:
            if occupied:
                self.state, reward, self.done = self._move_step()
            else:
                park_cost = self.park_costs[((self.state % self._num_park_states) - 0) // 2]
                walk_cost = (self.street_length - self.state // self._num_park_states) * self.walk_cost
                self.state = self._num_states - 2
                self.done = True
                reward = -park_cost - walk_cost
        return self.state, reward, self.done, None

    @property
    def nS(self):
        return self._num_states

    @property
    def nA(self):
        return self._num_actions

    @property
    def transition(self):
        trans_move = np.zeros((self._num_states, self._num_states))
        trans_block = np.ones((self._num_park_states, self._num_park_states))

        one_row = []
        for park_type, probas in self.park_probas.items():
            pexist, poccupied = probas['pexist'], probas['poccupied']
            # one_row += list(probas)
            one_row += [
                (1. - self.random_accident_proba) * pexist * poccupied,
                (1. - self.random_accident_proba) * pexist * (1. - poccupied),
                ]

        trans_block *= np.array(one_row)

        block_len = self._num_park_states
        for i in range((self._num_states - 3) // len(self.park_costs) // 2 - 1):
            trans_move[i*block_len:(i+1)*block_len, (i+1)*block_len:(i+2)*block_len] = trans_block.copy()

        trans_move[-3-block_len:-3, -3] = [1. - self.random_accident_proba] * block_len
        trans_move[:, -1] = self.random_accident_proba
        trans_move[-3:, -3:] = np.diag([1.] * 3)

        if self.allow_goback:
            trans_move[-3-block_len:-3, :block_len] = trans_block.copy() * self.pkatoto_full
            trans_move[-3-block_len:-3, -3] *= (1. - self.pkatoto_full)

        trans_park = trans_move.copy()

        parkable_row = [0.] * self._num_states
        parkable_row[-2] = 1.0

        trans_park[np.arange(1, self._num_states, 2)] = parkable_row


        return np.array([trans_move, trans_park])

    @property
    def rewards(self):
        trans_move = np.zeros((self._num_states, self._num_states))
        trans_block = np.ones((self._num_park_states, self._num_park_states))

        one_row = [-self.drive_cost] * self._num_park_states
        trans_block *= np.array(one_row)

        block_len = self._num_park_states
        for i in range((self._num_states - 3) // len(self.park_costs) // 2 - 1):
            trans_move[i*block_len:(i+1)*block_len, (i+1)*block_len:(i+2)*block_len] = trans_block.copy()

        trans_move[-3-block_len:-3, -3] = [-self.drive_cost - self.katoto_cost] * block_len
        trans_move[:, -1] = -self.random_accident_cost
        trans_move[-3:, -3:] = np.zeros((3, 3))

        if self.allow_goback:
            trans_move[-3-block_len:-3, :block_len] = -self.drive_cost * self.street_length

        trans_park = trans_move.copy()

        parkable_row = [0.] * self._num_states
        parkable_row[-2] = 1.0
        parkable_row = np.array(parkable_row)

        for i in np.arange(1, self._num_states,2):
            park_cost = self.park_costs[((i % self._num_park_states) - 1) // 2]
            walk_cost = (self.street_length - i // self._num_park_states) * self.walk_cost
            trans_park[i] = parkable_row * (-walk_cost - park_cost)

        trans_park[-3:,-3:] = 0.0

        return np.array([trans_move, trans_park])

    def get_params(self):
        attrs = {}
        for k in self.__annotations__:
            attrs[k] = getattr(self, k)
        return attrs

    def export_config(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.get_params(), f)

    @classmethod
    def from_config(cls, filename):
        with open(filename) as f:
            attrs = json.load(f)
        return cls(**attrs)


class StreetParkingTr(StreetParking):
    def step(self, action):
        if self.done:
            return self.state, 0., self.done, {'prob':1.}
        states = np.arange(self.nS)
        probas = self._tr[action][self.state]

        state_prime = np.random.choice(states, p=probas)
        if state_prime in states[-3:]:
            done = True
        else:
            done = False

        reward = self._rew[action, self.state, state_prime]

        proba = probas[state_prime]
        self.state = state_prime
        self.done = done
        return self.state, reward, done, {'prob':proba}



class ParkDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys_to_int()
        self._validate_keys()

    def _keys_to_int(self):
        keys = list(self)
        for k in keys:
            v = self.pop(k)
            try:
                self[int(k)] = v
            except ValueError:
                raise ValueError("Parking spot names must be integer or castable to integer.")

    def _validate_keys(self):
        keys = sorted(self)
        if keys[0] != 0:
            raise ValueError("Parking spot names must start from 0")
        if keys[-1] != len(self) - 1:
            raise ValueError("Parking spot names must be starting from 0 and ordinal.")


class ParkProbas(ParkDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_dicts()
        self._validate_probas()

    def _validate_probas(self):
        sump = sum(v['pexist'] for k, v in self.items())
        if not np.allclose(sump, 1.0):
            raise ValueError(f'Parking spot probabilities do not sum up to 1: {sump}')
        if 0 in [v['pexist'] for k, v in self.items()]:
            raise ValueError("Parking spot existence probability 'pexist' must be 0>p>1")

    def _check_dicts(self):
        for k, v in self.items():
            if 'pexist' not in v or 'poccupied' not in v:
                raise ValueError(f"Keys 'pexist' and 'poccupied' must exist in each parking dict: {k}:{v}")

    @property
    def marginals(self):
        return {k: (v['pexist']*v['poccupied'], v['pexist']*(1. - v['poccupied'])) for k, v in self.items()}


class ParkCosts(ParkDict):
    pass


if __name__ == '__main__':
    street_length = 20
    park_probas = {
         0: {
            'pexist': 0.5,
            'poccupied': 0.2,
            },
         1: {
            'pexist': 0.4,
            'poccupied': 0.6,
            },
         2: {
            'pexist': 0.07,
            'poccupied': 0.9,
            },
         3: {
            'pexist': 0.03,
            'poccupied': 0.9,
            },
        }
    park_costs = {
        0: 7.,
        1: 5.,
        2: 3.,
        3: 1.,
    }
    walk_cost = 0.3
    drive_cost = 0.5
    katoto_cost = 10.0
    random_accident_proba = 1e-6
    random_accident_cost = 9000.0

    gamma = 0.99

    params = dict(
        street_length=street_length,
        park_probas=park_probas,
        park_costs=park_costs,
        walk_cost=walk_cost,
        drive_cost=drive_cost,
        katoto_cost=katoto_cost,
        random_accident_proba=random_accident_proba,
        random_accident_cost=random_accident_cost,
        )

    p = StreetParking(**params)
    p.reset()
    p.step(0)

    tr = p.transition
    rew = p.rewards

    # vi = mdptoolbox.mdp.ValueIteration(tr, rew, gamma)
    # vi.run()
    #
    # pi = mdptoolbox.mdp.PolicyIteration(tr, rew, gamma)
    # pi.run()
    #
    # fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    # ax[0].imshow(np.flip(np.array(vi.policy[:-3]).reshape(street_length, len(park_costs) * 2).T, axis=1))
    # ax[0].set_title("Policy by parking spot and distance to pay-park: VI")
    # ax[1].imshow(np.flip(np.array(pi.policy[:-3]).reshape(street_length, len(park_costs) * 2).T, axis=1))
    # ax[1].set_title("Policy by parking spot and distance to pay-park: PI")
    # ax[2].imshow(np.array(vi.policy[:-3]).reshape(street_length, len(park_costs) * 2).T - np.array(pi.policy[:-3]).reshape(street_length, len(park_costs) * 2).T)
    # ax[2].set_title("Difference")
    # ax[2].set_xlabel("Distance to parking lot")
    # ax[0].colorbar().set_label('Action')
    # plt.show()
    #
    # p.export_config('mypark.json')
    # pd.DataFrame(np.array(vi.policy[:-3]).reshape(street_length, len(park_costs) * 2))
    # pd.DataFrame(p._create_transition_matrix()).to_csv('test_trans_move.csv')
    # pd.DataFrame(p._create_rewards_matrix()).to_csv('test_reward_move.csv')
    # lp = StreetParking.from_config('mypark.json')
    #
    # vi = mdptoolbox.mdp.ValueIteration(lp.transition, lp.rewards, .99)
    # vi.run()
