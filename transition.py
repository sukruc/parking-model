import dataclasses
from dataclasses import asdict, astuple
from typing import List
from functools import lru_cache
import sys
import numpy as np

GARAGE_COST = 10.0
WALKING_COST = 0.1


@dataclasses.dataclass
class State:
    name: str
    cost: float
    p_exist: float
    occupied: bool

    def p_occupied():
        doc = "The p_occupied property."

        def fget(self):
            return self._p_occupied

        def fset(self, value):
            if not 0 <= value <= 1:
                raise ValueError(
                    "Invalid probabilty of being occupied: %s" % value)
            self._p_occupied = value

        def fdel(self):
            del self._p_occupied
        return locals()

    p_occupied: float = property(**p_occupied())

    def spaces_left():
        doc = "The spaces_left property."
        def fget(self):
            return self._spaces_left
        def fset(self, value):
            if value < 0:
                raise ValueError("Spaces left must be >= 0.")
            if not isinstance(value, int):
                raise TypeError("Spaces left must be an integer.")

            self._spaces_left = value
        def fdel(self):
            del self._spaces_left
        return locals()
    spaces_left: int = property(**spaces_left())

    # spaces_left: int = 1000

    @property
    def transition(self):
        p = {True: self.p_occupied, False: 1. -
             self.p_occupied}[self.occupied]
        return self.p_exist * p

    def occupy(self):
        self.occupied = True
        return self

    def unoccupy(self):
        self.occupied = False
        return self

    def copy(self):
        return self.__class__(**asdict(self))

    def set_spaces_left(self, value):
        self.spaces_left = value
        return self


@dataclasses.dataclass
class StateConfig:
    def states():
        doc = "The states property."

        def fget(self):
            return self._states

        def fset(self, value):
            sums = sum(state.p_exist for state in value)
            if not abs(sums - 1.0) < 1e-6:
                raise ValueError(
                    "Invalid p_exist configuration: sum of existence probabilities %s" % sums)
            self._states = value

        def fdel(self):
            del self._states
        return locals()
    states: List[State] = property(**states())

@dataclasses.dataclass
class Street:
    config: StateConfig
    spaces_left: int

    def get_parking(self):
        if not self.spaces_left:
            parking = self.config.states[0].copy().unoccupy()
            parking.cost = 10.0
            return parking
        parking = np.random.choice(self.config.states, p=[state.p_exist for state in self.config.states]).copy()
        parking.set_spaces_left(self.spaces_left)
        if np.random.random() < parking.p_occupied:
            parking.occupy()
        else:
            parking.unoccupy()
        self.spaces_left -= 1
        return parking


class Action(int):
    pass


def reward(state: State, action: Action):
    if not state.spaces_left:
        return -GARAGE_COST
    if state.occupied:
        action = 0
    if action == 1:
        return -(state.cost + (state.spaces_left * WALKING_COST))
    return 0.0


def transition(state: State, action: Action, nextstate: State):
    if state.spaces_left - 1 != nextstate.spaces_left:
        return 0.
    if state.spaces_left == 0:
        return 1.0
    if state.occupied:
        action = 0
    if action == 0:
        return nextstate.transition
    if action == 1:
        return 0.0


GAMMA = 0.999
ACTIONS = [Action(0), Action(1)]
config = StateConfig(
    [
        State('red', 5.0, 0.6, True, 0.4, 1000),
        State('green', 3.0, 0.4, True, 0.6, 1000),
    ])

@lru_cache(maxsize=800000)
def V(state):
    state = State(*state)
    if not state.spaces_left:
        nstates = []
    else:
        nstates = [s.copy().unoccupy().set_spaces_left(state.spaces_left - 1) for s in config.states] + [s.copy().occupy().set_spaces_left(state.spaces_left - 1) for s in config.states]
    values = [(reward(state, action) + GAMMA * sum([transition(state, action, ns) * V(astuple(ns))[0] for ns in nstates]), action) for action in ACTIONS]

    # print(values)
    return max(values)

if __name__ == '__main__':
    # sys.setrecursionlimit(3000)
    LENGTH = int(sys.argv[1])
    for i in range(LENGTH + 10):
        V(astuple(State('red', 5.0, 0.6, False, 0.4, i)))
        V(astuple(State('green', 3.0, 0.4, False, 0.6, i)))

    rewards = []

    for _ in range(3000):
        street = Street(config, LENGTH)
        done = False
        for i in range(LENGTH, 0, -1):
            parking = street.get_parking()
            # if parking is None:
            #     break
            # print(parking)
            value, action = V(astuple(parking))
            if action and not parking.occupied:
                # print(value)
                # print(action)
                # print(i)
                done = True
                # print('Parked')
                # print(_)
                break
        if done:
            rewards.append(value)
        else:
            rewards.append(-GARAGE_COST)
    # print(rewards)
    print(np.mean(rewards))
    # print(config.states[0].copy().unoccupy().set_spaces_left(3))
