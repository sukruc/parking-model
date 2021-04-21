import model
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from typing import Callable


def timit(func):
    """Timer function. For profiling purposes."""
    def wrapper(*args, **kwargs):
        then = time.time()
        res = func(*args, **kwargs)
        now = time.time()
        print(func.__name__, 'executed in', round(now-then, 4), 'seconds.')
        return res
    return wrapper


def make_calc(conf: model.StreetConfig) -> Callable:
    """Create an expected cost calculator function from given configuration."""
    ne = pd.Series(conf.parks.empty_marginal()) * pd.Series(conf.parks.cost)
    empty_expected = ne.sum()
    p_occ = conf.parks.p_occupied()
    q_occ = 1. - p_occ

    def expected_cost(N: int) -> float:
        """Calculate expected cost at given step N."""
        if N < 1:
            raise ValueError('N < 1')

        def get_cost(k):
            if N == k:
                return q_occ * (empty_expected + k * conf.cost_walk_unit) \
                    + p_occ * conf.cost_garage
            return q_occ * (empty_expected + k * conf.cost_walk_unit) \
                + p_occ * get_cost(k + 1)

        return get_cost(1)
    return expected_cost


class Action(int):
    pass


class Agent(ABC):
    """Base class for Agents."""

    def __init__(self, env_conf: model.StreetConfig):
        self.state: model.ParkingOnStreet = None
        self.spaces_left: int = None
        self.done: bool = None
        self.conf: model.StreetConfig = env_conf

    def get_current_cost(self) -> float:
        """Get cost of current parking.

        Cost of walking is included.
        """
        return self.state.cost + self.conf.cost_walk_unit * self.spaces_left

    def observe(self, state: model.Parking, spaces_left: int, done: bool):
        """Observe current state."""
        self.state = state
        self.spaces_left = spaces_left
        self.done = done

    @abstractmethod
    def decide(self) -> Action:
        pass


class RandomAgent(Agent):
    """Random agent. Makes random decisions."""

    def decide(self):
        return np.random.choice([0, 1])


class ThresholdAgent(Agent):
    """Parks in the first empty space after threshold is reached."""

    def __init__(self, env_conf, threshold: int):
        self.threshold = threshold
        super().__init__(env_conf)

    def decide(self):
        """Decide 1 if threshold is reached, else 0."""
        if self.spaces_left <= self.threshold:
            return 1
        return 0


class GreenThresholdAgent(ThresholdAgent):
    """Attempts to park in spaces with specified name after exceeding
    threshold.

    If no name is specified, defaults to parking space name with minimum cost.
    """

    def __init__(self, *args, **kwargs):
        min_name = kwargs.get('min_name')
        if 'min_name' in kwargs:
            del kwargs['min_name']
        super().__init__(*args, **kwargs)
        if min_name is None:
            min_name = min(self.conf.parks, key=lambda x: x.cost).name
        self.min_name = min_name

    def decide(self):
        """Decide 1 if threshold is reached and the parking is cheap,
        else 0."""
        if self.spaces_left <= self.threshold:
            if self.state.name == self.min_name:
                return 1
        return 0


class ExpectedAgent(Agent):
    """Computes expected cost at each step and attempts to park if expected
    cost is higher than current cost."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculator: Callable = make_calc(self.conf)

    def decide(self):
        if not hasattr(self.state, 'cost'):
            return 0
        expected = self.calculator(self.spaces_left)
        current = self.get_current_cost()
        if expected >= current:
            return 1
        else:
            return 0
