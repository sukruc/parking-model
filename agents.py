from model import Parking, ParkConf, StreetConfig
import numpy as np

class Agent:
    def __init__(self, env_conf: StreetConfig):
        self.state = None
        self.spaces_left = None
        self.done = None
        self.conf = env_conf

    def observe(self, state: Parking, spaces_left: int, done: bool):
        self.state = state
        self.spaces_left = spaces_left
        self.done = done

    def decide(self):
        pass


class RandomAgent(Agent):
    def decide(self):
        return np.random.choice([0, 1])


class ThresholdAgent(Agent):
    def __init__(self, env_conf, threshold: int):
        self.threshold = threshold
        super().__init__(env_conf)

    def decide(self):
        if self.spaces_left <= self.threshold:
            return 1
        return 0


class GreenThresholdAgent(ThresholdAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_cost = min(map(lambda x: x['cost'], self.conf.parks))

    def decide(self):
        if self.spaces_left <= self.threshold:
            if self.state.cost == self.min_cost:
                return 1
        return 0
