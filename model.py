import numpy as np
import pandas as pd
import json
from attr import dataclass
from typing import List, Dict

@dataclass
class Parking:
    """Base Parking object.

    Attributes:
    --------------
    cost : float, cost of parking into this space
    p_occupied : float, probabilty of park space being occupied
    p_exist : float, probabilty of any park space being this type
    occupied : bool, indicates if parking space is occupied
    """
    cost: float
    p_occupied: float
    p_exist: float = None
    occupied: bool = None


class ParkingOnStreet(Parking):
    """Actualized Parking space."""
    def __init__(self, cost, p_occupied, p_exist=None):
        super().__init__(cost, p_occupied, p_exist)
        self.occupied = np.random.choice([True, False], p=[self.p_occupied, 1. - self.p_occupied])


@dataclass
class ParkConf(dict):
    """Park StreetConfiguration object."""
    cost: float
    p_occupied: float
    p_exist: float

    def __getitem__(self, key):
        return self.__dict__[key]


class StreetConfig(dict):
    """Street configuration object."""
    def __init__(self, parks: List[ParkConf], cost_garage: float, cost_walk_unit: float):
        super().__init__(parks=parks, cost_garage=cost_garage, cost_walk_unit=cost_walk_unit)
        self.parks = ParkList(parks)
        self.cost_garage = cost_garage
        self.cost_walk_unit = cost_walk_unit


class ParkList(list):
    """A list of park configurations."""
    def __init__(self, parks: List[Dict[str, float]]):
        parks = [ParkConf(**park) for park in parks]
        assert np.isclose(sum([p.p_exist for p in parks]), 1.0)
        self.parks = parks
        super().__init__(self.parks)


class Street:
    """Street object."""
    def __init__(self, length: int, parks: ParkList, cost_garage: float, cost_walk_unit: float):
        """Create a street object.

        Arguments:
        ---------------
        length: int, length of street.
        parks: ParkList, list of park configurations.
        cost_garage: float, cost of parking into garage
        cost_walk_unit: float, cost of walking per parking space
        """
        self.length = length
        self.street = []
        self.state = 0
        self.parked = False
        self.cost_garage = cost_garage
        self.cost_walk_unit = cost_walk_unit
        self.parks = ParkList(parks)
        self.last_reward = None
        self._generate_parks()

    @property
    def done(self):
        return self.state == self.length or self.parked

    def _generate_parks(self):
        """Generate parking spaces and determine occupation."""
        for i in range(self.length):
            parking_conf = np.random.choice(self.parks, p=[park.p_exist for park in self.parks])
            self.street.append(ParkingOnStreet(**parking_conf.__dict__))

    def __repr__(self):
        return repr(self.street)

    def __getitem__(self, key):
        return self.street[key]

    @property
    def spaces_left(self):
        """Return how many parking spaces left until garage."""
        return self.length - self.state

    def reset(self, regenerate=True):
        """Reset state and regenerate parking spaces."""
        self.state = 0
        self.street = []
        self.parked = False
        if regenerate:
            self._generate_parks()
        return self.state, 0., self.done

    @property
    def current_parking(self):
        """Return parking space of current state."""
        if self.parked:
            return self.street[self.state]
        if self.done:
            return None
        return self.street[self.state]

    @property
    def available_actions(self):
        return [0, 1]

    @property
    def reward(self):
        """Return reward of the current state."""
        if not self.done:
            return 0
        if not self.spaces_left:
            return self.cost_garage
        return self.spaces_left * self.cost_walk_unit + self.current_parking.cost

    def step(self, action):
        """Take an action.

        Arguments:
        -------------------
        action : int, 0 or 1
            0 : Skip current parking space and go forward.
            1 : Park into current parking space. If space is occupied, this action
                defaults to 0.

        Returns:
        -------------------
        current_parking : ParkingOnStreet(cost, p_occupied)
        reward : float, reward of the taken action
        done : bool, indicator whether the simulation ended
               this indicator is returned True when the car is parked.
        spaces_left : int, number of spaces left until garage.
        """
        if self.done:
            return None, self.last_reward, self.done
        occupied = self.current_parking.occupied
        if occupied:
            action = 0
        if action == 0:
            self.state += 1
            return self.current_parking, self.reward, self.done, self.spaces_left
        if action == 1:
            self.parked = True
            return self.current_parking, self.reward, self.done, self.spaces_left
