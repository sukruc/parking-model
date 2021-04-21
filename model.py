import numpy as np
import pandas as pd
from attr import dataclass
from typing import List, Dict, Tuple


@dataclass
class Parking:
    """Base Parking object.

    Attributes:
    --------------
    cost : float, cost of parking into this space
    p_occupied : float, probability of park space being occupied
    p_exist : float, probability of any park space being this type
    occupied : bool, indicates if parking space is occupied
    name: Union[str, None], a unique identifer for parking space class
    """
    cost: float
    p_occupied: float
    p_exist: float = None
    name: str = None
    occupied: bool = None


class ParkingOnStreet(Parking):
    """Actualized Parking space.

    occupied attribute is populated based on p_occupied value.
    """

    def __init__(self, cost, p_occupied, p_exist=None, name=None):
        super().__init__(cost, p_occupied, p_exist, name)
        self.occupied = np.random.choice(
            [True, False],
            p=[self.p_occupied, 1. - self.p_occupied]
        )

    def spaces_left():
        doc = "The spaces_left property."

        def fget(self):
            return self._spaces_left

        def fset(self, value):
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid value for spaces left: %s" % value)
            self._spaces_left = value

        def fdel(self):
            del self._spaces_left
        return locals()
    spaces_left = property(**spaces_left())


@dataclass
class ParkConf(dict):
    """Park Configuration object."""
    cost: float
    p_occupied: float
    p_exist: float
    name: str

    def __getitem__(self, key):
        return self.__dict__[key]


class StreetConfig(dict):
    """Street configuration object."""

    def __init__(self, parks: List[ParkConf], cost_garage: float,
                 cost_walk_unit: float):
        super().__init__(parks=parks, cost_garage=cost_garage,
                         cost_walk_unit=cost_walk_unit)
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

    def __getattr__(self, attr):
        return {park.name: getattr(park, attr) for park in self.parks}

    def to_frame(self) -> pd.DataFrame:
        """Export park configurations to a dataframe."""
        return pd.DataFrame([park.__dict__ for park in self.parks])

    def empty_marginal(self) -> Dict[str, float]:
        """Calculate marginal probability of each parking space
        given it's empty."""
        marg = pd.concat([self.to_frame().assign(o=1),
                          self.to_frame().assign(
                              p_occupied=lambda x: 1 - x['p_occupied'], o=0)
                          ]) \
            .assign(p=lambda x: x['p_occupied'] * x['p_exist']) \
            .assign(e_cost=lambda x: x['p'] * x['cost']) \
            .loc[lambda x: x['o'].eq(0)] \
            .assign(marginal=lambda x: x['p'] / x['p'].sum()) \
            .set_index('name').loc[self.name]['marginal'].to_dict()
        return marg

    def p_occupied(self) -> float:
        """Calculate marginal probability of being occupied for a
        parking space."""
        return pd.concat([self.to_frame().assign(o=1),
                          self.to_frame().assign(
                              p_occupied=lambda x: 1 - x['p_occupied'], o=0)
                          ]) \
            .assign(p=lambda x: x['p_occupied'] * x['p_exist']) \
            .assign(e_cost=lambda x: x['p'] * x['cost']) \
            .loc[lambda x: x['o'].eq(1)]['p'].sum()


class Street:
    """Street object."""

    def __init__(self, length: int, parks: ParkList, cost_garage: float,
                 cost_walk_unit: float):
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
        if not isinstance(parks, ParkList):
            parks = ParkList(parks)
        self.parks = parks
        self.last_reward = None
        self._generate_parks()

    @property
    def done(self):
        return self.state == self.length or self.parked

    def _generate_parks(self):
        """Generate parking spaces and determine occupation."""
        for i in range(self.length):
            parking_conf = np.random.choice(
                self.parks, p=[park.p_exist for park in self.parks])
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
        return self.spaces_left * self.cost_walk_unit \
            + self.current_parking.cost

    @property
    def best_possible_cost(self):
        best_parking = min([self.street[i].cost + (self.length - i)
                            * self.cost_walk_unit for i in range(self.length)])
        return min(best_parking, self.cost_garage)

    def step(self, action) -> Tuple[ParkingOnStreet, float, bool, int]:
        """Take an action.

        Arguments:
        -------------------
        action : int, 0 or 1
            0 : Skip current parking space and go forward.
            1 : Park into current parking space. If space is occupied,
                this action defaults to 0.

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
        if action == 1:
            self.parked = True
        return (self.current_parking, self.reward, self.done, self.spaces_left)


class TransitStreet(Street):
    def step(self, action) -> Tuple[ParkingOnStreet, float, bool, int]:
        park, reward, done, spaces_left = super().step(action)
        if park is not None:
            park.spaces_left = spaces_left
        return park, reward, done, spaces_left









