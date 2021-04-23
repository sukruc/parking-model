# Street Parking Model

Agents try to find optimal parking spot to minimize cost.

## Usage

### Configuration

Decide scenario parameters. Given below is an example set of parameters:
```python
street_length = 20
park_probas = {
     0: {
        'pexist': 0.25,
        'poccupied': 0.2,
        },
     4: {
        'pexist': 0.25,
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
    4: 7.,
    1: 5.,
    2: 3.,
    3: 1.,
}
walk_cost = 0.1
drive_cost = 0.1
katoto_cost = 10.0
random_accident_proba = 1e-6
random_accident_cost = 90.0
allow_goback = True
pkatoto_full = 0.4
```

See **Parameters** section for parameter details.

### Create `StreetParking` instance

```python
from streetv2 import StreetParking
env = StreetParking(
  street_length=street_length,
  park_probas=park_probas,
  park_costs=park_costs,
  walk_cost=walk_cost,
  drive_cost=drive_cost,
  katoto_cost=katoto_cost,
  random_accident_proba=random_accident_proba,
  random_accident_cost=random_accident_cost,
  pkatoto_full=pkatoto_full,
  allow_goback=allow_goback,
  )
```

### API

Street parking has an API nearly identical to OpenAI environments.

```python
env.reset()
state, reward, done, prob = env.step(2)
```

### Transition and Reward Matrix

Transition and reward matrices for the environment can be accessed as follows:

```python
>>> env.transition
>>> env.rewards
```

## Parameters

| Parameter | Type |Explanation     |
| :------------- | :------------- |:------------- |
| `street_length`   |  `int`  | Length of street in terms of parking spots       |
| `park_probas`    | `Dict[int: Dict[str: float]]`  | Length of street in terms of parking spots       |
| `park_costs`    | `Dict[int: float]`  | Cost of parking for each unique spot      |
| `walk_cost`    | `float`  | Cost of walking per block after parking      |
| `drive_cost`    | `float`  | Cost of driving per block      |
| `katoto_cost`    | `float`  | Cost of parking at the end of the street      |
| `random_accident_proba`    | `float`  | Probability of a costly accident      |
| `random_accident_cost`    | `float`  | Cost of a random accident      |
| `pkatoto_full`    | `float`  | Probability of parking lot being not empty at the end of the street (valid if only `allow_goback=True`)      |
| `allow_goback`    | `bool`  | Whether to allow going back to beginning of the street if office parking lot is full (use `pkatoto_full` to control occupancy of office parking lot)      |

## Examples

An example solution with Value Iteration, Policy Iteration and Q-Learning is
provided in `example_usage.py`.
