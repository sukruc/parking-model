# Street Parking Model

Agents try to find optimal parking spot to minimize cost.

## Usage

Launch `main.py` with arguments to run simulation.

Arguments:
```shell
--agent: Agent class name
--nsim : number of repeats, default is 30000
--len  : lengh of street, default is 20
```
Supply optional arguments if required by agent.

```shell
main.py --agent ThresholdAgent --threshold 8 --nsim 1000 --len 20
```

## Model

Configuration and model specifications are given in `model.py`.

## Agents

Agent requirements and sample agents are given in `agents.py`.

## Configuration

A sample configuration is given in `conf.json`.
Configuration object must include the following keys:
- `parks` : a list of park configurations. Must contain
  - `cost` float,
  - `p_occupied` float between 0 and 1,
  - `p_exist` float between 0 and 1, and must sum up to 1 across all park classes,
  - `name` string, must be unique for each parking class
- `cost_garage` : float, cost of parking if all spaces are exhausted,
- `cost_walk_unit`: float, cost of walking per parking space

See `model.StreetConfig` for details and `model.ParkConf` for parking space configuration details.
