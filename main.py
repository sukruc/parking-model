"""Run simulation with multiprocessing.

Arguments:
--------------
--agent : Agent class name
--nsim  : number of simulations to run, default 30000
--len   : length of street in simulation, default 20
"""

import json
from model import Street, StreetConfig
import numpy as np
import multiprocessing
import sys
import agents
import time

kwargs = {}
for i in range(1, len(sys.argv[1:]), 2):
    arg = sys.argv[i + 1]
    if arg.isnumeric():
        arg = float(arg)
    kwargs[sys.argv[i].lstrip('-')] = arg

with open('conf.json') as f:
    conf = StreetConfig(**json.load(f))

N = 20
N_SIM = 30000

Agent = getattr(agents, kwargs['agent'])
del kwargs['agent']

if 'nsim' in kwargs:
    N_SIM = int(kwargs['nsim'])
    del kwargs['nsim']

if 'len' in kwargs:
    N = int(kwargs['len'])
    del kwargs['len']


s = Street(N, **conf)
agent = Agent(conf, **kwargs)


def run(k):

    state, reward, done = s.reset()
    agent.observe(state, s.spaces_left, done)
    i = 0
    while not done and i < N:
        try:
            action = agent.decide()
        except Exception as e:
            print(i)
            raise e
        state, reward, done, spaces_left = s.step(action)
        agent.observe(state, spaces_left, done)
        i += 1
    return -reward, -s.best_possible_cost


if __name__ == '__main__':

    then = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        rewards, bests = list(zip(*p.map(run, [i for i in range(N_SIM)])))
    now = time.time()

    print("Average cost:", np.mean(rewards))
    print("Average best possible:", np.mean(bests))
    print(N_SIM, "simulations completed in", round(now - then, 3), "seconds.")
