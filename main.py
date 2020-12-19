import json
from model import Street, StreetConfig
from agents import ThresholdAgent, RandomAgent, GreenThresholdAgent
import numpy as np
import multiprocessing
import sys
import argparse

if __name__ == '__main__':
    kwargs = {}
    for i in range(1, len(sys.argv[1:]), 2):
        arg = sys.argv[i + 1]
        if arg.isnumeric():
            arg = float(arg)
        kwargs[sys.argv[i].lstrip('-')] = arg

    with open('conf.json') as f:
        conf = StreetConfig(**json.load(f))

    N = 50
    N_SIM = 10000
    s = Street(N, **conf)
    agent = GreenThresholdAgent(conf, **kwargs)

    def run(k):
        # agent = RandomAgent()

        state, reward, done = s.reset()
        agent.observe(state, s.spaces_left, done)
        i = 0
        while not done and i < N:
            action = agent.decide()
            state, reward, done, spaces_left = s.step(action)
            agent.observe(state, spaces_left, done)
            i += 1
        return -reward

    with multiprocessing.Pool(processes=12) as p:
        rewards = p.map(run, [i for i in range(N_SIM)])


    print(np.mean(rewards))
