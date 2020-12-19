import json
from model import Street
from agents import ThresholdAgent, RandomAgent
import numpy as np
import multiprocessing
import sys

if __name__ == '__main__':
    with open('conf.json') as f:
        conf = json.load(f)

    N = 50
    THRESHOLD = int(sys.argv[1])
    N_SIM = 20000
    s = Street(N, **conf)
    agent = ThresholdAgent(THRESHOLD)
    # agent = RandomAgent()
    rewards = []

    for _ in range(N_SIM):

        state, reward, done = s.reset()
        agent.observe(state, s.spaces_left, done)
        i = 0
        while not done and i < N:
            action = agent.decide()
            state, reward, done, spaces_left = s.step(action)
            agent.observe(state, spaces_left, done)
            i += 1
        rewards.append(-reward)

    print(np.mean(rewards))
