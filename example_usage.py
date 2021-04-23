from streetv2 import StreetParking
from rlagent import QLAgent
from iterators import iterate_value
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':
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

    print("Value Iteration")
    vi, arr = iterate_value(env.transition, env.rewards, discount=1.)
    plt.plot(arr)
    plt.xlabel("Iterations")
    plt.ylabel("Max absolute value change")
    plt.savefig('example-run-output/example-run-vi-convergence.png')
    plt.close()

    print('Q-Learning')
    agent = QLAgent(gamma=1., epsilon=0.99, alpha=0.95, epsilon_shrink=0.9999, alpha_shrink=0.9999)
    agent.fit(env, iters=90000)

    plt.plot(agent._abs_update_mean)
    plt.xlabel("Iterations")
    plt.ylabel("Max absolute Q-function change")
    plt.savefig('example-run-output/example-run-ql-convergence.png')
    plt.close()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].imshow(np.array(vi.V[:-3]).reshape(env.street_length, env._num_park_states).T, vmin=-20, vmax=5)
    ax[0].set_title('Value Iteration')
    ax[1].imshow(agent.Qsa.max(axis=1)[:-3].reshape(env.street_length, env._num_park_states).T, vmin=-20, vmax=5)
    ax[1].set_title('Q-Learning')
    ax[1].set_xlabel('')
    fig.suptitle("Value function for states")
    plt.savefig('example-run-output/example-run-values.png')
