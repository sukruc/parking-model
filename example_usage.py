from streetv2 import StreetParking
from rlagent import QLAgent
import matplotlib.pyplot as plt
import numpy as np

street_length = 50
park_probas = {
     0: {
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
     4: {
        'pexist': 0.21,
        'poccupied': 0.2,
        },
     5: {
        'pexist': 0.03,
        'poccupied': 0.9,
        },
     6: {
        'pexist': 0.01,
        'poccupied': 0.92,
        },
    }
park_costs = {
    0: 7.,
    1: 5.,
    2: 3.,
    3: 1.,
    4: 7.,
    5: 3.,
    6: 2.,
    }
walk_cost = 0.1
drive_cost = 0.1
katoto_cost = 10.0
random_accident_proba = 0.0
random_accident_cost = 90.0
allow_goback = False
pkatoto_full = .1

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

    pymdptoolbox_installed = True
    try:
        from iterators import PolicyIterationWithers, ValueIterationWithers
        print("Value Iteration")
        vi = ValueIterationWithers(env.transition, env.rewards, discount=0.99, max_iter=1000)
        vi.run()
        arr = vi.convergence_array
        print("Policy Iteration")
        pi = PolicyIterationWithers(env.transition, env.rewards, discount=0.99, max_iter=1000)
        pi.run()
        arrp = pi.convergence_array
    except (ImportError, NameError) as e:
        raise e
        vi, arr = None, []
        pi, arrp = None, []
        pymdptoolbox_installed = False
    plt.plot(arr)
    plt.xlabel("Iterations")
    plt.ylabel("Max absolute value change")
    plt.savefig('example-run-output/example-run-vi-convergence.png')
    plt.close()

    plt.plot(arrp)
    plt.xlabel("Iterations")
    plt.ylabel("Policy change")
    plt.savefig('example-run-output/example-run-pi-convergence.png')
    plt.close()

    print('Q-Learning')
    agent = QLAgent(gamma=0.99, epsilon=0.3, alpha=1, epsilon_shrink=0.9999, alpha_shrink=0.9999, max_episode_len=None)
    agent.fit(env, iters=9000)

    plt.plot(agent._abs_update_mean)
    plt.xlabel("Iterations")
    plt.ylabel("Max absolute Q-function change")
    plt.savefig('example-run-output/example-run-ql-convergence.png')
    plt.close()

    fig, ax = plt.subplots(3, 1, sharex=True)
    if pymdptoolbox_installed:
        ax[0].imshow(np.array(vi.V[:-3]).reshape(env.street_length, env._num_park_states).T, vmin=-20, vmax=5)
        ax[1].imshow(np.array(pi.V[:-3]).reshape(env.street_length, env._num_park_states).T, vmin=-20, vmax=5)
    else:
        pass

    ax[0].set_title('Value Iteration')
    ax[1].set_title('Policy Iteration')
    ax[2].imshow(agent.Qsa.max(axis=1)[:-3].reshape(env.street_length, env._num_park_states).T, vmin=-20, vmax=5)
    ax[2].set_title('Q-Learning')
    ax[2].set_xlabel('')
    fig.suptitle("Value function for states")
    plt.savefig('example-run-output/example-run-values.png')
