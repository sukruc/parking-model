from streetv2 import StreetParking
from rlagent import QLAgent, SarsaAgent
import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import iterators

if __name__ == '__main__':
    street_length = 24
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
    walk_cost = 0.3
    drive_cost = 0.01
    katoto_cost = 10.0
    random_accident_proba = 0.0
    random_accident_cost = 9000.0

    gamma = 0.99

    params = dict(
        street_length=street_length,
        park_probas=park_probas,
        park_costs=park_costs,
        walk_cost=walk_cost,
        drive_cost=drive_cost,
        katoto_cost=katoto_cost,
        random_accident_proba=random_accident_proba,
        random_accident_cost=random_accident_cost,
        )

    p = StreetParking(**params)

    gamma = 0.9999

    params = dict(
        street_length=street_length,
        park_probas=park_probas,
        park_costs=park_costs,
        walk_cost=walk_cost,
        drive_cost=drive_cost,
        katoto_cost=katoto_cost,
        random_accident_proba=random_accident_proba,
        random_accident_cost=random_accident_cost,
        )

    p = StreetParking(**params)
    vi = iterators.ValueIteration(p.transition, p.rewards, discount=0.99, max_iter=1, )


    vi = mdptoolbox.mdp.ValueIteration(p.transition, p.rewards, discount=1, max_iter=1)
    vi.run()
    vi.V[-65:-60]
    vi = mdptoolbox.mdp.ValueIteration(p.transition, p.rewards, discount=1, max_iter=1, initial_value=list(vi.V))
    vi.run()
    vi.V[-65:-60]

    plt.plot(vi._abs_diff_mean)
    plt.show()
    sars = QLAgent(gamma=gamma, epsilon=0.2, alpha=0.7, epsilon_shrink=0.99, alpha_shrink=0.999)
    # sars = QLAgent(gamma=0.9, epsilon=0.9, alpha=0.7, epsilon_shrink=0.99, alpha_shrink=0.99)

    sars.fit(p, iters=40000)

    plt.plot(sars._abs_update_mean)
    plt.show()


    pi = mdptoolbox.mdp.PolicyIteration(p.transition, p.rewards, discount=0.99, max_iter=200)
    pi.run()

    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(np.array(vi.V[:-3]).reshape(p.street_length, p._num_park_states).T)
    ax[0].set_title('VI')
    ax[2].imshow(np.array(pi.V[:-3]).reshape(p.street_length, p._num_park_states).T)
    ax[2].set_title('PI')
    # plt.plot(vi.V, '-', label='Value Iteration')
    # plt.plot(pi.V, '--', label='Policy Iteration')
    # plt.plot(qpolicy, '--', label='Q-Learning')
    # plt.legend()
    # plt.show()
    qpolicy = sars.Qsa.max(axis=1)[:-3]
    ax[1].imshow(qpolicy.reshape(p.street_length, p._num_park_states).T)
    ax[1].set_title('Q')
    # plt.colorbar()
    plt.show()

    # street_length = 200
    # park_probas = {
    #      0: {
    #         'pexist': 0.5,
    #         'poccupied': 0.2,
    #         },
    #      1: {
    #         'pexist': 0.4,
    #         'poccupied': 0.6,
    #         },
    #      2: {
    #         'pexist': 0.07,
    #         'poccupied': 0.9,
    #         },
    #      3: {
    #         'pexist': 0.03,
    #         'poccupied': 0.9,
    #         },
    #     }
    # park_costs = {
    #     0: 7.,
    #     1: 5.,
    #     2: 3.,
    #     3: 1.,
    # }
    # walk_cost = 0.3
    # drive_cost = 0.5
    # katoto_cost = 10.0
    # random_accident_proba = 1e-6
    # random_accident_cost = 9000.0
    #
    # gamma = 0.99
    #
    # params = dict(
    #     street_length=street_length,
    #     park_probas=park_probas,
    #     park_costs=park_costs,
    #     walk_cost=walk_cost,
    #     drive_cost=drive_cost,
    #     katoto_cost=katoto_cost,
    #     random_accident_proba=random_accident_proba,
    #     random_accident_cost=random_accident_cost,
    #     )
    #
    # p = StreetParking(**params)
    # sars = SarsaAgent(gamma=gamma, epsilon=0.1, alpha=0.25)
    #
    # sars.fit(p, iters=90000)
    # vi = mdptoolbox.mdp.ValueIteration(p.transition, p.rewards, discount=0.99)
    # vi.run()
    #
    # plt.plot(vi.V, '--', label='Value Iteration')
    # plt.plot(tuple(sars.Qsa.max(axis=1)), '--', label='Q-Learning')
    # plt.legend()
    #
    #
    # si = [sars.policy(i) for i in range(p.nS)]
    # fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    # ax[0].imshow(np.flip(np.array(vi.policy[:-3]).reshape(street_length, len(park_costs) * 2).T, axis=1))
    # ax[1].imshow(np.flip(np.array(si[:-3]).reshape(street_length, len(park_costs) * 2).T, axis=1))
    # plt.show()
