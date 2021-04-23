import mdptoolbox
import numpy as np

class ValueIteration(mdptoolbox.mdp.ValueIteration):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = kwargs['max_iter']
        self.args = args
        self.kwargs = kwargs
        del kwargs['max_iter']
        del kwargs['initial_value']
        # self.kwargs['max_iter'] = 1
        self._abs_diff_mean = []

    def run(self):
        iterator = mdptoolbox.mdp.ValueIteration(*self.args, **self.kwargs)
        iterator.run()
        Vi = np.array(iterator.V)
        for i in range(self.max_iter):
            iterator = mdptoolbox.mdp.ValueIteration(*self.args, **self.kwargs, initial_value=Vi.tolist(), max_iter=1)
            iterator.run()
            V = np.array(iterator.V)
            self._abs_diff_mean.append(np.abs(Vi - V).mean())
            Vi = V
        # import pdb; pdb.set_trace()
        self.V = V
        self.policy = iterator.policy


def iterate_value(transitions, reward, discount=1, epsilon=0.01, max_iter=1000, initial_value=0):
    arr = []
    iterator = mdptoolbox.mdp.ValueIteration(transitions, reward, discount, epsilon, max_iter=1, initial_value=initial_value)
    iterator.run()
    Vi = iterator.V
    for i in range(max_iter):
        iterator = mdptoolbox.mdp.ValueIteration(transitions, reward, discount, epsilon, max_iter=1, initial_value=list(Vi),)
        iterator.run()
        V = iterator.V
        arr.append(np.abs(np.array(V) - np.array(Vi)).max())
        Vi = V
    return iterator, arr


def iterate_policy(transitions, reward, discount=1, max_iter=1000, policy0=None):
    arr = []
    iterator = mdptoolbox.mdp.PolicyIteration(transitions=transitions, reward=reward, discount=discount, max_iter=1, policy0=policy0, eval_type=1)
    iterator.run()
    Pi = iterator.policy
    for i in range(max_iter):
        iterator = mdptoolbox.mdp.PolicyIteration(transitions=transitions, reward=reward, discount=discount, max_iter=1, policy0=list(Pi), eval_type=1)
        iterator.run()
        P = iterator.policy
        arr.append(np.abs(np.array(P) - np.array(Pi)).mean())
        Pi = P
    return iterator, arr
