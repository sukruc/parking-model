import mdptoolbox
import numpy as np
import time as _time
_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
    "condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal " \
    "policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value " \
    "function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


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


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))


class PolicyIterationWithers(mdptoolbox.mdp.PolicyIteration):

    def run(self):
        # Run the policy iteration algorithm.
        # If verbose the print a header
        self._arr = []
        self._time_arr = []
        if self.verbose:
            print('  Iteration\t\tNumber of different actions')
        # Set up the while stopping condition and the current time
        done = False
        self.time = _time.time()
        # loop until a stopping condition is reached
        while not done:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, null = self._bellmanOperator()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            self._arr.append(n_different)

            # if verbose then continue printing a table
            if self.verbose:
                print(('    %s\t\t  %s') % (self.iter, n_different))
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            if n_different == 0:
                done = True
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
            elif self.iter == self.max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
            else:
                self.policy = policy_next
        # update the time to return th computation time
        self.time = _time.time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())


def getSpan(W):
    """Return the span of W

    sp(W) = max W(s) - min W(s)

    """
    return W.max() - W.min()


class ValueIterationWithers(mdptoolbox.mdp.ValueIteration):
    def run(self):
        # Run the value iteration algorithm.
        self._arr = []
        if self.verbose:
            print('  Iteration\t\tV-variation')

        self.time = _time.time()
        while True:
            self.iter += 1

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = getSpan(self.V - Vprev)
            self._arr.append(variation)

            if self.verbose:
                print(("    %s\t\t  %s" % (self.iter, variation)))

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

        self.time = _time.time() - self.time
