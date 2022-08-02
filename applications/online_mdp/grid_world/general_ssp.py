import cvxpy as cp
import numpy as np

from pynol.environment.domain import Domain


class GridWorld(Domain):

    def __init__(self,
                 row,
                 column,
                 prob=1.,
                 horizon=1.,
                 num_samples=1) -> None:
        self.row = row
        self.column = column
        self.prob = prob
        self._horizon = horizon
        self.diameter = row + column + 5  # approximate
        self.num_states = row * column
        self.index_states = np.arange(self.num_states)
        self.num_actions = 2
        self.index_actions = np.arange(self.num_actions)
        self.dimension = (self.num_states-1) * self.num_actions
        assert self.num_actions > 1 or self.prob == 1.
        self.layer_num = row + column - 1
        self.transition = self.init_transition()
        self.constraint_1, self.constraint_2 = self.constraint()
        self.states = np.zeros(num_samples, dtype=int)
        self.end_state = self.num_states - 1
        self.epsilon = 1e-3

    def init_transition(self):
        transition = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.path_start = path_start = self.row + self.column - 2
        transition[0, 0, 1], transition[0, 0, path_start] = self.prob, 1 - self.prob
        transition[0, 1, path_start], transition[0, 1, 1] = self.prob, 1 - self.prob
        transition[path_start-1, 0, self.num_states-1] = self.prob
        transition[path_start-1, 0, path_start-2] = 1 - self.prob
        transition[path_start-1, 1, path_start-2] = self.prob
        transition[path_start-1, 1, self.num_states-1] = 1 - self.prob
        transition[path_start, 0, 0] = self.prob
        transition[path_start, 0, path_start+1] = 1 - self.prob
        transition[path_start, 1, path_start+1] = self.prob
        transition[path_start, 1, 0] = 1 - self.prob

        for i in range(1, path_start-1):
            transition[i, 0, i+1], transition[i, 0, i-1] = self.prob, 1-self.prob
            transition[i, 1, i-1], transition[i, 1, i+1] = self.prob, 1-self.prob
        for i in range(path_start+1, self.num_states-1):
            transition[i, 0, i-1], transition[i, 0, i+1] = self.prob, 1-self.prob
            transition[i, 1, i+1], transition[i, 1, i-1] = self.prob, 1-self.prob
        return transition

    def trans(self, state, action):
        prob = self.transition[state, action, :]
        return np.random.choice(self.index_states, p=prob / prob.sum())

    def init_x(self, prior=None, seed=None):
        if prior is None:
            np.random.seed(seed)
            decision = np.random.rand(self.dimension)
            return self.project(decision)
        elif prior == 'uniform':
            return self.project(np.ones(self.dimension))
        elif prior == 'minimum':
            x = cp.Variable(self.dimension)
            obj = cp.Minimize(-cp.sum(cp.entr(x)))
            constr = [
                x >= self.epsilon, self.constraint_1[0] @ x == self.constraint_1[1],
                self.constraint_2[0] @ x <= self.constraint_2[1]
            ]
            problem = cp.Problem(obj, constr)
            problem.solve()
            return x.value
        else:
            raise TypeError(f'{prior} is not defined.')

    def constraint(self):
        l_state_constraint = np.eye(self.num_states)[:, None, :].repeat(
            self.num_actions, axis=1)
        l_state_constraint = l_state_constraint.reshape(-1, self.num_states).T
        r_state_constraint = self.transition.reshape(-1, self.num_states).T
        l_constraint_1 = (l_state_constraint - r_state_constraint)[:-1, :-self.num_actions]
        l_constraint_2 = np.ones((1, self.dimension))
        r_constraint_1 = np.zeros(l_constraint_1.shape[0])
        r_constraint_1[0] = 1.
        r_constraint_2 = np.ones(1) * self.horizon
        return (l_constraint_1, r_constraint_1), (l_constraint_2, r_constraint_2)

    def project(self, q):
        q = np.maximum(q, self.epsilon)
        x = cp.Variable(len(q))
        obj = cp.Minimize(cp.sum(cp.kl_div(x, q)))
        constr = [
            x >= self.epsilon, self.constraint_1[0] @ x == self.constraint_1[1],
            self.constraint_2[0] @ x <= self.constraint_2[1]
        ]
        problem = cp.Problem(obj, constr)
        try:
            problem.solve()
        except Exception:
            problem.solve(solver='SCS', max_iters=100)
        if not problem.status.startswith('optimal'):
            raise RuntimeError('Optimal solution is not found.')
        return np.maximum(x.value, self.epsilon)

    def sample(self, q, loss):
        q = q.reshape(-1, self.num_actions)
        loss = loss.reshape(-1, self.num_actions)
        index_actions = np.arange(self.num_actions)
        cum_loss = np.zeros_like(self.states, dtype=float)
        for i in range(len(self.states)):
            while(self.states[i] != self.end_state):
                action = np.random.choice(
                    index_actions,
                    p=q[self.states[i]] / q[self.states[i]].sum())
                cum_loss[i] += loss[self.states[i], action]
                self.states[i] = self.trans(self.states[i], action)
        self.states = np.zeros_like(self.states, dtype=int)
        return cum_loss

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon
        self.constraint_1, self.constraint_2 = self.constraint()
