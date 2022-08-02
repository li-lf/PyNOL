import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from multiprocessing import Pool

import numpy as np
from applications.online_mdp.grid_world.loop_free_ssp import GridWorld
from applications.online_mdp.models.doreps import DOREPS, OptimisticDOREPS
from pynol.environment.environment import Environment
from pynol.learner.base import Hedge
from pynol.utils.plot import plot

K = 1000
row, column, prob = 10, 10, 0.9
actions = [(0, 1), (1, 0)]
num_actions = len(actions)
num_samples = 10
domain = GridWorld(row, column, actions, prob, num_samples)

seed = 0
np.random.seed(seed)


def loss_generator():
    loss = np.ones((row * column, num_actions))
    indii = np.random.randint(2, size=loss.shape[0])
    loss[range(loss.shape[0]), indii] = 0.
    loss = loss.flatten()[:-num_actions]
    return loss


oreps = Hedge(domain=domain, step_size=K**(-0.5), prior='minimum')
doreps = DOREPS(domain=domain, K=K, prior='minimum')
odoreps = OptimisticDOREPS(domain=domain, K=K, prior='minimum')

models = [oreps, doreps, odoreps]
labels = ['O-REPS', 'DO-REPS', 'Optimistic-DO-REPS']

freq = 50
loss_sequence = [loss_generator() for _ in range(K // freq + 1)]
loss_alg = np.zeros((len(models), num_samples, K))


def online_mdp(model):
    loss_once = np.zeros((num_samples, K))
    for k in range(K):
        print(k)
        loss_k = loss_sequence[k // freq]
        env = Environment()
        env.func = lambda x: np.dot(x, loss_k)
        env.optimism = loss_k
        q, _, _ = model.opt(env)
        loss_once[:, k] = domain.sample(q, loss_k)
    return loss_once


if __name__ == "__main__":
    p = Pool(processes=3)
    results = []
    for i in range(len(models)):
        results.append((i, p.apply_async(online_mdp, (models[i], ))))
    p.close()
    p.join()
    for i, result in results:
        loss_alg[i] = result.get()
    if os.path.exists('applications/online_mdp/results') is False:
        os.makedirs('applications/online_mdp/results')
    plot(loss_alg, labels, file_path='applications/online_mdp/results/loop_free_ssp.pdf')
