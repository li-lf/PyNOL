import os
import sys


sys.path.insert(
    0,
     os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from multiprocessing import Pool
from pynol.environment.environment import Environment
import numpy as np
from applications.online_mdp.grid_world.general_ssp import GridWorld
from applications.online_mdp.models.codoreps import CDOREPS, AdaOREPS, OptimisticCDOREPS
from pynol.learner.base import Hedge
from pynol.utils.plot import plot

seed = 0
K = 1000
row, column, prob = 10, 10, 0.9
num_samples = 10
num_actions = 2
domain = GridWorld(row, column, prob=prob, num_samples=num_samples)
domain.horizon = domain.diameter / K**(-0.25)

oreps = Hedge(domain=domain, step_size=K**(-0.5), prior='minimum')
doreps = AdaOREPS(domain=domain, K=K, prior='minimum')
cdoreps = CDOREPS(domain=domain, K=K, prior='minimum')
ocdoreps = OptimisticCDOREPS(domain=domain, K=K, prior='minimum')
models = [oreps, doreps, cdoreps, ocdoreps]
labels = ['SSP-O-REPS', 'Ada-O-REPS', 'CODO-REPS', 'Optimistic-CODO-REPS']
loss_alg = np.zeros((len(models), num_samples, K))

np.random.seed(seed)
loss = np.ones((2, row * column, 2))
loss[0, :, 0] = 0.
loss[1, :, 1] = 0.
loss = loss.reshape(2, -1)[:, :-num_actions]

freq = 50


def online_mdp(model):
    loss_once = np.zeros((num_samples, K))
    for k in range(K):
        print(k)
        loss_k = loss[k % (2 * freq) // freq]
        env = Environment()
        env.func = lambda x: np.dot(x, loss_k)
        env.optimism = 2 * loss_k
        q, _, _ = model.opt(env)
        loss_once[:, k] = domain.sample(q, loss_k)
    return loss_once


if __name__ == "__main__":
    p = Pool(processes=4)
    results = []
    for i in range(len(models)):
        results.append((i, p.apply_async(online_mdp, (models[i], ))))
    p.close()
    p.join()
    for i, result in results:
        loss_alg[i] = result.get()
    if os.path.exists('../results') is False:
        os.makedirs('../results')
    plot(loss_alg, labels, file_path='../results/general_ssp_2.pdf')
