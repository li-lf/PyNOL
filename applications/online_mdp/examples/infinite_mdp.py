from functools import partial
import os
import sys


sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import copy
from multiprocessing import Pool
from pynol.environment.environment import Environment
import numpy as np
from applications.online_mdp.grid_world.infinite_horizon_mdp import GridWorld
from applications.online_mdp.models.redoreps import REDOREPS
from pynol.learner.base import Hedge
from pynol.environment.loss_function import FuncWithSwitch, InnerLoss
from pynol.utils.plot import plot

T = 5000
row, column, prob = 10, 10, 0.9
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
freq, num_samples = 500, 10
domain = GridWorld(row, column, actions, prob, num_samples)
seed = 0
mixing = 10.
oreps = Hedge(domain=domain, step_size=T**(-0.5))
doreps = REDOREPS(
    domain=domain,
    T=T,
    penalty=0.,
    min_step_size=T**(-0.5),
    max_step_size=1,
    prior='minimum')
odoreps = REDOREPS(
    domain=domain,
    T=T,
    penalty=mixing,
    min_step_size=T**(-0.5),
    max_step_size=1,
    prior='minimum')

np.random.seed(seed)


def loss_generator():
    loss = np.ones((row * column, len(actions)))
    indii = np.random.randint(len(actions), size=loss.shape[0])
    loss[range(loss.shape[0]), indii] = 0.
    loss = loss.flatten()
    return loss


loss_sequence = [loss_generator() for _ in range(T // freq + 1)]
models = [oreps, doreps, odoreps]
labels = ['O-REPS', 'DO-REPS', 'REDO-REPS']
loss = np.zeros((len(models), num_samples, T))
surrogate_loss = np.zeros((len(models), T))
domains = [copy.deepcopy(domain) for _ in range(len(models))]


def online_mdp(model, domain):
    loss_once = np.zeros((num_samples, T))
    surrogate_loss_once = np.zeros(T)
    surrogate_func = FuncWithSwitch(penalty=mixing, norm=1, order=1)

    for t in range(T):
        print(t)
        env = Environment(use_surrogate_grad=False)
        loss_t = loss_sequence[t // freq]
        env.func = lambda x: np.dot(x, loss_t)
        env.surrogate_func = partial(surrogate_func.func, f=env.func)
        q, _, surrogate_loss_once[t] = model.opt(env)
        loss_once[:, t] = domain.sample(q, loss_t)
    return loss_once, surrogate_loss_once


if __name__ == "__main__":
    p = Pool(processes=3)
    results = []
    for i in range(len(models)):
        results.append((i, p.apply_async(online_mdp, (models[i], domains[i]))))
    p.close()
    p.join()
    for i, result in results:
        loss[i], surrogate_loss[i] = result.get()
    if os.path.exists('applications/online_mdp/results') is False:
        os.makedirs('applications/online_mdp/results')
    plot(
        loss,
        labels,
        file_path='applications/online_mdp/results/infinite_mdp.pdf',
        x_label='Step')
    plot(
        surrogate_loss,
        labels,
        file_path='applications/online_mdp/results/infinite_mdp_surr.pdf',
        x_label='Step',
        y_label='Cumulative Surrogate Loss')
