import argparse
import os, sys
import time
from functools import partial
from multiprocessing import Pool
import math
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

from pynol.learner.base import OGD
from pynol.environment.domain import Ball
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.learner.models.dynamic.ader import Ader
from pynol.learner.models.dynamic.sword import SwordBest
from pynol.learner.models.dynamic.swordpp import SwordPP
from pynol.environment.environment import Environment
from pynol.environment.loss_function import HuberLoss
from pynol.online_learning import online_learning, multiple_online_learning
from pynol.utils.plot import plot

parser = argparse.ArgumentParser()
parser.add_argument('--real_world', action='store_true')
parser.add_argument('--threshold', type=float, default=2.0)
args = parser.parse_args()
real_world = args.real_world
if real_world is False:
    dataset = 'synAbrupt'
    radius, Gamma = 1., 1.
    D = 2 * radius
    G = D * Gamma**2
    L_smooth = Gamma**2
    T, dimension, stage, data_seed = 20000, 10, 10, 42
    X, y = LinearRegressionGenerator().generate_data(T, dimension, stage, radius, Gamma, seed=data_seed)
else:
    data = loadmat("dataset/SRU_1.mat")
    dataset = 'realWorld'
    X = data["x_train"]
    y = data["y_train"]
    T = len(X)
    dimension = len(X[0])
    D = 2
    radius = D / 2
    Gamma = dimension**0.5
    G = args.threshold * Gamma
    L_smooth = Gamma**2


domain = Ball(dimension=dimension, radius=radius)
seeds = range(5)
min_step_size, max_step_size = D / G * T**(-0.5), 1 / (8 * L_smooth)
ogd = [OGD(domain, step_size=min_step_size, seed=seed) for seed in seeds]
ader = [Ader(domain, T, G, surrogate=True, min_step_size=min_step_size, max_step_size=max_step_size, seed=seed) for seed in seeds]
sword = [SwordBest(domain, T, G, L_smooth, min_step_size=min_step_size, max_step_size=max_step_size, seed=seed) for seed in seeds]
swordpp = [SwordPP(domain, T, G, L_smooth, min_step_size=min_step_size, max_step_size=max_step_size, seed=seed) for seed in seeds]

learners = [ogd, ader, sword, swordpp]
labels = ['OGD', 'Ader', 'Sword', 'Sword++']

loss = np.zeros((len(learners), len(learners[0]), T))
time_all = np.zeros_like(loss)


if __name__ == "__main__":
    loss_func = HuberLoss(feature=X, label=y, threshold=args.threshold)
    env = Environment(func_sequence=loss_func)
    loss = np.zeros((len(learners), len(learners[0]), T))
    tm = np.zeros_like(loss)
    for i in range(len(learners)):
        for j in range(len(learners[i])):
            _, loss[i, j], _, tm[i, j]= online_learning(T, env, learners[i][j])
    ### if you only care about the loss, you can uncomment the following lines to use multiple processes to speed up
    # _, loss, _, _ = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/swordpp_'+ dataset +'_loss.pdf')
    plot(tm, labels, file_path='./results/swordpp_'+ dataset +'_time.pdf', y_label='Cumulative Time', loc='lower right', scale='log')
