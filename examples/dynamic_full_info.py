import os
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import OGD
from pynol.learner.models.dynamic.ader import Ader
from pynol.learner.models.dynamic.sword import SwordBest
from pynol.learner.models.dynamic.swordpp import SwordPP
from pynol.online_learning import multiple_online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

T, dimension, stage, R, Gamma, scale, seed = 10000, 3, 100, 1, 1, 1 / 2, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
D, r = 2 * R, R
G = scale * D * Gamma**2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2

seeds = range(5)
domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G
ogd = [
    OGD(domain=domain, step_size=min_step_size, seed=seed) for seed in seeds
]
ader = [
    Ader(
        domain=domain,
        T=T,
        G=G,
        surrogate=False,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

aderpp = [
    Ader(
        domain=domain,
        T=T,
        G=G,
        surrogate=True,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

sword = [
    SwordBest(
        domain=domain,
        T=T,
        G=G,
        L_smooth=L_smooth,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

swordpp = [
    SwordPP(
        domain=domain,
        T=T,
        G=G,
        L_smooth=L_smooth,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]
learners = [ogd, ader, aderpp, sword, swordpp]
labels = ['OGD', 'Ader', 'Ader++', 'Sword', 'Sword++']

if __name__ == "__main__":
    loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    _, loss, _ = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/dynamic_full_info.pdf')
