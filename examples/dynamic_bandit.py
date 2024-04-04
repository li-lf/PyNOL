import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import BGDOnePoint, BGDTwoPoint
from pynol.learner.models.dynamic.pbgd import PBGDOnePoint, PBGDTwoPoint
from pynol.online_learning import multiple_online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

T, dimension, stage, R, Gamma, scale, seed = 10000, 3, 100, 1, 1, 1 / 2, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
D, r = 2 * R, R
G = scale * D * Gamma**2
C = scale * 1 / 2 * (D * Gamma)**2
L_lipschitz = scale * D * Gamma**2
L_tilde = 3 * L_lipschitz + L_lipschitz * R / r

seeds = range(5)
domain = Ball(dimension=dimension, radius=R)

step_one_point = (dimension * C * L_tilde)**(-0.5) * (7 * R**2 / T)**0.75
perturb_one_point = (dimension * C / L_tilde)**0.5 * (7 * R**2 / T)**0.25
bgd1 = [
    BGDOnePoint(
        domain=domain,
        step_size=step_one_point,
        scale_perturb=perturb_one_point,
        seed=seed) for seed in seeds
]

step_two_point = R / (dimension * L_lipschitz) * (7 / (2 * T))**0.5
perturb_two_point = dimension * R / T**0.5
bgd2 = [
    BGDTwoPoint(
        domain=domain,
        step_size=step_two_point,
        scale_perturb=perturb_two_point,
        seed=seed) for seed in seeds
]

pbgd1 = [PBGDOnePoint(domain=domain, T=T, C=C, seed=seed) for seed in seeds]
pbgd2 = [
    PBGDTwoPoint(domain=domain, T=T, L_lipschitz=L_lipschitz, seed=seed)
    for seed in seeds
]

learners = [bgd1, bgd2, pbgd1, pbgd2]
labels = ['BGD1', 'BGD2', 'PBGD1', 'PBGD2']

if __name__ == "__main__":
    loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    _, loss, _, _ = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/dynamic_bandit.pdf')
