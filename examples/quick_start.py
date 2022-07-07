import os

import matplotlib.pyplot as plt
import numpy as np
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.learner.base import OGD

T, dimension, step_size, radius, seed = 100, 10, 0.01, 1, 0
domain = Ball(dimension=dimension, radius=radius)
ogd = OGD(domain=domain, step_size=step_size, seed=seed)
loss = np.zeros(T)
env = Environment(func=lambda x: (x**2).sum())
for t in range(T):
    _, loss[t], _ = ogd.opt(env)
plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('Instantaneous Loss')
if os.path.exists('./results') is False:
    os.makedirs('./results')
plt.savefig('results/quick_start.pdf')
