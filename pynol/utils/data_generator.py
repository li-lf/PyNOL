import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DataGenerator(ABC):
    """Synthetic data generator."""

    def __init__(self):
        pass

    @abstractmethod
    def generate_data(self):
        """Generate data by parameters."""
        raise NotImplementedError()


class LinearRegressionGenerator(DataGenerator):
    """Generate data for linear regression."""

    def __init__(self):
        super().__init__()

    def generate_data(self,
                      T: int,
                      dimension: int,
                      stage: int = 1,
                      radius: float = 1.,
                      Gamma: float = 1.,
                      mu: float = 0.,
                      sigma: float = 0.05,
                      seed: Optional[int] = None):
        """Generate linear data with with abrupt changing environment.

        The synthetic data are generated as follows: at each round, the feature
        :math:`\\varphi_t \in \mathbb{R}^d` is is randomly generated from a ball
        with a radius of :math:`\Gamma`, i.e., :math:`\{\\varphi \in \mathbb{R}^d \mid
        \lVert \\varphi \\rVert_2 \leq \Gamma\}`; the associated label is set as
        :math:`y_t = \\varphi_t^\\top x_t^* + \epsilon_t`, where :math:`\epsilon_t` is
        a random Gaussian noise and :math:`x_t^*` is the underlying model. The
        underlying model :math:`x_t^*` is randomly sampled from a ball with a
        radius of :math:`R`. To simulate the non-stationary environments with
        abrupt changes. The total rounds are divided into :math:`S` stages in
        which the underlying model is forced to be stationary.

        Args:
            T (int): Number of total rounds.
            dimension (int): Dimension of the feature.
            stage (int): Numbers of stages.
            radius (float): radius of the underlying model.
            Gamma (float): radius of the feature.
            mu (float): Mean ("center") of the noise distribution.
            sigma (float): Standard deviation (spread or "width") of the noise
                distribution. Must be non-negative.
            seed (Optional): Random seed to generate data.
        """
        np.random.seed(seed)
        random_directions = np.random.normal(size=(dimension, T))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        random_radii = np.random.random(T)**(1 / dimension)
        feature = Gamma * (random_directions * random_radii).T
        label = np.zeros(T)
        step = math.ceil(T / stage)
        for i in range(stage):
            random_vec = np.random.normal(size=dimension)
            x = random_vec / np.linalg.norm(
                random_vec) * radius * np.random.rand()
            left, right = i * step, min((i + 1) * step, T)
            label[left:right] = np.dot(feature[left:right, :], x)
        noise = np.random.normal(mu, sigma, T)
        return feature, label + noise
