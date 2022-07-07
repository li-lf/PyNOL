from abc import ABC, abstractmethod
import numpy as np
from pynol.environment.environment import Environment


class Perturbation(ABC):
    """The abstract class for gradient estimation in bandit setting."""

    def __init__(self):
        pass

    @abstractmethod
    def perturb_x(self, x: np.ndarray):
        """Perturb the decision."""
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, env: Environment):
        """Get the loss of the perturbed decision(s)."""
        pass

    @abstractmethod
    def construct_grad(self, env: Environment):
        """Estimate the gradient."""
        pass


class OnePointPerturbation(Perturbation):
    """The abstract class for gradient estimation in bandit setting with
    one-point feedback.

    Args:
        domain (Domain): Feasible set for decisions.
        scale_perturb (float): Scale of perturbation.

    """

    def __init__(self, domain, scale_perturb):
        self.domain = domain
        self.scale_perturb = scale_perturb

    def perturb_x(self, x: np.ndarray):
        self.unit_vec = self.domain.unit_vec()
        self.x = x + self.scale_perturb * self.unit_vec
        return self.x

    def compute_loss(self, env: Environment):
        self.loss, surrogate_loss = env.get_loss(self.x)
        return self.loss, surrogate_loss

    def construct_grad(self):
        gradient = self.domain.dimension / self.scale_perturb * self.loss * self.unit_vec
        return gradient


class TwoPointPerturbation(Perturbation):
    """The abstract class for gradient estimation in bandit setting with
    two-point feedback.

    Args:
        domain (Domain): Feasible set for decisions.
        scale_perturb (float): Scale of perturbation.

    """

    def __init__(self, domain, scale_perturb):
        self.domain = domain
        self.scale_perturb = scale_perturb

    def perturb_x(self, x: np.ndarray):
        self.unit_vec = self.domain.unit_vec()
        self.x1 = x + self.scale_perturb * self.unit_vec
        self.x2 = x - self.scale_perturb * self.unit_vec
        return (self.x1, self.x2)

    def compute_loss(self, env: Environment):
        self.loss1, surrogate_loss1 = env.get_loss(self.x1)
        self.loss2, surrogate_loss2 = env.get_loss(self.x2)
        if surrogate_loss1 is not None and surrogate_loss2 is not None:
            surrogate_loss = 1 / 2 * (surrogate_loss1 + surrogate_loss2)
        else:
            surrogate_loss = None
        return 1 / 2 * (self.loss1 + self.loss2), surrogate_loss

    def construct_grad(self):
        gradient = self.domain.dimension / (2 * self.scale_perturb) * (
            self.loss1 - self.loss2) * self.unit_vec
        return gradient
