from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union

import numpy as np


class Domain(ABC):
    """An abstract class representing the feasible domain.

    Args:
        dimension (int): Dimension of the feasible set.

    """

    def __init__(self, dimension):
        self.dimension = dimension

    @abstractmethod
    def init_x(self, prior: Optional[Union[str, np.ndarray]],
               seed: Optional[int]) -> np.ndarray:
        """Initialize a decision in the domain.
        """
        pass

    @abstractmethod
    def project(self, x: np.ndarray):
        """Project the decision :math:`x` back to the feasible set.
        """
        pass


class Ball(Domain):
    """This class defines a Euclid ball as the feasible set.

    Args:
        dimension (int): Dimension of the feasible set.
        radius (float): Radius of the ball.
        center (numpy.ndarray, optional): Coordinates of the center point.
        Default to the origin point if not specified.

    Attributes:
        R (float): Radius of the minimum outside ball, which is useful for
            irregular domains.
        r (float): Radius of the maximum inside ball, which is useful for
            irregular domains.
    """

    def __init__(self,
                 dimension: int,
                 radius: float = 1.,
                 center: np.ndarray = None):
        super().__init__(dimension=dimension)
        self.radius = radius
        self.center = center if center is not None else np.zeros(dimension)
        self.R = radius  # the radius of the minimum outside ball
        self.r = radius  # the radius of the maximum inside ball

    def init_x(self, prior: Optional[Union[str, np.ndarray]],
               seed: Optional[int]) -> np.ndarray:
        """Initialize a decision in the domain.

        Args:
            prior (numpy.ndarray, optional): Prior information to initialize
                the decision. If a ``numpy.ndarray`` is given, the method will
                return ``prior`` as the decision, otherwise return a random vector
                in the Ball.
            seed (int, optional): Random seed to initial the decision if `prior=None`.

        Returns:
            numpy.ndarray: a decision in the ball.
        """
        if prior is not None:
            assert len(prior) == self.dimension
            return np.array(prior)
        else:
            np.random.seed(seed)
            random_direction = np.random.normal(size=self.dimension)
            random_direction /= np.linalg.norm(random_direction)
            random_radius = np.random.random()
            return self.radius * random_direction * random_radius

    def unit_vec(self, seed: Optional[int] = None) -> np.ndarray:
        """Sample a unit vector uniformly at random.

        Args:
            seed (int, optional): Random seed to sample the vector.

        Returns:
            numpy.ndarray: a decision in the ball.
        """
        np.random.seed(seed)
        random_direction = np.random.normal(size=self.dimension)
        random_direction /= np.linalg.norm(random_direction)
        return random_direction

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project the decision :math:`x` back to the ball by Euclid distance.

        Args:
            x(numpy.ndarray): the vector to be projected.

        Returns:
            numpy.ndarray: the projected vector.
        """
        distance = np.linalg.norm(x - self.center)
        if distance > self.r:
            x = self.center + (x - self.center) * self.r / distance
        return x

    def __mul__(self, scale: float):
        new_ball = deepcopy(self)
        new_ball.radius *= scale
        new_ball.R *= scale
        new_ball.r *= scale
        return new_ball

    def __rmul__(self, scale: float):
        return self.__mul__(scale)


class Simplex(Domain):
    """This class defines a simplex as the feasible set.

    Args:
        dimension (int): Dimension of the feasible set.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension=dimension)

    def init_x(self,
               prior: Union[str, np.ndarray] = 'uniform',
               seed: Optional[int] = None) -> np.ndarray:
        """Initialize a decision x in the domain.

        Args:
            prior (numpy.ndarray, 'uniform', 'nonuniform', optional): Prior
                information to initialize the decision. If a ``numpy.ndarray`` is
                given, the method will return ``prior`` as the decision; if
                ``prior='uniform'``, the method will return the uniform vector
                :math:`x_i = 1/d, \\forall i \in [d]`; if ``prior='nonuniform'``,
                the method will return :math:`x_i = \\frac{d+1}{d} \cdot
                \\frac{1}{i(i+1)}, \\forall i \in [d]`, where :math:`d` is the
                dimension of the simplex; if ``prior=None``, the method will return
                a random vector in the simplex.
            seed (int, optional): Random seed to initial the decision if `prior=None`.

        Returns:
            numpy.ndarray: a decision in the ball.
        """
        if prior is None:
            np.random.seed(seed)
            x = np.random.rand(self.dimension)
            x /= np.linalg.norm(x, ord=1)
        elif isinstance(prior, np.ndarray):
            x = prior
        elif prior == 'uniform':
            x = np.ones(self.dimension) / self.dimension
        elif prior == 'nonuniform':
            x = np.array([(self.dimension + 1) / (self.dimension * i * (i + 1))
                          for i in range(1, self.dimension + 1)])
        else:
            raise TypeError(f'{prior} is not defined.')
        return x

    def project(self,
                x: np.ndarray,
                dist: str = 'kl_div',
                norm: Union[int, str] = 1):
        """Project the decision :math:`x` back to the simplex.

        Args:
            x (numpy.ndarray): Vector to be projected.
            dist (str): Distance metric used to project the decision.  Valid
                options include ``'kl_div'`` or ``'norm'``. if ``dist=kl_div``, the
                return decision will be :math:`x_i = x_i / \sum_j x_j`, otherwise,
                the norm distance will be used to project the decision.
            norm (int, str, optional): Type of norm which is only used when
                ``dist='norm'``. Valid options include any positive integer or
                ``'inf'`` (infinity norm).

        Returns:
            numpy.ndarray: the projected vector.
        """
        if dist == 'kl_div':
            return x / np.linalg.norm(x, ord=1)
        elif dist == 'norm':
            import cvxpy as cp  # Only import when it needed since it will raise error when using too many sub-processes in multiprocessing.
            y = cp.Variable(self.dimension)
            obj = cp.Minimize(cp.sum(cp.atoms.norm(y - x, p=norm)))
            constr = [y >= 0, cp.sum(y) == 1]
            problem = cp.Problem(obj, constr)
            problem.solve()
            return np.array(y.value).squeeze()
        else:
            raise TypeError(f'{dist} is not defined.')
