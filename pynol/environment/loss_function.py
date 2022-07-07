from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np


class LossFunction(ABC):
    """An abstract class for loss function.

    Users can define their loss functions by inheriting from this class and override the method :meth:`~ pynol.environment.loss_function.LossFunction.__getitem__`.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        pass


class InnerLoss(LossFunction):
    """This class defines the inner loss function.

    Args:
        feature (numpy.ndarray): Features of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        func = InnerLoss(feature=np.random.rand(1000, 5))  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the inner loss function :math:`f_t(x) =
    \langle \\varphi_t, x \\rangle`, where :math:`\\varphi_t` is the feature at
    round :math:`t`.
    """

    def __init__(self, feature: np.ndarray = None, scale: float = 1.) -> None:
        self.feature = feature
        self.scale = scale

    def __getitem__(self, t: int):
        return lambda x: self.scale * np.dot(x, self.feature[t])


class SquareLoss(LossFunction):
    """This class defines the logistic loss function.

    Args:
        feature (numpy.ndarray): Features of the environment.
        label (numpy.ndarray): Labels of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        func = SquareLoss(feature, label)  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the square loss function :math:`f_t(x) =
    \\frac{1}{2} (y_t - \langle \\varphi_t, x \\rangle)^2`, where
    :math:`\\varphi_t` and :math:`y_t` are the feature and label at
    round :math:`t`.
    """

    def __init__(self,
                 feature: np.ndarray = None,
                 label: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.feature = feature
        self.label = label
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: self.scale * 1 / 2 * (
            (np.dot(x, self.feature[t]) - self.label[t])**2)


class LogisticLoss(LossFunction):
    """This class defines the logistic loss function.

    Args:
        Feature (numpy.ndarray): Features of the environment.
        label (numpy.ndarray): Labels of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        func = LogisticLoss(feature, label)  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the loss function :math:`f_t(x) = y \log (\\frac{1}{1+e^{-\\varphi_t^\\top x}})+(1-y) \log (1-\\frac{1}{1+e^{-\\varphi_t^\\top x}})` where
    :math:`\\varphi_t` and :math:`y_t` are the feature and label at
    round :math:`t`.
    """

    def __init__(self,
                 feature: np.ndarray = None,
                 label: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.feature = feature
        self.label = label
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return partial(self.func, t=t)

    def func(self, x, t):
        prediction = 1 / (1 + np.e**(-np.dot(x, self.feature[t])))
        loss = prediction * np.log(
            self.label[t]) + (1 - prediction) * np.log(1 - self.y[t])
        return self.scale * loss


class FuncWithSwitch:
    """This class defines the loss function with switching cost.

    Args:
        f (LossFunction): Origin loss function.
        penalty (float): Penalty coefficient of the switching cost.
        norm (non-zero int, numpy.inf): Order of the norm. The default is 2 norm.
        order (int): Order the the switching cost. The default is 2.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        f = SquareLoss(feature, label)
        func = FuncWithSwitch(f, penalty=1, norm=2, order=2)

    Then, call ``func[t]`` will return the square loss function with switching
    cost :math:`f_t(x) = \\frac{1}{2} (y_t - \langle \\varphi_t, x \\rangle)^2 +
    \lVert x - x_{t-1}\\rVert_2^2` where :math:`\\varphi_t` and :math:`y_t` are
    the feature and label at round :math:`t`.
    """

    def __init__(self,
                 f: LossFunction = None,
                 penalty: float = 1.,
                 norm: int = 2,
                 order: int = 2) -> None:
        self.f = f
        self.penalty = penalty
        self.norm = norm
        self.order = order
        self.x_last = None

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return partial(self.func, f=self.f[t])

    def func(self, x: np.ndarray, f: Callable[[np.ndarray], float]):
        assert x.ndim == 1 or x.ndim == 2
        if self.x_last is None:
            self.x_last = x
        if x.ndim == 1:
            loss = f(x) + self.penalty * np.linalg.norm(
                x - self.x_last, ord=self.norm)**self.order
        else:
            loss = f(x) + self.penalty * np.linalg.norm(
                x - self.x_last, ord=self.norm, axis=1)**self.order
        self.x_last = x
        return loss
