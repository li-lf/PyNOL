from abc import ABC, abstractmethod


class OptimismBase(ABC):
    """The abstract class defines the optimism for base-learners.

    Attributes:
        is_external (bool): Indicates the optimism is given by the environment
            or computed by the algorithm itself. The default is True.
    """

    def __init__(self, is_external: bool = True):
        self.is_external = is_external

    @abstractmethod
    def compute_optimism_base(self):
        """Compute the optimism for base-learners."""
        raise NotImplementedError()


class EnvironmentalOptimismBase(OptimismBase):
    """The class indicates the base-learner will use the environmental optimism.
    """

    def __init__(self):
        super().__init__()

    def compute_optimism_base(self, variables):
        return variables['optimism_env']


class LastGradOptimismBase(OptimismBase):
    """The class will set the optimism :math:`m_{t+1}` of the base-learners as
    :math:`m_t = \\nabla f_{t-1}(x_{t-1})`, where :math:`x_{t-1}` is the
    submitted decision at round :math:`t-1`."""

    def __init__(self):
        super().__init__(is_external=False)

    def compute_optimism_base(self, variables):
        return variables['grad']
