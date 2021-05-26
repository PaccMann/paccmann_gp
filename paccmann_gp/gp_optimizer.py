"""GP optimization module."""
from skopt import gp_minimize
from typing import Dict, Any
from .minimization_function import MinimizationFunction


class GPOptimizer:
    """GP optimizer."""

    def __init__(self, minimization_function: MinimizationFunction) -> None:
        """
        Initialization of the optimizer.

        Args:
            minimization_function: the minimization_function.
        """
        self.minimization_function = minimization_function

    def optimize(self, parameters: Dict) -> Any:
        """
        GP optimization.

        Args:
            parameters: parameters for gp_minimize.

        Returns:
            optimization result.
        """
        return gp_minimize(self.minimization_function, **parameters)
