"""GP Optimisation module."""

import torch
from skopt import gp_minimize

class GPOptimizer:
    """GP Optimisation module."""
    def __init__(self, minimization_function):
        """
        Initialization.

        Arguments:
            minimization_function: the minimization_function e.g. MWMinization.evaluate.
        """

        self.minimization_function = minimization_function

    def optimize(self,params):
        """
        GP optimisation.

        Arguments:
            params: parameter for gp_minize (dict)

        Returns:
            res: The optimization result returned as a OptimizeResult object
        """

        res = gp_minimize(self.minimization_function,**params)
        return res
