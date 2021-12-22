"""Combined minimization module."""
from typing import Any, List, Optional

from .minimization_function import MinimizationFunction


class CombinedMinimization(MinimizationFunction):
    """Combined minimization function"""

    def __init__(
        self,
        minimization_functions: List[MinimizationFunction],
        batch_size: int,
        function_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize a combined minimization function.

        Args:
            minimization_functions: a list of minimization functions.
            batch_size: size of the batch.
            function_weights: weights for the functions in the list. Defaults to None, a.k.a., equal weight to all functions.

        Raises:
            ValueError: in case there is a mismatch between the number of functions and the provided weights.
        """
        self.functions = minimization_functions
        self.weights = (
            function_weights if function_weights else [1.0] * len(self.functions)
        )
        self.batch_size_combined = batch_size
        if len(self.functions) != len(self.weights):
            raise ValueError("Length of function and weights lists do not match.")

    def evaluate(self, point: Any) -> float:
        """
        Evaluate a point.

        Args:
            point: point to evaluate.

        Returns:
            evaluation for the given point.
        """
        evaluation_batch = []
        for _ in range(self.batch_size_combined):
            evaluation = 0.0
            for function, weight in zip(self.functions, self.weights):
                evaluation += function.evaluate(point) * weight
            evaluation_batch.append(evaluation)

        return sum(evaluation_batch) / len(evaluation_batch)
