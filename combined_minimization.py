"""Combined minimization Class module."""

from minimization_function import DecoderBasedMinimization


class CombinedMinimization(DecoderBasedMinimization):
    """ Combined minimization function"""

    def __init__(self, minimization_functions, function_weights, batch_size):
        self.functions = minimization_functions
        self.weights = function_weights
        self.batchcombined = batch_size
        if len(self.functions) != len(self.weights):
            raise ValueError("Length of function and weights lists do not match.")

    def evaluate(self, point):
        evaluation_batch = []
        for i in range(self.batchcombined):
            evaluation = 0
            for function, weight in zip(self.functions, self.weights):
                evaluation += function.evaluate(point) * weight
            evaluation_batch.append(evaluation)

        return sum(evaluation_batch) / len(evaluation_batch)
