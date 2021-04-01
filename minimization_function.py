"""Minimization function module."""

class MinimizationFunction:
    def evaluate(self, point):
        raise NotImplementedError("Please Implement this evaluation method")


class DecoderBasedMinimization(MinimizationFunction):
    def __init__(self, smiles_decoder):
        self.generator = smiles_decoder
