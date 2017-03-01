import numpy as np
from perceptron import Perceptron


class Adaline(Perceptron):
    """
    Implementation of an Adaptive Linear Neuron, that can be abstracted to
    various input sizes or dimensions. Displays using pyplot.
    """
    ETA = 1

    def __init__(self, dim, num, grph, eta):
        Perceptron.__init__(self, dim, num, grph)
        self.ETA = eta

    def update(self, y_t, x):
        r = []
        s_t = np.sign(self.inner_product(x))
        for i in range(self.DIM):
            r.append(self.grph.w[i] + (self.ETA * (y_t - s_t) * x[i]))
        return r
