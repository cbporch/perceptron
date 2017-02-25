import numpy as np
from perceptron import Perceptron


class Adaline(Perceptron):
    ETA = 1

    def __init__(self, dim, num, eta):
        Perceptron.__init__(self, dim, num)
        self.ETA = eta

    def update(self, y_t, x):
        r = []
        s_t = np.sign(self.inner_product(x))
        for i in range(self.DIM):
            r.append(self.grph.w[i] + (self.ETA * (y_t - s_t) * x[i]))
        return r
