import numpy as np
from perceptron import Perceptron


class Adaline(Perceptron):
    """
    Implementation of an Adaptive Linear Neuron, that can be abstracted to
    various input sizes or dimensions. Displays using pyplot.
    """
    ETA = 1

    def __init__(self, grph, eta, max_t):
        Perceptron.__init__(self, grph)
        self.ETA = eta
        self.max_t = max_t

    def update(self, y_t, x):
        r = []
        s_t = np.sign(np.inner(self.grph.w, x))
        for i in range(self.DIM):
            r.append(self.grph.w[i] + (self.ETA * (y_t - s_t) * x[i]))
        return r

    def fit(self):
        t = 0
        c = True
        while c:
            n = self.random_check()
            if n == -1 or t == self.max_t:
                c = False
            else:
                self.grph.w = self.update(self.grph.y[n], self.grph.training_matrix[n])
            t += 1
            print("t: {0}, w: {1}".format(t, self.grph.w))
        if self.grph.PLOT:
            self.grph.plot_g()  # In calling g() the 0th value is 1, corresponding to w_0
            self.grph.show_plot()
        # and the last value is not used in calculation, so is set as 0
        return t
