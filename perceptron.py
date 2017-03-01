from math import ceil
import numpy as np


class Perceptron:
    """
    Implementation of a Perceptron Learning Algorithm, that can be abstracted to
    various input sizes (NUM) or dimensions (DIM). Displays using pyplot.
    """

    def __init__(self, dim, num, grph):
        self.DIM = dim
        self.NUM = num
        self.grph = grph

    def update(self, y_t, x):  # Perceptron update function
        r = []  # w(t+1)
        for i in range(self.DIM):
            r.append(self.grph.w[i] + y_t * x[i])
        return r

    def inner_product(self, x):  # Inner Product/Dot Product of vectors w and x
        return sum(self.grph.w[i] * x[i] for i in range(self.DIM))

    def check(self):  # verify if all points are classified correctly
        for n in range(len(self.grph.y)):
            if self.grph.y[n] * self.inner_product(self.grph.training_matrix[n]) <= 1:
                return n
        return -1

    def random_check(self):
        misclass = []
        for n in range(len(self.grph.y)):
            if self.grph.y[n] * self.inner_product(self.grph.training_matrix[n]) <= 1:
                misclass.append(n)
        if len(misclass) > 0:
            return misclass[ceil(np.random.rand(1) * (len(misclass) - 1))]
        else:
            return -1

    def fit(self):
        t = 0
        c = True
        while c:
            n = self.random_check()
            if n == -1 or (not isinstance(self, Perceptron)) and t == 1000:
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
