from math import ceil
import numpy as np


class Perceptron:
    """
    Implementation of a Perceptron, using the Perceptron Learning Algorithm, that can be abstracted to
    various input sizes or dimensions. Displays using pyplot.

    Parameters:
        dim: Dimension of data
        num: number of datapoints
        grph: Graph to fit
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

    # def inner_product(self, x):  # Inner Product/Dot Product of vectors w and x
    #     return sum(self.grph.w[i] * x[i] for i in range(self.DIM))

    def check(self):  # verify if all points are classified correctly
        for n in range(len(self.grph.y)):
            if self.grph.y[n] * np.inner(self.grph.w, self.grph.training_matrix[n]) <= 1:
                return n
        return -1

    def random_check(self):
        misclass = self.grph.get_misclassed()
        if misclass == -1:
            return -1
        else:
            return misclass[ceil(np.random.rand(1) * (len(misclass) - 1))]

    def fit(self):
        t = 0
        err = True
        while err:
            n = self.random_check()  # get a misclassed point
            if n == -1:
                err = False
            else:
                self.grph.w = self.update(self.grph.y[n], self.grph.training_matrix[n])
            t += 1
            print("t: {0}, w: {1}".format(t, self.grph.w))
        if self.grph.PLOT:
            self.grph.plot_g()  # In calling g() the 0th value is 1, corresponding to w_0
            # self.grph.show_plot()  # and the last value is not used in calculation, so is set as 0
        return t

    def pocket_fit(self):
        x_iteration = []
        y_err_in = []
        t = 0
        best_w = [0, 0, 0]
        lowest_err = 1
        has_err = True
        while has_err:
            x_iteration.append(t)
            err = len(self.grph.get_misclassed()) / self.NUM
            if err < lowest_err:
                lowest_err = err
                best_w = self.grph.w
            y_err_in.append(lowest_err)
            n = self.random_check()  # get a misclassed point
            if n == -1 or t >= 999:
                has_err = False
            else:
                self.grph.w = self.update(self.grph.y[n], self.grph.training_matrix[n])
            t += 1
            print("t: {0}, E_in: {1}, w: {2}".format(t, lowest_err, best_w))
        if self.grph.PLOT:
            self.grph.plot_g()  # In calling g() the 0th value is 1, corresponding to w_0
            # self.grph.show_plot()  # and the last value is not used in calculation, so is set as 0
        # self.grph.plotly_pocket(y_err_in)
        return t, best_w
