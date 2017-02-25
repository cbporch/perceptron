"""
Implementation of a Perceptron Learning Algorithm, that can be abstracted to
various input sizes (NUM) or dimensions (DIM). Displays using pyplot.
"""
from math import ceil
import numpy as np


class Perceptron:

    def __init__(self, dim, num):
        self.DIM = dim
        self.NUM = num
        from graph import Graph
        self.grph = Graph(dim, num)
        self.grph.generate_semicirc_points(5,10,5)

    def update(self, y_t, x):  # Perceptron update function
        r = []  # w(t+1)
        for i in range(self.DIM):
            r.append(self.grph.w[i] + y_t * x[i])
        return r

    def inner_product(self, x):  # Inner Product/Dot Product of vectors w and x
        return sum(self.grph.w[i] * x[i] for i in range(self.DIM))

    def check(self):  # verify if all points are classified correctly
        for n in range(self.NUM):
            if self.grph.y[n] * self.inner_product(self.grph.training_matrix[n]) <= 1:
                return n
        return -1

    def random_check(self):
        misclass = []
        for n in range(self.NUM):
            if self.grph.y[n] * self.inner_product(self.grph.training_matrix[n]) <= 1:
                misclass.append(n)
        if len(misclass) > 0:
            return misclass[ceil(np.random.rand(1) * (len(misclass) - 1))]
        else:
            return -1
