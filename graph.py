import numpy as np
from matplotlib import pyplot as plt

import adaline


class Graph:
    w = []
    y = []
    training_matrix = []
    test_matrix = []
    test_data_size = 1000
    PLOT = False
    line_start = -10
    line_end = 25

    def __init__(self, d, n):
        self.NUM = n
        self.DIM = d
        if self.DIM == 3:
            self.PLOT = True  # whether or not to show the plot (only works in 2D)
        else:
            self.PLOT = False

    @staticmethod
    def f(x):  # Target Function
        return 0.5 * x + 1.25

    def g(self, vector_x):  # used to graph selected hypothesis g, which should emulate f with some error
        s = 0
        for i in range(len(vector_x) - 1):
            s += (-1 * self.w[i] / self.w[len(self.w) - 1]) * vector_x[i]
        return s

    def gen_points(self):
        self.w = np.random.rand(self.DIM)  # randomly selected weights
        self.training_matrix = np.random.rand(self.NUM, self.DIM) * 10  # randomly selected points
        self.test_matrix = np.random.rand(self.test_data_size, self.DIM) * 10
        self.y = [1] * self.NUM  # default values for y
        for i in range(self.NUM):
            self.training_matrix[i][0] = 1
            if (self.f(self.training_matrix[i][1])) > self.training_matrix[i][2]:
                # determine if point is above line formed by f()
                self.y[i] = -1  # correct y[i] if below line
        if self.PLOT:
            self.plot_f()
            self.plot_points()

    def gen_semicirc_points(self, thk, rad, sep):
        # self.w = np.random.rand(self.DIM)  # randomly selected weights
        self.w = [0] * self.DIM
        top_matrix = []
        bot_matrix = []
        for i in range(int(self.NUM / 2)):
            angle = np.random.rand() * np.pi
            top_matrix.append([1,
                               np.cos(angle) * (rad + thk * np.random.rand()),
                               np.sin(angle) * (rad + thk * np.random.rand())])
            angle = np.random.rand() * np.pi + np.pi
            bot_matrix.append([1,
                               np.cos(angle) * (rad + thk * np.random.rand()) + (rad + thk/2),
                               -sep + np.sin(angle) * (rad + thk * np.random.rand())])
        y1 = [1] * int(self.NUM / 2)  # default values for y
        y2 = [-1] * int(self.NUM / 2)
        self.training_matrix = top_matrix + bot_matrix
        self.y = [*y1, *y2]
        self.plot_points()

    def plot_f(self):
        plt.plot([self.line_start, self.line_end],
                 [self.f(self.line_start), self.f(self.line_end)], 'b', linewidth=0.8)  # plot Target function f(x)

    def plot_g(self):
        plt.plot([self.line_start, self.line_end],
                 [self.g([1, self.line_start, 0]), self.g([1, self.line_end, 0])], 'g', linewidth=0.8)

    def plot_points(self):
        for x in range(len(self.y)):
            if self.y[x] == 1:
                plt.plot(self.training_matrix[x][1], self.training_matrix[x][2], 'Dr', ms=0.5)
            else:
                plt.plot(self.training_matrix[x][1], self.training_matrix[x][2], 'bo', ms=0.5)

        plt.ylabel('X_2 Axis')
        plt.xlabel('X_1 Axis')

    def show_adaline_error(self, ada):
        assert isinstance(ada, adaline.Adaline)

        data_size = self.test_data_size
        m = self.test_matrix
        error = 0
        ftest_y = [1] * data_size
        gtest_y = [1] * data_size
        for i in range(data_size):
            m[i][0] = 1
            if self.f(m[i][1]) > m[i][2]:
                ftest_y[i] = -1
            if self.g(m[i]) > m[i][2]:
                gtest_y[i] = -1
            if gtest_y[i] != ftest_y[i]:
                error += 1
                plt.plot(m[i][1], m[i][2], 'Dy')
        print("errors: {0}%, eta: {1}".format(100 * error / data_size, ada.ETA))
        plt.show()
        return error

    def run_etas(self, ada):
        self.plot_points()
        self.w = np.random.rand(ada.DIM)

    @staticmethod
    def show_plot():
        plt.show()
