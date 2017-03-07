import numpy as np
from matplotlib import pyplot as plt
from plotly import graph_objs, plotly
import setapi
import adaline
from sklearn.preprocessing import PolynomialFeatures

setapi.setup()


class Graph:
    w = []  # weights
    y = []
    training_matrix = []
    test_matrix = []
    test_data_size = 1000
    line_start = -15
    line_end = 30
    polyDIM = 3

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

    def g(self, x):  # used to graph selected hypothesis g, which should emulate f with some error
        if self.w[2] == 0:
            return 0
        s = -1 * ((self.w[0] / self.w[2]) + (self.w[1] / self.w[2]) * x)
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
        for i in range(self.NUM):
            angle = np.random.rand() * np.pi
            top_matrix.append([1,
                               np.cos(angle) * (rad + thk * np.random.rand()),
                               np.sin(angle) * (rad + thk * np.random.rand())])
            angle = np.random.rand() * np.pi + np.pi
            bot_matrix.append([1,
                               np.cos(angle) * (rad + thk * np.random.rand()) + (rad + thk / 2),
                               -sep + np.sin(angle) * (rad + thk * np.random.rand())])
        y1 = [1] * self.NUM  # default values for y
        y2 = [-1] * self.NUM
        self.training_matrix = [*top_matrix, *bot_matrix]
        self.y = [*y2, *y1]
        self.plot_points()

    def plot_f(self):
        plt.plot([self.line_start, self.line_end],
                 [self.f(self.line_start), self.f(self.line_end)], 'b', linewidth=0.8)  # plot Target function f(x)

    def plot_g(self):
        plt.plot([self.line_start, self.line_end],
                 [self.g(self.line_start), self.g(self.line_end)], linewidth=0.8)

    def plot_points(self):
        for x in range(len(self.y)):
            if self.y[x] == 1:
                plt.plot(self.training_matrix[x][1], self.training_matrix[x][2], 'Dr', ms=1)
            else:
                plt.plot(self.training_matrix[x][1], self.training_matrix[x][2], 'bo', ms=1)
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

    def plotly_pocket(self, y_err_in):
        t = np.arange(1, len(y_err_in) + 1)
        e_in = y_err_in
        data = [graph_objs.Scatter(
            x=t,
            y=e_in
        )]
        layout = dict(title='Pocket Algorithm: Error over Time',
                      xaxis=dict(title='$Iterations (t)$'),
                      yaxis=dict(title='$E_in$'),
                      )
        fig = dict(data=data, layout=layout)
        # plotly.plot(fig, filename='Problem 3.3 d 2d')
        plt.plot(t, e_in)
        axes = plt.gca()
        x1, x2, y1, y2 = plt.axis()
        axes.set_ylim([0, y2])
        plt.show()

    def get_misclassed(self):
        misclass = []
        for n in range(len(self.y)):
            if self.y[n] * np.inner(self.w, self.training_matrix[n]) <= 0:
                misclass.append(n)
        if len(misclass) > 0:
            return misclass
        else:
            return -1

    def e_in(self):  # get current e_in for weights
        if self.get_misclassed() != -1:
            return len(self.get_misclassed()) / self.NUM
        else:
            return 0

    @staticmethod
    def show_plot():
        plt.show()

    def shade(self):
        pol = PolynomialFeatures(self.polyDIM, include_bias=False)
        x1 = np.arange(min(x[1] for x in self.training_matrix),
                       max(x[1] for x in self.training_matrix), 0.7)
        x2 = np.arange(min(x[2] for x in self.training_matrix),
                       max(x[2] for x in self.training_matrix), 0.7)
        for x_1 in x1:
            for x_2 in x2:
                if 1 == np.sign(np.inner(self.w, pol.fit_transform(np.array([[1, x_1, x_2]])))):
                # if 1 == np.sign(np.inner(self.w, np.array([[1, x_1, x_2]]))):
                    plt.plot(x_1, x_2, 'or', alpha=0.1)
                else:
                    plt.plot(x_1, x_2, 'ob', alpha=0.1)

    def poly(self):
        pol = PolynomialFeatures(self.polyDIM, include_bias=False)
        # m = [[x[1], x[2]] for x in self.training_matrix]
        self.training_matrix = pol.fit_transform(self.training_matrix)
        print(self.training_matrix)
        self.DIM = len(self.training_matrix[0])
        self.w = np.random.rand(self.DIM)
