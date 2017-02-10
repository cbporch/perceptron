"""
Implementation of a Perceptron Learning Algorithm, that can be abstracted to
various input sizes (NUM) or dimensions (DIM). Displays using pyplot.
"""
from matplotlib import pyplot as plt
from math import ceil
import numpy as np

NUM = 100  # Sample size
DIM = 3  # Number of Dimensions
ETA = 0.0001  # used for Adaline update
test_data_size = 10000
w = []
y = []
training_matrix = []
test_matrix = []
if DIM == 3:
    PLOT = True  # whether or not to show the plot (only works in 2D)
else:
    PLOT = False


def f(x):  # Target Function
    return 0.5 * x + 1.25


def setup():
    global w, y, training_matrix, test_matrix
    w = np.random.rand(DIM)  # randomly selected weights
    training_matrix = np.random.rand(NUM, DIM) * 10  # randomly selected points
    test_matrix = np.random.rand(test_data_size, DIM)*10
    y = [1] * NUM  # default values for y
    for i in range(NUM):
        training_matrix[i][0] = 1
        if (f(training_matrix[i][1])) > training_matrix[i][2]:  # determine if point is above line formed by f()
            y[i] = -1  # correct y[i] if below line
    if PLOT:
        plot_points()


def plot_points():
    for x in range(NUM):
        if y[x] == 1:
            plt.plot(training_matrix[x][1], training_matrix[x][2], 'Dg')
        else:
            plt.plot(training_matrix[x][1], training_matrix[x][2], 'ro')
    plt.plot([0, 10], [f(0), f(10)], 'b')  # plot Target function f(x)
    plt.ylabel('X_2 Axis')
    plt.xlabel('X_1 Axis')


def perceptron_update(y_t, x):  # Perceptron update function
    r = []  # w(t+1)
    for i in range(DIM):
        r.append(w[i] + y_t * x[i])
    return r


def adaline_update(y_t, x):
    global w
    r = []
    s_t = sign(inner_product(x))
    for i in range(DIM):
        r.append(w[i] + (ETA * (y_t - s_t) * x[i]))
    return r


def sign(k):  # returns 1 if k > 0, -1 otherwise
    return 1 if float(k) > 0 else -1


def inner_product(x):  # Inner Product/Dot Product of vectors w and x
    return sum(w[i] * x[i] for i in range(DIM))


def check():  # verify if all points are classified correctly
    global y, training_matrix
    for n in range(NUM):
        if y[n]*inner_product(training_matrix[n]) <= 1:
            return n
    return -1


def random_check():
    global y, training_matrix
    misclass = []
    for n in range(NUM):
        if y[n]*inner_product(training_matrix[n]) <= 1:
            misclass.append(n)
    if len(misclass) > 0:
        return misclass[ceil(np.random.rand(1) * (len(misclass) - 1))]
    else:
        return -1


def g(vector_x):  # used to graph selected hypothesis g, which should emulate f with some error
    global w
    s = 0
    for i in range(len(vector_x) - 1):
        s += (-w[i]/w[len(w) - 1]) * vector_x[i]
    return s


def run_perceptron():
    global w
    # setup()
    t = 0
    c = True
    while c:
        n = random_check()
        if n == -1 or t == 1000:
            c = False
        else:
            w = adaline_update(y[n], training_matrix[n])
        t += 1
        print("t: {0}, w: {1}".format(t, w))
    if PLOT:
        plt.plot([0, 10], [g([1, 0, 0]), g([1, 10, 0])], 'g')  # In calling g() the 0th value is 1, corresponding to w_0
    # plt.show()                                      # and the last value is not used in calculation, so is set as 0
    compare()
    return t


def run_trials(i=100):
    j = []
    for x in range(i):
        j.append(run_perceptron())
    plt.ylabel("Trials")
    plt.xlabel("Updates to converge")
    plt.hist(j, bins=30, data=j)
    # plt.show()
    print(j)


def compare():
    error = 0
    ftest_y = [1]*test_data_size
    gtest_y = [1]*test_data_size
    for i in range(test_data_size):
        test_matrix[i][0] = 1
        if f(test_matrix[i][1]) > test_matrix[i][2]:
            ftest_y[i] = -1
        if g(test_matrix[i]) > test_matrix[i][2]:
            gtest_y[i] = -1
        if gtest_y[i] != ftest_y[i]:
            error += 1
            plt.plot(test_matrix[i][1], test_matrix[i][2], 'Dy')
    print("errors: {0}%, eta: {1}".format(100 * error/test_data_size, ETA))
    plt.show()
    return error

eta_set = [100, 1, 0.01, 0.0001]
setup()
for e in range(len(eta_set)):
    ETA = eta_set[e]
    plot_points()
    w = np.random.rand(DIM)
    run_perceptron()
    plt.show()
