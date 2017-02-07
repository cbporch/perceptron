"""
Implementation of a Perceptron Learning Algorithm, that can be abstracted to
various input sizes (NUM) or dimensions (DIM). Displays using pyplot.
"""
from matplotlib import pyplot as plt
from math import ceil
import numpy as np

NUM = 100  # Sample size
DIM = 11  # Number of Dimensions
w = []
y = []
matrix = []
if DIM == 3:
    PLOT = True  # whether or not to show the plot (only works in 2D)
else:
    PLOT = False


def f(x):  # Target Function
    return 0.5 * x + 1.25


def setup():
    global w, y, matrix
    w = np.random.rand(DIM)  # randomly selected weights
    matrix = np.random.rand(NUM, DIM) * 5  # randomly selected points
    y = [1] * NUM  # default values for y
    for i in range(NUM):
        matrix[i][0] = 1
        if (f(matrix[i][1])) > matrix[i][2]:  # determine if point is above line/hyperplane formed by f()
            y[i] = -1  # correct y[i] if below line
    # if PLOT:
        # for x in range(NUM):
            # if y[x] == 1:
            #     plt.plot(matrix[x][1], matrix[x][2], 'Dg')
            # else:
            #     plt.plot(matrix[x][1], matrix[x][2], 'ro')
        # plt.plot([0,5], [f(0), f(5)], label='F')  # plot Target function f(x)
        # plt.ylabel('X_2 Axis')
        # plt.xlabel('X_1 Axis')


def next_w(w, y, x):  # Perceptron update function
    r = []
    for i in range(DIM):
        r.append(w[i] + y*x[i])
    return r


def sign(k):  # returns 1 if k > 0, -1 otherwise
    return 1 if float(k) > 0 else -1


def perceptron(w, x):  # Inner Product/Dot Product of vectors w and x
    return sign(sum(w[i] * x[i] for i in range(DIM)))


def check():  # verify if all points are classified correctly
    global y, matrix
    for n in range(NUM):
        if y[n] != perceptron(w, matrix[n]):
            return n
    return -1


def random_check():
    global y, matrix
    misclass = []
    for n in range(NUM):
        if y[n] != perceptron(w, matrix[n]):
            misclass.append(n)
    if len(misclass) > 0:
        return misclass[ceil(np.random.rand(1) * (len(misclass) - 1))]
    else:
        return -1


def g(vector_x):  # used to graph selected hypothesis g, which should emulate f with some error
    s = 0
    for i in range(len(vector_x) - 1):
        s += (-w[i]/w[len(w) - 1]) * vector_x[i]
    return s


def run():
    global w
    setup()
    t = 0
    c = True
    while c:
        n = random_check()
        if n == -1:
            c = False
        else:
            w = next_w(w, y[n], matrix[n])
        t += 1
        print("t: {0}, w: {1}".format(t, w))
    # if PLOT:
    #     plt.plot([0, 5], [g([1, 0, 0]), g([1, 5, 0])])  # In calling g(), the 0th value is 1, corresponding to w_0
        #plt.show()                                  # and the last value is not used in calculation, so is set as 0

    return t                                                # This is just to properly display the line formed by g().


def run_trials(i=100):
    j=[]
    for x in range(i):
        j.append(run())
    plt.ylabel("Trials")
    plt.xlabel("Updates to converge")
    plt.hist(j, bins=40,data=j)
    plt.show()
    print(j)

run_trials()