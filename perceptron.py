"""
Implementation of a Perceptron Learning Algorithm, that can be abstracted to
various input sizes (NUM) or dimensions (DIM). Displays using pyplot.
"""
from matplotlib import pyplot as plt
import numpy as np


def f(x):  # Target Function
    return 0.5 * x + 1.25

NUM = 20
DIM = 3
PLOT = True if DIM == 3 else PLOT = False
w = np.random.rand(DIM) * 5
matrix = np.random.rand(NUM, DIM) * 5
y = [1] * NUM

for i in range(NUM):
    matrix[i][0] = 1
    if (f(matrix[i][1])) > matrix[i][2]:
        y[i] = -1

if PLOT:
    for x in range(NUM):
        if y[x] == 1:
            plt.plot(matrix[x][1], matrix[x][2], 'Dg')
        else:
            plt.plot(matrix[x][1], matrix[x][2], 'ro')
    plt.plot([0,5], [f(0), f(5)], label='F')  # plot Target function f(x)
    plt.ylabel('Y Axis')
    plt.xlabel('X Axis')


def next_w(w, y, x):
    r = []
    for i in range(DIM):
        r.append(w[i] + y*x[i])
    return r


def sign(k):
    return 1 if float(k) > 0 else -1


def perceptron(w, x):
    return sign(sum(w[i] * x[i] for i in range(DIM)))


def check(y=y, m=matrix):
    for n in range(NUM):
        if y[n] != perceptron(w, m[n]):
            return n
    return -1

t = 0
c = True
while c:
    n = check()
    c = False if n == -1 else w = next_w(w, y[n], matrix[n])
    t += 1
print("t: {0}, w: {1}".format(t,w))

def g(x):
    return ((-w[1]/w[2]) * x) + (-w[0]/w[2])

if PLOT:
    plt.plot([0, 5], [g(0), g(5)])
    plt.show()
