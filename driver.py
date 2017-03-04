from adaline import Adaline
from graph import Graph
from perceptron import Perceptron
import numpy as np

DIM = 3
NUM = 1000
grph = Graph(DIM, NUM)
grph.gen_semicirc_points(thk=5, rad=10, sep=-5)
per = Perceptron(DIM, NUM, grph)
# t, w = per.pocket_fit()
# print(w)


def sep_change():
    for sep in np.arange(5, 5.2, 0.2):
        print("============================")
        print(sep)
        grph.gen_semicirc_points(thk=5, rad=10, sep=sep)
        per = Perceptron(DIM, NUM, grph)
        per.fit()


def run_trials(i=100):
    j = []
    for x in range(i):
        j.append(per.fit())
    per.grph.plt.ylabel("Trials")
    per.grph.plt.xlabel("Updates to converge")
    per.grph.plt.hist(j, bins=30, data=j)
    # plt.show()
    print(j)


def test_etas():
    eta_set = [100, 1, 0.01, 0.0001]
    for e in range(len(eta_set)):
        ada = Adaline(DIM, NUM, grph, e, 1000)
        per.grph.run_etas(ada)
        ada.fit()
        per.grph.show_plot()


def linear_regression():
    mat = np.array(per.grph.training_matrix)
    print(np.linalg.pinv(mat))
    w = np.inner(np.linalg.pinv(mat), per.grph.y)
    print(w)
    setattr(per.grph, 'w', w)
    grph.plot_g()
    print(grph.e_in())

linear_regression()
grph.show_plot()
# print("{1}x + {0})".format(-grph.w[0]/grph.w[2], -grph.w[1]/grph.w[2]))
#
# per.fit()
# print("{1}x + {0})".format(-grph.w[0]/grph.w[2], -grph.w[1]/grph.w[2]))
