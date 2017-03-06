import adaline
from graph import Graph
from perceptron import Perceptron
import numpy as np

DIM = 3
NUM = 1000
grph = Graph(DIM, NUM)
grph.gen_semicirc_points(thk=5, rad=10, sep=-5)
per = Perceptron(grph)


def sep_change():
    for sep in np.arange(5, 5.2, 0.2):
        print("============================")
        print(sep)
        grph.gen_semicirc_points(thk=5, rad=10, sep=sep)
        p = Perceptron(grph)
        p.fit()


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
        ada = adaline.Adaline(grph, e, 1000)
        per.grph.run_etas(ada)
        ada.fit()
        per.grph.show_plot()


def linear_regression():
    mat = np.array(per.grph.training_matrix)
    w = np.inner(np.linalg.pinv(mat), per.grph.y)
    print(w)
    setattr(per.grph, 'w', w)
    grph.plot_g()
    print(grph.e_in())


# grph.show_plot()
original = grph.training_matrix
grph.poly()
# linear_regression()
setattr(per, 'grph', grph)
grph.show_plot()  # clear graph
per.pocket_fit()
print("pocket done")
setattr(grph, 'training_matrix', original)
grph.shade()
print("shade done")

grph.plot_points()
per.grph.show_plot()
