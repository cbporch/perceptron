from adaline import Adaline
from graph import Graph
from perceptron import Perceptron

DIM = 3
NUM = 1000
grph = Graph(DIM, NUM)
grph.gen_semicirc_points(thk=5, rad=10, sep=0)
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
        ada = Adaline(DIM, NUM, grph, e)
        per.grph.run_etas(ada)
        ada.fit()
        per.grph.show_plot()

