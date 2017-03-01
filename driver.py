from adaline import Adaline
from graph import Graph
from perceptron import Perceptron

DIM = 3
NUM = 1000
grph = Graph(DIM,NUM)
grph.gen_semicirc_points(thk=5, rad=10, sep=5)
per = Perceptron(DIM, NUM, grph)


def run_model(p=per):
    t = 0
    c = True
    while c:
        n = p.random_check()
        if n == -1 or t == 1000:
            c = False
        else:
            p.grph.w = p.update(p.grph.y[n], p.grph.training_matrix[n])
        t += 1
        print("t: {0}, w: {1}".format(t, p.grph.w))
    if p.grph.PLOT:
        p.grph.plot_g()  # In calling g() the 0th value is 1, corresponding to w_0
        p.grph.show_plot()
    # and the last value is not used in calculation, so is set as 0
    return t


def run_trials(i=100):
    j = []
    for x in range(i):
        j.append(run_model())
    per.grph.plt.ylabel("Trials")
    per.grph.plt.xlabel("Updates to converge")
    per.grph.plt.hist(j, bins=30, data=j)
    # plt.show()
    print(j)


def test_etas():
    eta_set = [100, 1, 0.01, 0.0001]
    for e in range(len(eta_set)):
        ada = Adaline.__init(DIM,NUM,e)
        per.grph.run_etas(ada)
        run_model(ada)
        per.grph.show_plot()

run_model()