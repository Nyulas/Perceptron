import matplotlib.pyplot as plt
import time
import numpy as item


data = item.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
labels = item.array([0, 0, 0, 1])


def hardlim(val):
    return 0 if val < 0 else 1

def perceptron_learning(data, result):

    N, n = data.shape
    lr = .1
    w = item.random.randn(n, 1)
    E = 1

    plt.ion()
    figure = plt.figure()
    figure.suptitle('AND')
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.grid()
    plt.scatter(data[:, 1], data[:, 2])

    x = item.linspace(-5, 5, 50)



    while E != 0:
        E = 0

        for i in range(N):
            yi = item.sign(hardlim(data[i], w))
            ei = labels[i] - yi
            w += lr * ei * data[i].reshape(n, 1)
            E += ei ** 2

        print(w)
        a = [0, -w[0] / w[2]]
        c = [-w[0] / w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])

        line, = plt.plot(x, x * m + a[1])
        line.set_ydata(x * m + a[1])
        figure.canvas.draw()
        time.sleep(.5)
        line.remove()
        figure.canvas.flush_events()


if __name__=="__main__":

    perceptron_learning(data, labels)
