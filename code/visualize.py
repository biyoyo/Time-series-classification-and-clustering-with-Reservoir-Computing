import numpy as np
import matplotlib.pyplot as plt


def write_data_to_file(input, file):
    for num in input:
        np.savetxt(file, num)


def plot_AB(iterations, internal_units):
    #plot a
    dataA = np.loadtxt('a_data.txt', delimiter=',').reshape(
        iterations, internal_units)

    _, p = plt.subplots()
    x = np.arange(internal_units)
    for i in range(iterations):
        plt.scatter(x, dataA[i, :],
                    label='Values after iteration № ' + str(i + 1), s=10)

    p.set(xlabel='nth neuron', ylabel='value of a',
          title='Variance of the parameter \'a\' for ' + str(iterations) + ' iterations of IP tuning')

    plt.legend(loc="upper right")
    plt.show()

    #plot b
    dataB = np.loadtxt('b_data.txt', delimiter=',').reshape(
        iterations, internal_units)

    _, p = plt.subplots()

    x = np.arange(internal_units)
    for i in range(iterations):
        plt.scatter(x, dataB[i, :],
                 label='Values after iteration № ' + str(i + 1), s=10)

    p.set(xlabel='nth neuron', ylabel='value of b',
          title='Variance of the parameter \'b\' for ' + str(iterations) + ' iterations of IP tuning')

    plt.legend(loc="upper right")
    plt.show()
