import numpy as np
import matplotlib.pyplot as plt


def write_data_to_file(input, file_name):
    data = open(file_name, "a")
    for num in input:
        np.savetxt(data, num)
    data.close()


def plot_AB(iterations, internal_units):
    #plot a
    dataA = np.loadtxt('a_data.txt', delimiter=',').reshape(
        iterations, internal_units)

    _, p = plt.subplots()

    for i in range(iterations):
        plt.plot(dataA[i, :], label='Values after iteration № ' + str(i + 1))

    p.set(xlabel='nth neuron', ylabel='value of a',
          title='Variance of the parameter \'a\' for ' + str(iterations) + ' of IP tuning')
    plt.xticks(np.arange(internal_units))
    p.grid()
    plt.legend(loc="upper right")
    plt.show()

    #plot b
    dataB = np.loadtxt('b_data.txt', delimiter=',').reshape(
        iterations, internal_units)

    _, p = plt.subplots()

    for i in range(iterations):
        plt.plot(dataB[i, :], label='Values after iteration № ' + str(i + 1))

    p.set(xlabel='nth neuron', ylabel='value of b',
          title='Variance of the parameter \'b\' for ' + str(iterations) + ' of IP tuning')
    plt.xticks(np.arange(internal_units))
    p.grid()
    plt.legend(loc="upper right")
    plt.show()
