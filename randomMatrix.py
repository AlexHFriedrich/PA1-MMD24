from random import randint
import numpy as np


def random_value():
    """
    Generate a random value for the matrix according to the scheme introduced in the assignment
    :return: random value
    """
    val = randint(1, 6)
    if val == 1:
        return np.sqrt(3)
    elif val == 6:
        return -np.sqrt(3)
    else:
        return 0


def random_matrix(n, m):
    """
    Generate a random matrix of size n x m
    :param n: feature dimension of the data
    :param m: depth of the hash table
    :return:
    """
    return np.array([[random_value() for _ in range(m)] for _ in range(n)])


if __name__ == "__main__":
    rand_m = random_matrix(5, 3)
    print(rand_m.shape)