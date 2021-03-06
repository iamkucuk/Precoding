import numpy as np
import numpy.linalg as linalg
import numpy.random as random

b_prev = np.zeros((2, 2)).astype(np.complex)
b_current = b_prev

h_real = np.random.normal(size=(2, 2)) * np.sqrt(1 / 2)
h_imag = np.random.normal(size=(2, 2)) * np.sqrt(1 / 2)
H = h_real + 1.j * h_imag

# z =

while True:
    ##### Part I

    r_1 = 1 + np.matmul(H[1:2], b_prev[:, 1:2]) * (
        np.matmul(np.conj(H[1:2].T), np.conj(b_prev[:, 1:2]).T))
    
    r_2 = 1 + np.matmul(H[0:1], b_prev[:, 0:1]) * (
        np.matmul(np.conj(H[0:1].T), np.conj(b_prev[:, 0:1]).T))

    a_1 = (np.matmul(np.conj(H[0:1].T), np.conj(b_prev[:, 0:1].T))) * \
          linalg.inv(
              np.matmul(np.matmul(np.matmul(H[0:1], b_prev[:, 0:1]), np.conj(b_prev[:, 0:1].T)), np.conj(H[:, 0:1].T))
              + r_1)

    a_2 = (np.matmul(np.conj(H[1:2].T), np.conj(b_prev[:, 1:2].T))) * \
          linalg.inv(
              np.matmul(np.matmul(np.matmul(H[1:2], b_prev[:, 1:2]), np.conj(b_prev[:, 1:2].T)), np.conj(H[:, 1:2].T))
              + r_2)

    ##### Part II

    noise_complex = np.random.multivariate_normal(np.zeros(2), 0.5 * np.eye(2), size=500).view(np.complex128)

    e_1 = linalg.inv(1 + np.matmul(
        np.matmul(
            np.matmul(
                np.matmul(
                    np.conj(b_prev[:, 0:1]).T, np.conj(H[0:1].T)), r_1 ** (-1)), H[0:1]), b_prev[:, 0:1]))

    e_2 = linalg.inv(1 + np.matmul(
        np.matmul(
            np.matmul(
                np.matmul(
                    np.conj(b_prev[:, 1:2]).T, np.conj(H[1:2].T)), r_2 ** (-1)), H[1:2]), b_prev[:, 1:2]))

    w_1 = 1 * linalg.inv(e_1)
    w_2 = 1 * linalg.inv(e_2)

    abuzer = 0