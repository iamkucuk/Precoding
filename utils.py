from tensorflow import linalg
from tensorflow import math
from keras import backend as K
import tensorflow as tf


def calc_rate_loss(d_original, d_tilda, weights=None):
    if weights is None:
        weights = [1, 1]

    return math.reduce_sum(weights * linalg.diag_part(
        math.log(math.real(linalg.matmul((d_original - d_tilda), linalg.adjoint(d_original - d_tilda)))) / math.log(
            2.0))) / 256


def rate_loss(weights=None):
    def rate(y_true, y_pred):
        y_true = tf.squeeze(y_true, 0)
        return calc_rate_loss(y_true, y_pred, weights)

    return rate


def rate_dummy_loss(weights=None):
    def rate(y_true, y_pred):
        y_true = tf.squeeze(y_true, 0)
        return calc_rate(y_true, y_pred, weights)

    return rate


def calc_rate(d_original, d_tilda, weights=None):
    if weights is None:
        weights = [1, 1]

    E_1 = linalg.matmul((d_original[0:1] - d_tilda[0:1]), linalg.adjoint(d_original[0:1] - d_tilda[0:1]))
    E_1 /= 256
    E_2 = linalg.matmul((d_original[1:2] - d_tilda[1:2]), linalg.adjoint(d_original[1:2] - d_tilda[1:2]))
    E_2 /= 256

    R_1 = linalg.logdet(linalg.pinv(E_1)) / math.log(2)
    R_2 = linalg.logdet(linalg.pinv(E_2)) / math.log(2)

    return R_1 + R_2
