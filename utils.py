from tensorflow import linalg
from tensorflow import math
from keras import backend as K
import tensorflow as tf


def calc_rate_loss(d_original, d_tilda, weights=None):
    if weights is None:
        weights = [1, 1]

    return math.reduce_sum(weights * linalg.diag_part(
        math.log(math.real(linalg.matmul((d_original - d_tilda), linalg.adjoint(d_original - d_tilda)))) / math.log(
            2.0))) / 4


def rate_loss(weights=None):
    def rate(y_true, y_pred):
        y_true = tf.squeeze(y_true, 0)
        return calc_rate_loss(y_true, y_pred, weights)

    return rate
