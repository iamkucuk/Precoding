import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from tensorflow._api.v1 import math, linalg, random

from DataGenerator import DataGenerator


class ComputeDTilda(Layer):
    def __init__(self, d, h, etx = 1, sigma=tf.ones(2, 2)):
        super(ComputeDTilda, self).__init__()
        self.trainable = False
        self.h_tensor = h
        self.etx = etx
        self.sigma = sigma
        self.d_tensor = d

    def compute_output_shape(self, input_shape):
        return self.d_tensor.shape

    def call(self, inputs, **kwargs):
        b_tensor = inputs
        h_tensor = self.h_tensor
        etx = self.etx
        d_tensor = DataGenerator.d_complex
        # self.sigma = sigma
        return self.find_d_tilda(d_tensor, b_tensor, h_tensor, etx)

    def find_d_tilda(self, d_tensor, b_tensor, h_tensor, etx=1, sigma=tf.ones(2, 2)):
        # d_tensor_real = random.normal((2, 1), seed=41)
        # d_tensor_imag = random.normal((2, 1), seed=41)
        # d_tensor = tf.complex(d_tensor_real, d_tensor_imag)
        #
        # b_tensor_real = random.normal((2, 2), seed=42)
        # b_tensor_imag = random.normal((2, 2), seed=42)
        # b_tensor = tf.complex(b_tensor_real, b_tensor_imag)
        energy = math.real(K.sum(linalg.diag_part(linalg.matmul(b_tensor, linalg.adjoint(b_tensor)))))
        b_tensor = math.multiply(b_tensor, tf.complex(math.sqrt(etx / (energy)), 0.0))
        noise_complex_real = random.normal(d_tensor.shape) / math.sqrt(2.0)
        noise_complex_imag = random.normal(d_tensor.shape) / math.sqrt(2.0)
        noise_complex = tf.complex(noise_complex_real, noise_complex_imag)

        # h_real = random.normal(b_tensor.shape, seed=44) / K.sqrt(tf.constant(2.0))
        # h_imag = random.normal(b_tensor.shape, seed=44) / K.sqrt(tf.constant(2.0))
        # h_tensor = tf.complex(h_real, h_imag)
        r_1 = tf.complex(1.0, 0.0) + linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2]) * linalg.adjoint(
            linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2]))
        r_2 = tf.complex(1.0, 0.0) + linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1]) * linalg.adjoint(
            linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1]))

        a_1 = linalg.adjoint(linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1])) * \
              linalg.inv(linalg.matmul(linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1]),
                                       linalg.adjoint(linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1]))) + r_1)

        a_2 = linalg.adjoint(linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2])) * \
              linalg.inv(linalg.matmul(linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2]),
                                       linalg.adjoint(linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2])) + r_2))

        d_tilda_1 = linalg.matmul(a_1, (
                linalg.matmul(linalg.matmul(h_tensor[0:1], b_tensor[:, 0:1]), d_tensor[0:1]) + 0*noise_complex[0:1]))
        d_tilda_2 = linalg.matmul(a_2, (
                linalg.matmul(linalg.matmul(h_tensor[1:2], b_tensor[:, 1:2]), d_tensor[1:2]) + 0*noise_complex[1:2]))

        return tf.concat([d_tilda_1, d_tilda_2], 0)