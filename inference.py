import complexnn
# from tensorflow._api.v1 import random
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.layers import Lambda, concatenate
from keras.models import Model

from ComputeDTilda import ComputeDTilda
from DataGenerator import DataGenerator

from utils import calc_rate, rate_loss, rate_dummy_loss


def inference_model(snr):
    K.clear_session()
    generator = DataGenerator()
    input_tensor = Input(batch_shape=(1, 2, 2, 2))
    x_1 = complexnn.conv.ComplexConv2D(8, (3, 3), activation="relu", transposed=True)(input_tensor)
    # # x = complexnn.bn.ComplexBatchNormalization()(x)
    x = complexnn.conv.ComplexConv2D(16, (5, 5), activation="relu", transposed=True)(x_1)
    x = complexnn.bn.ComplexBatchNormalization()(x)
    # # x = complexnn.conv.ComplexConv2D(16, (4, 4), activation="relu")(x)
    # x = complexnn.conv.ComplexConv2D(32, (7, 7), activation="relu", transposed=True)(x)
    #
    # x = complexnn.conv.ComplexConv2D(16, (7, 7), activation="relu")(x)
    # # # # x = complexnn.bn.ComplexBatchNormalization()(x)
    x = complexnn.conv.ComplexConv2D(8, (5, 5), activation="relu")(x)
    x = complexnn.bn.ComplexBatchNormalization()(x)
    x = concatenate([x_1, x])
    x = complexnn.conv.ComplexConv2D(1, (3, 3), activation="sigmoid")(x)

    b_tensor = Lambda(lambda x: K.squeeze(tf.complex(x[:, :, 0], x[:, :, 1]), 0))(x)

    output_tensor = ComputeDTilda(DataGenerator.d_complex,
                                  tf.squeeze(
                                      tf.complex(input_tensor[:, :, 0], input_tensor[:, :, 1]
                                                 ), 0), db=snr)(b_tensor)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])

    loss_fn = rate_dummy_loss()

    model_name = "{}db.hdf5".format(float(snr))

    model.load_weights("models/-10.0db.hdf5".format(float(snr)))

    model.compile(optimizer="adam", loss=loss_fn)

    out = model.evaluate_generator(generator, verbose=0)

    return out

rates = []

for snr in range(-10, 31, 5):
    rates.append(inference_model(snr))



