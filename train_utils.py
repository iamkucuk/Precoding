import complexnn
from keras import optimizers, Input
from keras.layers import Lambda, add, concatenate
from keras.models import Model
# from tensorflow._api.v1 import random
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

from ComputeDTilda import ComputeDTilda
from DataGenerator import DataGenerator
from utils import *
import time


def train_for_experiment(snr: float):
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

    loss_fn = rate_loss()

    model.compile(optimizer="adam", loss=loss_fn)

    model.summary()
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    kerasboard = TensorBoard(log_dir="logs/{}db-{}".format(snr, time_str),
                             # write_grads=True,
                             batch_size=1,
                             # embeddings_layer_names=["Lambda"],
                             write_graph=True)
    model.fit_generator(generator=generator,
                        epochs=10,
                        callbacks=[
                            kerasboard,
                            EarlyStopping(monitor="loss", min_delta=1e-6, patience=1),
                            ModelCheckpoint(filepath="models/{}db.hdf5".format(snr),
                                            monitor="loss", save_best_only=True)
                        ])


