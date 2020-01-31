import keras
import numpy as np
import tensorflow as tf


class DataGenerator(keras.utils.Sequence):
    # d_complex = None
###
    def __init__(self, batch_size=256, dim=(2, 2)):
        self.dim = dim
        self.batch_size = batch_size

        target = np.random.randn(2, batch_size * 2).view(np.complex128).astype(np.complex64)
        print(target.shape)
        DataGenerator.d_complex = target

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        return self.__data_generation()

    def __data_generation(self):
        h_real = np.random.normal(size=self.dim) * np.sqrt(1 / 2)
        h_imag = np.random.normal(size=self.dim) * np.sqrt(1 / 2)

        feature = np.expand_dims(np.stack((h_real, h_imag), -1), 0)

        target = np.random.randn(2, self.batch_size * 2).view(np.complex128).astype(np.complex64)
        DataGenerator.d_complex = target

        return feature, np.expand_dims(target, 0)

data = DataGenerator()

a = data.__getitem__(1)
b = 0