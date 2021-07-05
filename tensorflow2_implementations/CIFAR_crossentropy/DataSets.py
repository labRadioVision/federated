#import mat73
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
#import threading
# from tensorflow.keras.utils import to_categorical
# print_lock = threading.Lock()

class CIFARData:
    def __init__(self, device_index, start_samples, samples, validation_train, random_data_distribution=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.device_index = device_index
        self.samples = samples
        self.start_samples = start_samples
        self.validation_train = 50000
        self.validation_test = 10000

        if random_data_distribution == 1:
            s_list = random.sample(range(self.validation_train), self.samples)
        else:
            # s_list = np.arange(self.device_index * self.samples, (self.device_index + 1) * self.samples)
            s_list = np.arange(self.start_samples, self.samples + self.start_samples)

        # self.x_train = np.expand_dims(x_train[s_list, :, :], 3) # DATA PARTITION
        self.x_train = x_train[s_list, :, :, :]  # DATA PARTITION
        self.x_train = (self.x_train.astype('float32').clip(0)) / 255
        self.y_train = np.squeeze(y_train[s_list])

        self.y_test = np.squeeze(y_test[:self.validation_test])
        self.x_test = x_test[:self.validation_test, :, :, :]
        self.x_test = (self.x_test.astype('float32').clip(0)) / 255
        del x_test, x_train, y_test, y_train

    def getTrainingData(self, batch_size):
        s_list = random.sample(range(self.samples), batch_size)
        batch_xs = self.x_train[s_list, :, :, :]
        batch_ys = self.y_train[s_list]
        return batch_xs, batch_ys

    def getRandomTestData(self, batch_size):
        s_list = random.sample(range(self.validation_test - 1), batch_size)
        batch_xs = self.x_test[s_list, :, :, :]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys

    def getTestData(self, batch_size, batch_number):
        s_list = np.arange(batch_number * batch_size, (batch_number + 1) * batch_size)
        batch_xs = self.x_test[s_list, :, :, :]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys