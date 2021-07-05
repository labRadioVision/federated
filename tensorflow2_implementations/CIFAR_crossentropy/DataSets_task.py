#import mat73
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
# from tensorflow.keras.utils import to_categorical
# choose a number of classes per node (<10), by num_class_per_node, randomly for the selected device and assigns self.samples samples to it
class CIFARData_task:
    def __init__(self, device_index, start_samples, samples, validation_train, num_class_per_node=8):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # classes to train
        num_class_per_node = 8
        self.validation_train = 50000
        self.validation_test = 10000
        classes_per_node = random.sample(range(10), num_class_per_node)
        # print(classes_per_node)
        ra = np.arange(self.validation_train)
        vec_list = []
        for q in range(num_class_per_node):
            mask = np.squeeze((y_train == classes_per_node[q]))
            ctr = ra[mask]
            for qq in range(ctr.size):
                vec_list.append(ctr[qq])

        #x_train_sub = x_train[mask]
        #y_train_sub = y_train[mask]
        self.device_index = device_index
        self.samples = samples
        self.start_samples = start_samples

        # print(vec_list)
        s_list = random.sample(vec_list, self.samples)

        # self.x_train = np.expand_dims(x_train[s_list, :, :], 3) # DATA PARTITION
        self.x_train = x_train[s_list, :, :, :]  # DATA PARTITION
        self.x_train = (self.x_train.astype('float32').clip(0)) / 255
        self.y_train = np.squeeze(y_train[s_list])
        # print(self.y_train)
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
