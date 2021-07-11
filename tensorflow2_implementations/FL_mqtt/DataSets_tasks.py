#import mat73
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
# from tensorflow.keras.utils import to_categorical

class RadarData_tasks:
    def __init__(self, filepath, device_index, start_samples, samples, validation_train, num_class_per_node=4):
        # filepath = 'data_mimoradar/data_mmwave_900.mat'
        self.filepath = filepath
        self.device_index = device_index
        self.samples = samples
        self.start_samples = start_samples
        self.validation_train = validation_train
        # train data
        # database = sio.loadmat(self.filepath)
        database = sio.loadmat('data/data_training_mmwave_900.mat')
        x_train = database['mmwave_data_train']
        y_train = database['label_train']
        # y_train_t = to_categorical(y_train)
        x_train = (x_train.astype('float32').clip(0)) / 1000  # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)

        num_class_per_node = 4
        classes_per_node = random.sample(range(6), num_class_per_node)
        # print(classes_per_node)
        ra = np.arange(self.validation_train)
        vec_list = []
        for q in range(num_class_per_node):
            mask = np.squeeze((y_train == classes_per_node[q]))
            ctr = ra[mask]
            for qq in range(ctr.size):
                vec_list.append(ctr[qq])

        # x_train_sub = x_train[mask]
        # y_train_sub = y_train[mask]

        # print(vec_list)
        s_list = random.sample(vec_list, self.samples)

        self.x_train = np.expand_dims(x_train[s_list, :, :], 3) # DATA PARTITION
        self.y_train = np.squeeze(y_train[s_list])
        #test data
        database = sio.loadmat('data/data_validation_mmwave_900.mat')
        x_test = database['mmwave_data_test']
        y_test = database['label_test']
        self.y_test = np.squeeze(y_test[:self.validation_train])
        x_test = (x_test.astype('float32').clip(0)) / 1000
        self.x_test = np.expand_dims(x_test[:self.validation_train, :, :], 3)
        # self.y_test = to_categorical(y_test)

    def getTrainingData(self, batch_size):
        s_list = random.sample(range(self.samples), batch_size)
        batch_xs = self.x_train[s_list, :, :, 0]
        batch_ys = self.y_train[s_list]
        return batch_xs, batch_ys

    def getRandomTestData(self, batch_size):
        s_list = random.sample(range(self.validation_train - 1), batch_size)
        batch_xs = self.x_test[s_list, :, :, 0]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys

    def getTestData(self, batch_size, batch_number):
        s_list = np.arange(batch_number * batch_size, (batch_number + 1) * batch_size)
        batch_xs = self.x_test[s_list, :, :, 0]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys