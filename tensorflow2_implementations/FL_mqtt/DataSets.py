#import mat73
import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
# from tensorflow.keras.utils import to_categorical

class RadarData_mqtt:
    def __init__(self, filepath, filepath2, device_index, start_samples, samples, validation_train, validation_test, random_data_distribution=0):
        # filepath = 'data_mimoradar/data_mmwave_900.mat'
        self.filepath = filepath
        self.device_index = device_index
        self.samples = samples
        self.start_samples = 0
        self.validation_train = validation_train
        self.validation_test = validation_test
        # train data
        # database = sio.loadmat('data/mmwave_data_train_{}.mat'.format(device_index+1))
        database = sio.loadmat(filepath)
        x_train = database['mmwave_data_train_{}'.format(device_index+1)]
        y_train = database['label_train_{}'.format(device_index+1)]
        # y_train_t = to_categorical(y_train)

        if random_data_distribution == 1:
            s_list = random.sample(range(self.validation_train), self.samples)
        else:
            # s_list = np.arange(self.device_index * self.samples, (self.device_index + 1) * self.samples)
            s_list = np.arange(self.start_samples, self.samples + self.start_samples)

        self.x_train = x_train[s_list, :, :]
        self.x_train = (self.x_train.astype('float32').clip(0)) / 1000  # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)

        self.x_train = np.expand_dims(self.x_train, 3) # DATA PARTITION
        self.y_train = np.squeeze(y_train[s_list])
        #test data
        # database = sio.loadmat('data/mmwave_data_test.mat')
        database = sio.loadmat(filepath2)
        x_test = database['mmwave_data_test']
        y_test = database['label_test']
        self.y_test = np.squeeze(y_test[:self.validation_test])
        x_test = (x_test.astype('float32').clip(0)) / 1000
        self.x_test = np.expand_dims(x_test[:self.validation_test, :, :], 3)
        # self.y_test = to_categorical(y_test)

    def getTrainingData(self, batch_size):
        s_list = random.sample(range(self.samples), batch_size)
        batch_xs = self.x_train[s_list, :, :, 0]
        batch_ys = self.y_train[s_list]
        return batch_xs, batch_ys

    def getRandomTestData(self, batch_size):
        s_list = random.sample(range(self.validation_test - 1), batch_size)
        batch_xs = self.x_test[s_list, :, :, 0]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys

    def getTestData(self, batch_size, batch_number):
        s_list = np.arange(batch_number * batch_size, (batch_number + 1) * batch_size)
        batch_xs = self.x_test[s_list, :, :, 0]
        batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys

class RadarData:
    def __init__(self, filepath, device_index, start_samples, samples, validation_train, random_data_distribution=0):
        # filepath = 'data_mimoradar/data_mmwave_900.mat'
        self.filepath = filepath
        self.device_index = device_index
        self.samples = samples
        self.start_samples = start_samples
        self.validation_train = validation_train
        # train data
        database = sio.loadmat('data/data_training_mmwave_900.mat')
        # database = sio.loadmat('dati_mimoradar/data_mmwave_450.mat')
        x_train = database['mmwave_data_train']
        y_train = database['label_train']
        # y_train_t = to_categorical(y_train)

        if random_data_distribution == 1:
            s_list = random.sample(range(self.validation_train), self.samples)
        else:
            # s_list = np.arange(self.device_index * self.samples, (self.device_index + 1) * self.samples)
            s_list = np.arange(self.start_samples, self.samples + self.start_samples)

        self.x_train = x_train[s_list, :, :]
        self.x_train = (self.x_train.astype('float32').clip(0)) / 1000  # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)

        self.x_train = np.expand_dims(self.x_train, 3) # DATA PARTITION
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