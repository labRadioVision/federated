from __future__ import absolute_import, division, print_function, unicode_literals
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import multiprocessing
import math
from matplotlib.pyplot import pause
import os
import glob

class CFA_process:
    # sets neighbor indexes for k-regular networks (number of neighbors is 'neighbors'
    def get_connectivity(self, ii_saved_local, neighbors, devices):
        if (ii_saved_local == 0):
            sets_neighbors_final = np.arange(ii_saved_local + 1, ii_saved_local + neighbors + 1)
        elif (ii_saved_local == devices - 1):
            sets_neighbors_final = np.arange(ii_saved_local - neighbors, ii_saved_local)
        elif (ii_saved_local >= math.ceil(neighbors / 2)) and (
                ii_saved_local <= devices - math.ceil(neighbors / 2) - 1):
            sets_neighbors = np.arange(ii_saved_local - math.floor(neighbors / 2),
                                       ii_saved_local + math.floor(neighbors / 2) + 1)
            index_ii = np.where(sets_neighbors == ii_saved_local)
            sets_neighbors_final = np.delete(sets_neighbors, index_ii)
        else:
            if (ii_saved_local - math.ceil(neighbors / 2) < 0):
                sets_neighbors = np.arange(0, neighbors + 1)
            else:
                sets_neighbors = np.arange(devices - neighbors - 1, devices)
            index_ii = np.where(sets_neighbors == ii_saved_local)
            sets_neighbors_final = np.delete(sets_neighbors, index_ii)
        return sets_neighbors_final

    # compute weights for CFA
    def federated_weights_computing2(self, filename, filename2, ii, ii2, epoch, devices, neighbors, eps_t_control):
        saved_epoch = epoch
        b_v = 1 / devices
        # eps_t_control = 1 #from paper
        while not os.path.isfile(filename2):
            print('Waiting..')
            pause(1)

        try:
            mathcontent = sio.loadmat(filename2)
        except:
            print('Detected problem while loading file')
            pause(3)
            mathcontent = sio.loadmat(filename2)

        weights_current_l1 = mathcontent['weights1']
        biases_current_l1 = mathcontent['biases1']
        weights_current_l2 = mathcontent['weights2']
        biases_current_l2 = mathcontent['biases2']

        while not os.path.isfile(filename):
            print('Waiting..')
            pause(1)

        try:
            mathcontent = sio.loadmat(filename)
        except:
            print('Detected problem while loading file')
            pause(3)
            mathcontent = sio.loadmat(filename)

        balancing_vect = np.ones(devices) * b_v
        weight_factor = (balancing_vect[ii2] / (
                    balancing_vect[ii2] + (neighbors - 1) * balancing_vect[ii]))  # equation (11) from paper
        updated_weights_l1 = weights_current_l1 + eps_t_control * weight_factor * (
                    mathcontent['weights1'] - weights_current_l1)  # see paper section 3
        updated_biases_l1 = biases_current_l1 + eps_t_control * weight_factor * (
                    mathcontent['biases1'] - biases_current_l1)
        updated_weights_l2 = weights_current_l2 + eps_t_control * weight_factor * (
                    mathcontent['weights2'] - weights_current_l2)  # see paper section 3
        updated_biases_l2 = biases_current_l2 + eps_t_control * weight_factor * (
                    mathcontent['biases2'] - biases_current_l2)

        weights_l1 = updated_weights_l1
        biases_l1 = updated_biases_l1
        weights_l2 = updated_weights_l2
        biases_l2 = updated_biases_l2

        try:
            sio.savemat('temp_datamat{}_{}.mat'.format(ii, saved_epoch), {
                "weights1": weights_l1, "biases1": biases_l1, "weights2": weights_l2, "biases2": biases_l2})
            mathcontent = sio.loadmat('temp_datamat{}_{}.mat'.format(ii, saved_epoch))
        except:
            print('Unable to save file .. retrying')
            pause(3)
            print(biases)
            sio.savemat('temp_datamat{}_{}.mat'.format(ii, saved_epoch), {
                "weights1": weights_l1, "biases1": biases_l1, "weights2": weights_l2, "biases2": biases_l2})
        return weights_l1, biases_l1, weights_l2, biases_l2

    def __init__(self, federated, devices, ii_saved_local, neighbors):
        self.federated = federated # true for federation active
        self.devices = devices # number of devices
        self.ii_saved_local = ii_saved_local # device index
        self.neighbors = neighbors # neighbors number (given the network topology)
        self.neighbor_vec = self.get_connectivity(ii_saved_local, neighbors, devices) # neighbor list

    def getFederatedWeight(self, n_W_l1, n_W_l2, n_b_l1, n_b_l2, epoch, v_loss, eps_t_control):
        if (self.federated):
            if self.devices > 1:  # multihop topology
                if epoch == 0:
                    sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch, "loss_sample": v_loss})
                    W_up_l1 = n_W_l1
                    n_up_l1 = n_b_l1
                    W_up_l2 = n_W_l2
                    n_up_l2 = n_b_l2
                else:
                    sio.savemat('temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch, "loss_sample": v_loss})
                    # neighbor_vec = get_connectivity(self.ii_saved_local, self.neighbors, self.devices)
                    for neighbor_index in range(self.neighbor_vec.size):
                        while not os.path.isfile(
                                'datamat{}_{}.mat'.format(self.neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                                'temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch)):
                            print('Waiting for datamat{}_{}.mat'.format(self.ii_saved_local - 1, epoch - 1))
                            pause(1)
                        [W_up_l1, n_up_l1, W_up_l2, n_up_l2] = self.federated_weights_computing2(
                            'datamat{}_{}.mat'.format(self.neighbor_vec[neighbor_index], epoch - 1),
                            'temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch), self.ii_saved_local,
                            self.neighbor_vec[neighbor_index],
                            epoch, self.devices, self.neighbors, eps_t_control)
                        pause(5)

                    W_up_l1 = np.asarray(W_up_l1)
                    n_up_l1 = np.squeeze(np.asarray(n_up_l1))
                    W_up_l2 = np.asarray(W_up_l2)
                    n_up_l2 = np.squeeze(np.asarray(n_up_l2))

        else:
            W_up_l1 = n_W_l1
            n_up_l1 = n_b_l1
            W_up_l2 = n_W_l2
            n_up_l2 = n_b_l2
        return W_up_l1, n_up_l1, W_up_l2, n_up_l2
