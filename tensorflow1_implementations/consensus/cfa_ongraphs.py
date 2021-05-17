from __future__ import absolute_import, division, print_function, unicode_literals
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import math
import time
from matplotlib.pyplot import pause
import os
import glob
import random

# randomized neighbors

class CFA_process:
    # sets neighbor indexes for k-regular networks (number of neighbors is 'neighbors'
    def getRandomNetwork_connectivity(self, ii_saved_local, neighbors, devices, epoch):
        sets_neighbors_final = np.zeros(neighbors, dtype=int)
        # for k in range(neighbors):
        #    sets_neighbors_final[k] = np.random.randint(devices, size=1)
        #    while sets_neighbors_final[k] == ii_saved_local:
        #        sets_neighbors_final[k] = np.random.randint(devices, size=1)
        perm2 = np.random.permutation(devices)
        sets_neighbors_final = perm2[0:neighbors]
        check = np.where(sets_neighbors_final == ii_saved_local)
        while check[0].size:
            perm2 = np.random.permutation(devices)
            sets_neighbors_final = perm2[0:neighbors]
            check = np.where(sets_neighbors_final == ii_saved_local)
        return sets_neighbors_final

    def getMobileNetwork_connectivity(self, ii_saved_local, max_neighbors, devices, graph):
        graph_index = sio.loadmat('consensus/vGraph.mat')
        dev = np.arange(1, devices + 1)
        graph_mobile = graph_index['graph']
        set = graph_mobile[ii_saved_local, :, graph]
        tot_neighbors = np.sum(set, dtype=np.uint8)
        sets_neighbors_final = np.zeros(tot_neighbors, dtype=np.uint8)
        counter = 0
        for kk in range(devices):
            if set[kk] == 1:
                sets_neighbors_final[counter] = kk
                counter = counter + 1
        # choose randomly up to max_neighbors
        if sets_neighbors_final.size > max_neighbors:
            neighbor_list = random.choices(sets_neighbors_final, k=max_neighbors)
            neighbor_list = np.asarray(neighbor_list)
        else:
            neighbor_list = sets_neighbors_final

        return neighbor_list

    def get_connectivity(self, ii_saved_local, neighbors, devices): # k connected static
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
        # eps_t_control = 1 # from paper
        pause(2)
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

        start_time = time.time()
        while not os.path.isfile(filename):
            print('Waiting..')
            pause(1)

        try:
            mathcontent = sio.loadmat(filename)
        except:
            print('Detected problem while loading file')
            pause(3)
            mathcontent = sio.loadmat(filename)

        parameters_received = mathcontent['counter_param']
        balancing_vect = np.ones(devices) * b_v
        weight_factor = (balancing_vect[ii2] / (
                    balancing_vect[ii2] + (neighbors) * balancing_vect[ii]))  # equation (11) from paper
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
        time_info = time.time() - start_time
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
        return weights_l1, biases_l1, weights_l2, biases_l2, parameters_received, time_info

    def __init__(self, federated, devices, ii_saved_local, neighbors, graph, compression, consensus_mode):
        self.federated = federated # true for federation active
        self.devices = devices # number of devices
        self.ii_saved_local = ii_saved_local # device index
        self.compression = compression
        self.max_neighbors = neighbors # sets the max number of neighbors from inputs
        self.graph = graph
        self.neighbor_vec = np.asarray(0, dtype=int)
        self.neighbors = self.neighbor_vec.size
        self.consensus_mode = consensus_mode

    def disable_consensus(self, federated):
        self.federated = federated

    def getFederatedWeight(self, n_W_l1, n_W_l2, n_b_l1, n_b_l2, epoch, v_loss, eps_t_control, current_neighbor, stop_consensus):
        if self.federated:
            if self.devices > 1:  # multihop topology
                if epoch == 0:
                    W_up_l1 = n_W_l1
                    n_up_l1 = n_b_l1
                    W_up_l2 = n_W_l2
                    n_up_l2 = n_b_l2
                    counter_param = W_up_l2.shape[0]*W_up_l2.shape[1]
                    sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch,
                        "loss_sample": v_loss, "counter_param": counter_param})
                    time_info = 0
                    compression_computational_time = 0
                else:
                    # temp datamat : saves temporary model parameters
                    # obtained by model averaging over a fraction of the total neighbors
                    sio.savemat('temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch, "loss_sample": v_loss})
                    # neighbor_vec = get_connectivity(self.ii_saved_local, self.neighbors, self.devices)

                    if self.graph == 0: # use k-degree network
                        self.neighbor_vec = self.get_connectivity(self.ii_saved_local, self.neighbors, self.devices)  # neighbor list
                    else:
                        if self.consensus_mode == 0:
                            if stop_consensus:
                                self.neighbor_vec = np.asarray(current_neighbor, dtype=int)
                            else:
                                self.neighbor_vec = np.zeros(1, dtype=int)
                                self.neighbor_vec[0] = current_neighbor
                        elif self.consensus_mode == 1:
                            self.neighbor_vec = np.asarray(current_neighbor, dtype=int)
                        else:
                            print("Unknown consensus mode profile, exiting")
                            exit(1)
                    self.neighbors = self.neighbor_vec.size

                    ################ reception time #######################
                    time_info = 0
                    # frame_time = 10e-3 # ieee 802.15.4
                    # payload = 1000 # bytes MPDU
                    # binary_cod = 32
                    # success_ p = 0.99
                    ############################################
                    if self.neighbors > 0: # then do model averaging
                        for neighbor_index in range(self.neighbors):
                            start_time = time.time()
                            while not os.path.isfile(
                                'datamat{}_{}.mat'.format(self.neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                                'temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch)):
                                # print('Waiting for datamat{}_{}.mat'.format(self.neighbor_vec[neighbor_index], epoch - 1))
                                # print('Waiting for temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch))
                                pause(1)
                            time_info = time_info + time.time() - start_time
                            [W_up_l1, n_up_l1, W_up_l2, n_up_l2, parameters_received, time_info_neighbor] = self.federated_weights_computing2(
                                'datamat{}_{}.mat'.format(self.neighbor_vec[neighbor_index], epoch - 1),
                                'temp_datamat{}_{}.mat'.format(self.ii_saved_local, epoch), self.ii_saved_local,
                                self.neighbor_vec[neighbor_index],
                                epoch, self.devices, self.neighbors, eps_t_control)
                            # time_info = time_info + math.ceil(parameters_received * binary_cod / payload) * frame_time
                            time_info = time_info + time_info_neighbor
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
                        time_info = 0

                    # Compression
                    start_time = time.time()
                    if self.compression == 1:
                        Model_size = W_up_l2.shape
                        threshold = 0.001
                        replacement = 0.0001
                        counter_param = 0
                        for kk in range(Model_size[0]):
                            for hh in range(Model_size[1]):
                                if np.abs(W_up_l2[kk,hh]) < threshold:
                                    W_up_l2[kk, hh] = np.sign(W_up_l2[kk, hh])*replacement
                                else:
                                    counter_param = counter_param + 1
                    # Compression differential
                    elif self.compression == 2:
                        Model_size = W_up_l2.shape
                        threshold = 1.e-4
                        replacement = 1.e-4
                        counter_param = 0
                        for kk in range(Model_size[0]):
                            for hh in range(Model_size[1]):
                                if np.abs(W_up_l2[kk, hh] - n_W_l2[kk, hh]) < threshold:
                                    W_up_l2[kk, hh] = n_W_l2[kk, hh] + np.sign(W_up_l2[kk, hh]-n_W_l2[kk, hh]) * replacement
                                else:
                                    counter_param = counter_param + 1
                    elif self.compression == 3: # high compression factor, not recommended
                        Model_size = W_up_l2.shape
                        threshold = 1.e-3
                        replacement = 1.e-3
                        counter_param = 0
                        for kk in range(Model_size[0]):
                            for hh in range(Model_size[1]):
                                if np.abs(W_up_l2[kk, hh] - n_W_l2[kk, hh]) < threshold:
                                    W_up_l2[kk, hh] = n_W_l2[kk, hh] + np.sign(W_up_l2[kk, hh]-n_W_l2[kk, hh]) * replacement
                                else:
                                    counter_param = counter_param + 1
                    elif self.compression == 4:
                        Model_size = W_up_l2.shape
                        threshold = 0.01 # high compression factor, not recommended
                        replacement = 0.001
                        counter_param = 0
                        for kk in range(Model_size[0]):
                            for hh in range(Model_size[1]):
                                if np.abs(W_up_l2[kk, hh]) < threshold:
                                    W_up_l2[kk, hh] = np.sign(W_up_l2[kk, hh]) * replacement
                                else:
                                    counter_param = counter_param + 1
                    else: # no compression
                        counter_param = W_up_l2.shape[0]*W_up_l2.shape[1]

                    if (self.compression > 0):
                        compression_computational_time = time.time() - start_time
                    else:
                        compression_computational_time = 0

                    # print(stop_consensus)
                    time_info = time_info + compression_computational_time
                    if stop_consensus:
                        try:
                            sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                                "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "counter_param": counter_param})
                            mathcontent = sio.loadmat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch))
                        except:
                            print('Unable to save file .. retrying')
                            pause(3)
                            sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                                "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "counter_param": counter_param})
                    # saving parameters
                    # try:
                    #     sio.savemat('parameters{}_{}.mat'.format(self.ii_saved_local, epoch), {
                    #         "counter_param": counter_param})
                    # except:
                    #     print('unable to save file .. retrying')
                    #     pause(3)
                    #     sio.savemat('parameters{}_{}.mat'.format(self.ii_saved_local, epoch), {
                    #         "counter_param": counter_param})

        else:
            counter_param = 0
            time_info = 0
            compression_computational_time = 0
            sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch,
                "loss_sample": v_loss, "counter_param": counter_param})
            W_up_l1 = n_W_l1
            n_up_l1 = n_b_l1
            W_up_l2 = n_W_l2
            n_up_l2 = n_b_l2
            counter_param = 0
        return W_up_l1, n_up_l1, W_up_l2, n_up_l2, counter_param,time_info, compression_computational_time
