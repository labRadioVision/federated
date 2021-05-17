from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import math
import time
from matplotlib.pyplot import pause
import os
import glob
from tensorflow.keras import layers
from tensorflow.keras import models
import warnings

class CFA_process:

    def __init__(self, devices, ii_saved_local, neighbors, federated=True, graph=0):
        self.federated = federated # true for federation active
        self.devices = devices # number of devices
        self.ii_saved_local = ii_saved_local # device index
        self.neighbors = neighbors # neighbors number (given the network topology)
        self.graph = graph
        if graph == 0:  # use k-degree network
            self.neighbor_vec = self.get_connectivity(ii_saved_local, neighbors, devices) # neighbor list
        else:
            mat_content = self.getMobileNetwork_connectivity(self.ii_saved_local, self.neighbors, self.devices, 0)
            self.neighbor_vec = np.asarray(mat_content[0], dtype=int)

    def getMobileNetwork_connectivity(self, ii_saved_local, neighbors, devices, epoch):
        graph_index = sio.loadmat('consensus/vGraph.mat')
        dev = np.arange(1, devices + 1)
        graph_mobile = graph_index['graph']
        set = graph_mobile[ii_saved_local, :, epoch]
        tot_neighbors = np.sum(set, dtype=np.uint8)
        sets_neighbors_final = np.zeros(tot_neighbors, dtype=np.uint8)
        counter = 0
        for kk in range(devices):
            if set[kk] == 1:
                sets_neighbors_final[counter] = kk
                counter = counter + 1
        return sets_neighbors_final

    def get_connectivity(self, ii_saved_local, neighbors, devices):
        if neighbors < 2:
            neighbors = 2 # set minimum to 2 neighbors
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

        if neighbors < 2:
            neighbors_final = np.delete(sets_neighbors_final, 1)
        else:
            neighbors_final = sets_neighbors_final

        return neighbors_final

    def federated_target_weights_computing(self, neighbor, neighbors, eps_t_control, epoch=0):
        warnings.filterwarnings("ignore")
        for q in range(neighbors):
            checkpointpath1 = 'results/model{}.h5'.format(neighbor[q])
            # load neighbor target model
            while not os.path.isfile(checkpointpath1):
                # implementing consensus
                print("waiting")
                pause(1)
            model = models.load_model(checkpointpath1)
            neighbor_weights = np.asarray(model.get_weights())
            for k in range(self.layers):
                self.local_weights[k] = self.local_weights[k] + eps_t_control*(neighbor_weights[k]-self.local_weights[k])
        return self.local_weights.tolist()

    def federated_weights_computing(self, neighbor, neighbors, frame_count, eps_t_control, epoch=0):
        warnings.filterwarnings("ignore")
        max_lag = 30
        stop_federation = False
        old_weights = self.local_weights
        for q in range(neighbors):
            # neighbor model and stats (train variables)
            checkpointpath1 = 'results/model{}.h5'.format(neighbor[q])
            outfile = 'results/dump_train_variables{}.npz'.format(neighbor[q])

            while not os.path.isfile(outfile):
                print("waiting for variables")
                pause(1)

            try:
                dump_vars = np.load(outfile, allow_pickle=True)
                neighbor_frame_count = dump_vars['frame_count']
                training_end = dump_vars['training_end']
            except:
                pause(5)
                print("retrying opening variables")
                try:
                    dump_vars = np.load(outfile, allow_pickle=True)
                    neighbor_frame_count = dump_vars['frame_count']
                    training_end = dump_vars['training_end']
                except:
                    print("halting federation")
                    stop_federation = True
                    break

            # load neighbor model
            pause(round(np.random.random(), 2))
            # check file and updated neighbor frame count, max lag
            while not os.path.isfile(checkpointpath1) or neighbor_frame_count < frame_count - max_lag and not training_end:
                # implementing consensus
                print("neighbor frame {} local frame {}, device {} neighbor {}".format(neighbor_frame_count, frame_count, self.ii_saved_local, neighbor[q]))
                pause(1)
                try:
                    dump_vars = np.load(outfile, allow_pickle=True)
                    neighbor_frame_count = dump_vars['frame_count']
                except:
                    pause(2)
                    print("retrying opening variables")
                    try:
                        dump_vars = np.load(outfile, allow_pickle=True)
                        neighbor_frame_count = dump_vars['frame_count']
                    except:
                        print("problems loading variables")
            try:
                model = models.load_model(checkpointpath1, compile=False)
            except:
                pause(5)
                print("retrying opening model")
                try:
                    model = models.load_model(checkpointpath1, compile=False)
                except:
                    print("halting federation")
                    stop_federation = True
                    break

            if not stop_federation:
                neighbor_weights = np.asarray(model.get_weights())
                for k in range(self.layers):
                    self.local_weights[k] = self.local_weights[k] + eps_t_control*(neighbor_weights[k]-self.local_weights[k])
                del model
            else:
                break

        if stop_federation:
           self.local_weights = old_weights

        del old_weights

        return self.local_weights.tolist()

    def federated_target_weights_aggregation(self, neighbor, neighbors, eps_t_control, epoch=0):
        for q in range(neighbors):
            checkpointpath1 = 'results/model{}.h5'.format(neighbor[q])
            # load neighbor target model
            while not os.path.isfile(checkpointpath1):
                # implementing consensus
                print("waiting")
                pause(1)
            model = models.load_model(checkpointpath1)
            neighbor_weights = np.asarray(model.get_weights())
            for k in range(self.layers):
                self.local_weights[k] = self.local_weights[k] + eps_t_control*(neighbor_weights[k]-self.local_weights[k])
        return self.local_weights.tolist()


    def update_local_target_model(self, model):
        self.local_weights = np.asarray(model)
        self.layers = self.local_weights.size

    def update_local_model(self, model):
        self.local_weights = np.asarray(model)
        self.layers = self.local_weights.size