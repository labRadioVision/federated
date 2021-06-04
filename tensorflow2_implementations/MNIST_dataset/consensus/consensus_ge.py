from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import math
import time
from matplotlib.pyplot import pause
import os
import tensorflow as tf
from tensorflow import keras
import glob
from tensorflow.keras import layers
from tensorflow.keras import models
import warnings

class CFA_GE_process:

    def __init__(self, devices, ii_saved_local, neighbors, federated=True, mu2=0.01, graph=0):
        self.federated = federated # true for federation active
        self.devices = devices # number of devices
        self.ii_saved_local = ii_saved_local # device index
        self.neighbors = neighbors # neighbors number (given the network topology)
        self.graph = graph
        self.training_end = False
        self.optimizer2 = keras.optimizers.Adam(learning_rate=mu2)



    def federated_weights_computing(self, neighbor, neighbors, epoch_count, eps_t_control, epoch=0, max_lag=30):
        warnings.filterwarnings("ignore")
        # max_lag = 30 # default 30
        stop_federation = False
        old_weights = self.local_weights

        neighbor_weights = []
        # seqc = random.sample(range(self.devices), self.active)

        for q in range(neighbors):
            # neighbor model and stats (train variables)
            outfile_models = 'results/dump_train_model{}.npy'.format(neighbor[q])
            outfile = 'results/dump_train_variables{}.npz'.format(neighbor[q])

            while not os.path.isfile(outfile):
                print("waiting for variables")
                pause(1)

            try:
                dump_vars = np.load(outfile, allow_pickle=True)
                neighbor_epoch_count = dump_vars['epoch_count']
                self.training_end = dump_vars['training_end']
            except:
                pause(5)
                print("retrying opening variables")
                try:
                    dump_vars = np.load(outfile, allow_pickle=True)
                    neighbor_epoch_count = dump_vars['epoch_count']
                    self.training_end = dump_vars['training_end']
                except:
                    print("halting federation")
                    stop_federation = True
                    break

            pause(round(np.random.random(), 2))
            # check file and updated neighbor frame count, max lag
            if not stop_federation:
                while not os.path.isfile(outfile_models) or neighbor_epoch_count < epoch_count - max_lag and not self.training_end:
                    # implementing consensus
                    # print("neighbor frame {} local frame {}, device {} neighbor {}".format(neighbor_frame_count, frame_count, self.ii_saved_local, neighbor[q]))
                    pause(1)
                    try:
                        dump_vars = np.load(outfile, allow_pickle=True)
                        neighbor_epoch_count = dump_vars['epoch_count']
                        self.training_end = dump_vars['training_end']
                    except:
                        pause(2)
                        print("retrying opening variables")
                        try:
                            dump_vars = np.load(outfile, allow_pickle=True)
                            neighbor_epoch_count = dump_vars['epoch_count']
                            self.training_end = dump_vars['training_end']
                        except:
                            print("problems loading variables")

                # load neighbor model
                try:
                    neighbor_weights.append(np.load(outfile_models, allow_pickle=True))
                except:
                    pause(5)
                    print("retrying opening model")
                    try:
                        neighbor_weights.append(np.load(outfile_models, allow_pickle=True))
                    except:
                        print("failed to load model federation")

                if self.training_end and len(neighbor_weights) > 0:
                    # one of the neighbors solved the optimization, apply transfer learning
                    break


        if len(neighbor_weights) > 0:
            eps_t_control = 1 / (len(neighbor_weights) + 1) # overwrite
            for q in range(len(neighbor_weights)):
                if self.training_end:
                    print("detected training end")
                    # it is reasonable to replace local model with the received one as succesful, stop model averaging with other neighbors
                    for k in range(self.layers):
                        self.local_weights[k] = neighbor_weights[-1][k]
                    break
                else: # apply model averaging
                    for k in range(self.layers):
                        self.local_weights[k] = self.local_weights[k] + eps_t_control*(neighbor_weights[q][k]-self.local_weights[k])
                        # self.local_weights[k] = self.local_weights[k] + eps_t_control * (neighbor_weights[k] - self.local_weights[k])
            del neighbor_weights

        return self.local_weights.tolist()

    def federated_grads_computing(self, neighbor, neighbors, epoch_count, eps_t_control, epoch=0, max_lag=30):
        warnings.filterwarnings("ignore")
        # max_lag = 30 # default 30
        stop_federation = False
        old_grads = self.local_gradients

        neighbor_grads = []
        # seqc = random.sample(range(self.devices), self.active)

        for q in range(neighbors):
            # neighbor model and stats (train variables)
            outfile = 'results/dump_train_variables{}.npz'.format(neighbor[q])
            outfile_models_grad = 'results/dump_train_grad{}.npy'.format(neighbor[q])

            while not os.path.isfile(outfile):
                print("waiting for variables")
                pause(1)

            try:
                dump_vars = np.load(outfile, allow_pickle=True)
                neighbor_epoch_count = dump_vars['epoch_count']
                self.training_end = dump_vars['training_end']
            except:
                pause(5)
                print("retrying opening variables")
                try:
                    dump_vars = np.load(outfile, allow_pickle=True)
                    neighbor_epoch_count = dump_vars['epoch_count']
                    self.training_end = dump_vars['training_end']
                except:
                    print("halting federation")
                    stop_federation = True
                    break

            pause(round(np.random.random(), 2))
            # check file and updated neighbor frame count, max lag
            if not stop_federation:
                while not os.path.isfile(
                        outfile_models_grad) or neighbor_epoch_count < epoch_count - max_lag and not self.training_end:
                    # implementing consensus
                    # print("neighbor frame {} local frame {}, device {} neighbor {}".format(neighbor_frame_count, frame_count, self.ii_saved_local, neighbor[q]))
                    pause(1)
                    try:
                        dump_vars = np.load(outfile, allow_pickle=True)
                        neighbor_epoch_count = dump_vars['epoch_count']
                        self.training_end = dump_vars['training_end']
                    except:
                        pause(2)
                        print("retrying opening variables")
                        try:
                            dump_vars = np.load(outfile, allow_pickle=True)
                            neighbor_epoch_count = dump_vars['epoch_count']
                            self.training_end = dump_vars['training_end']
                        except:
                            print("problems loading variables")

                # load neighbor model
                try:
                    neighbor_grads.append(np.load(outfile_models_grad, allow_pickle=True))
                except:
                    pause(5)
                    print("retrying opening model")
                    try:
                        neighbor_grads.append(np.load(outfile_models_grad, allow_pickle=True))
                    except:
                        print("failed to load model federation")

                if self.training_end and len(neighbor_grads) > 0:
                    # one of the neighbors solved the optimization, apply transfer learning
                    break

        if len(neighbor_grads) > 0:
            eps_t_control = 1 / (len(neighbor_grads) + 1)  # overwrite
            for q in range(len(neighbor_grads)):
                # apply model averaging
                for k in range(self.layers):
                    self.local_gradients[k] = self.local_gradients[k] + eps_t_control * (
                                    neighbor_grads[q][k] - self.local_gradients[k])
            del neighbor_grads

        grads_out = []
        for ii in range(self.layers):
            grads_out.append(tf.convert_to_tensor(self.local_gradients[ii]))

        return grads_out

    def getTrainingStatusFromNeightbor(self):
        return self.training_end

    def update_local_target_model(self, model):
        self.local_weights = model
        self.layers = self.local_weights.size

    def update_local_gradient(self, gradients):
        self.local_gradients = gradients


    def update_local_model(self, model):
        self.local_weights = model
        self.layers = self.local_weights.size