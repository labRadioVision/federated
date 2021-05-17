from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import math
import time
from matplotlib.pyplot import pause
import os
import random
import glob
from tensorflow.keras import layers
from tensorflow.keras import models

class Parameter_Server:

    def __init__(self, devices, model_parameters, active_device_per_round, federated=True, graph=0, update_factor=0.99):
        self.federated = federated # true for federation active
        self.devices = devices # number of devices
        self.active = active_device_per_round
        self.model_parameters = model_parameters
        self.layers = self.model_parameters.size
        self.graph = graph
        self.update_factor = update_factor
        # self.checkpointpath = []
        self.file_paths = []
        self.outfile_models = []
        self.outfile = []
        self.global_model = 'results/model_global.npy'
        self.eps_t_control = 1 / devices
        self.loss = math.inf * np.ones(self.devices, dtype=float)
        for k in range(devices):
            self.outfile_models.append('results/dump_train_model{}.npy'.format(k))
            self.outfile.append('results/dump_train_variables{}.npz'.format(k))

    def federated_target_weights_aggregation(self, epoch=0, aggregation_type=0):
        # old_weights = self.model_target_parameters
        stop_aggregation = False
        # check learning status
        if (aggregation_type == 1):# opportunistic best device
            for k in random.sample(range(self.devices), self.active):
                while not os.path.isfile(self.outfile[k]):
                    # implementing consensus
                    print("waiting on server")
                    pause(1)

                try:
                    dump_vars = np.load(self.outfile[k], allow_pickle=True)
                    self.loss[k] = dump_vars['loss']
                except:
                    pause(5)
                    print("retrying opening variables on server")
                    try:
                        dump_vars = np.load(self.outfile[k], allow_pickle=True)
                        self.loss[k] = dump_vars['loss']
                    except:
                        print("failed opening variables on server")
            best_device = np.argmax(self.loss)
            print("Using model {} as target with running reward {}".format(best_device, self.loss[best_device]))
            while not os.path.isfile(self.outfile_models[int(best_device)]):
                # implementing consensus
                print("waiting")
                pause(1)

            try:
                model_weights = np.load(self.outfile_models[int(best_device)], allow_pickle=True)
            except:
                pause(5)
                print("retrying opening model")
                try:
                    model_weights = np.load(self.outfile_models[int(best_device)], allow_pickle=True)
                except:
                    print("halting aggregation")
                    stop_aggregation = True

            if not stop_aggregation:
                for q in range(self.layers):
                    self.model_parameters[q] = model_weights[q]

        elif (aggregation_type == 0): # standard model averaging (fedavg)
            combined_models = 0
            model_weights = []
            # seqc = random.sample(range(self.devices), self.active)
            for k in random.sample(range(self.devices), self.active):
                while not os.path.isfile(self.outfile_models[k]):
                    # implementing consensus
                    print("waiting")
                    pause(1)

                # model_weights.append(np.load(self.outfile_models[k], allow_pickle=True))
                try:
                    model_weights.append(np.load(self.outfile_models[k], allow_pickle=True))
                except:
                    pause(5)
                    print("retrying opening model on server")
                    try:
                        model_weights.append(np.load(self.outfile_models[k], allow_pickle=True))
                    except:
                        print("halting aggregation on server")
                        stop_aggregation = True

                if not stop_aggregation:
                    combined_models += 1
                    # for q in range(self.layers):
                    #     self.model_parameters[q] = self.model_parameters[q] + model_weights[q]
            if combined_models > 0: # normalize wrt the received models on server
                print("Received models on the PS to combine {}".format(combined_models))
                for q in range(self.layers):
                    for k in range(combined_models):
                        self.model_parameters[q] = self.model_parameters[q] + self.update_factor*(model_weights[k][q] - self.model_parameters[q])/combined_models
            # else:
            #     self.model_parameters = old_weights
            del model_weights
        return self.model_parameters


    def publish_global_model(self):
        np.save(self.global_model, self.model_parameters)