from __future__ import division
from DataSets import MnistData
from consensus.parameter_server_v2 import Parameter_Server
# best use with PS active
# from ReplayMemory import ReplayMemory
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import glob
import scipy.io as sio
import math
from matplotlib.pyplot import pause
import time
import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
import sys
import argparse
import warnings
import json
import paho.mqtt.client as mqtt
import datetime

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-MQTT", default="10.79.5.62", help="mqtt broker ex 192.168.1.3", type=str)
parser.add_argument("-topic_PS", default="PS", help="FL with PS topic", type=str)
parser.add_argument("-topic_post_model", default="post model", help="post models", type=str)
parser.add_argument('-devices', default=9, help="sets the number of total devices", type=int)
parser.add_argument('-active_devices', default=2, help="sets the number of active devices", type=int)
args = parser.parse_args()

max_epochs = 500
devices = args.devices
active = args.active_devices
# Configuration paramaters for the whole setup
publishing = False
seed = 42
local_models_storage = [ [] for _ in range(devices) ]
detObj = {}
counter = 0
n_outputs = 6
training_end_signal = False
active_check = np.zeros(devices, dtype=bool)
scheduling_tx = np.zeros((devices, max_epochs*2), dtype=int)
indexes_tx = np.zeros((active, max_epochs*2), dtype=int)
for k in range(max_epochs*2):
    # inds = np.random.choice(devices, args.Ka, replace=False)
    sr = devices - active + 1
    sr2 = k % sr
    inds = np.arange(sr2, active + sr2)
    scheduling_tx[inds, k] = 1
    indexes_tx[:,k] = inds

def on_publish(client,userdata,result):             #create function for callback
    global publishing
    publishing = False
    print("data published \n")
    pass

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(256, 63, 1,))

    # Convolutions
    # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    #
    # layer4 = layers.Flatten()(layer3)
    #
    # layer5 = layers.Dense(512, activation="relu")(layer4)
    layer1 = layers.Conv2D(4, kernel_size=(5, 5), activation="relu")(inputs)
    layer2 = layers.AveragePooling2D(pool_size=(2, 2))(layer1)
    layer3 = layers.Conv2D(8, kernel_size=(5, 5), activation="relu")(layer2)
    layer4 = layers.AveragePooling2D(pool_size=(2, 2))(layer3)
    layer5 = layers.Flatten()(layer4)
    classification = layers.Dense(n_outputs, activation="softmax")(layer5)

    return keras.Model(inputs=inputs, outputs=classification)

def PS_callback(client, userdata, message):
    # print("ok")
    global mqttc
    global model_global, layers
    global epoch_count, frame_count
    global local_models_storage
    global counter
    global active_check
    global training_end_signal
    global publishing
    print("received")
    st = pickle.loads(message.payload)
    detObj = {}
    local_models = []
    update_factor = 1
    if active == 1:
        update_factor = 0.5

    # for k in range(layers):
    #     local_models.append(np.asarray(st['model_layer{}'.format(k)]))
    # aa = model_global.get_weights()
    # model_global.set_weights(local_models)
    # epoch_count += 1

    if st['training_end']:
        training_end_signal = True
        for k in range(layers):
            local_models.append(np.asarray(st['model_layer{}'.format(k)]))
        # aa = model_global.get_weights()
        model_global.set_weights(local_models)
    # print(scheduling_tx[st['device'], epoch_count])
    # print(active_check)
    if scheduling_tx[st['device'], epoch_count] == 1 and not training_end_signal:
        # wait for all models
        if not active_check[st['device']]:
            counter += 1
            active_check[st['device']] = True
        for k in range(layers):
            local_models.append(np.asarray(st['model_layer{}'.format(k)]))
        local_models_storage[st['device']] = local_models # replacing with the new model

    #print(counter)
    #print(active_check)
    if counter == active or training_end_signal:
        # start averaging
        active_check = np.zeros(devices, dtype=bool) # reset
        counter = 0
        if not training_end_signal:
            model_parameters = model_global.get_weights()
            # m = model_parameters
            active_device_indexes = indexes_tx[:, epoch_count]
            for q in range(layers):
                for k in range(active):
                    model_parameters[q] = model_parameters[q] + update_factor * (
                            local_models_storage[active_device_indexes[k]][q] - model_parameters[q]) / active
            model_global.set_weights(model_parameters)
        local_models_storage = [ [] for _ in range(devices) ] # reset
        epoch_count += 1
        model_list = model_global.get_weights()
        for k in range(layers):
            detObj['global_model_layer{}'.format(k)] = model_list[k].tolist()
        detObj['global_epoch'] = epoch_count
        detObj['training_end'] = training_end_signal

        print('Global epoch count {}, check training end: {}'.format(epoch_count, training_end_signal))
        if training_end_signal:
            while True:
                mqttc.publish(args.topic_PS, pickle.dumps(detObj), retain=False)
                pause(4) # send the final model on every 4 sec
        else:
            mqttc.publish(args.topic_PS, pickle.dumps(detObj), retain=False)

# -------------------------    MAIN   -----------------------------------------


if __name__ == "__main__":
    MQTT_broker = args.MQTT
    client_py = "PS"
    mqttc = mqtt.Client(client_id=client_py, clean_session=True)
    mqttc.connect(host=MQTT_broker, port=1885, keepalive=20)
    PS_mqtt_topic = args.topic_PS
    # mqttc.on_publish = on_publish
    device_topic = args.topic_post_model
    mqttc.subscribe(device_topic, qos=1)
    mqttc.message_callback_add(device_topic, PS_callback)

    checkpointpath1 = 'results/model_global.h5'

    np.random.seed(1)
    tf.random.set_seed(1)  # common initialization

    training_signal = False

    # check for backup variables on start
    if os.path.isfile(checkpointpath1):
        train_start = False

        # backup the model and the model target
        model_global = models.load_model(checkpointpath1)
        layers = model_global.get_weights().size
    else:
        train_start = True
        model_global = create_q_model()
        # for k in range(devices):
        #     local_models.append(create_q_model())
        #
        frame_count = np.zeros(devices)
        epoch_count = 0

    training_end = False

    model_weights = np.asarray(model_global.get_weights())
    layers = model_weights.size

    del model_weights

    print("start PS")
    mqttc.loop_forever()





