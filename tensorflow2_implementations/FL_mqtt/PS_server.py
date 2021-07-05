from __future__ import division
from DataSets import MnistData
from consensus.parameter_server_v2 import Parameter_Server
# best use with PS active
# from ReplayMemory import ReplayMemory
import os
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
parser.add_argument("-MQTT", default="192.168.1.3", help="mqtt broker ex 192.168.1.3", type=str)
parser.add_argument("-topic_PS", default="PS", help="FL with PS topic", type=str)
parser.add_argument("-topic_post_model", default="post model", help="post models", type=str)
parser.add_argument('-devices', default=2, help="sets the number of active devices", type=int)
args = parser.parse_args()


devices = args.devices
# Configuration paramaters for the whole setup
seed = 42

detObj = {}

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(28, 28, 1,))

    layer1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    layer2 = layers.MaxPooling2D(pool_size=(2, 2))(layer1)
    layer3 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(layer2)
    layer4 = layers.MaxPooling2D(pool_size=(2, 2))(layer3)
    layer5 = layers.Flatten()(layer4)

    # Convolutions
    # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    #
    # layer4 = layers.Flatten()(layer3)
    #
    # layer5 = layers.Dense(512, activation="relu")(layer4)
    classification = layers.Dense(n_outputs, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=classification)

def PS_callback(client, userdata, message):
    # print("ok")
    global mqttc
    global model_global, layers
    global epoch_count, frame_count

    st = json.loads(message.payload)
    detObj = {}

    #local_rounds = st['local_rounds']

    # ps
    # st['local_rounds']
    # st['epoch_global']
    # st['global_model']

    print('Frame count {}',format(frame_count))

    detObj['model'] = model.get_weights()
    detObj['device'] = device_index
    detObj['framecount'] = frame_count
    detObj['epoch'] = epoch_count
    detObj['training_end'] = training_end

    while mqttc.publish(args.topic_PS, json.dumps(detObj)):
        pause(2)
        print("error sending")

    if training_end:
        mqttc.stop_loop()
    # try:
    #     mqttc.publish(args.topic_post_model, json.dumps(detObj))
    # except:
    #     print("error sending")
    #     mqttc.disconnect()

# -------------------------    MAIN   -----------------------------------------


if __name__ == "__main__":
    MQTT_broker = args.MQTT
    client_py = "PS"
    mqttc = mqtt.Client(client_id=client_py, clean_session=True)
    mqttc.connect(host=MQTT_broker, port=1885, keepalive=60)
    PS_mqtt_topic = args.PS_topic
    device_topic = args.topic_post_model
    mqttc.subscribe(device_topic, qos=0)
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

    else:
        train_start = True
        model_global = create_q_model()
        frame_count = 0
        epoch_count = 0

    training_end = False

    model_weights = np.asarray(model_global.get_weights())
    layers = model_weights.size

    del model_weights

    # start PS and wait
    mqttc.loop_forever()




