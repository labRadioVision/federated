from __future__ import division
from DataSets import MnistData
from consensus.consensus_v3 import CFA_process
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
parser.add_argument('-resume', default=0, help="set 1 to resume from a previous simulation, 0 to start from the beginning", type=float)
parser.add_argument("-MQTT", default="192.168.1.3", help="mqtt broker ex 192.168.1.3", type=str)
parser.add_argument("-topic_PS", default="PS", help="FL with PS topic", type=str)
parser.add_argument("-topic_post_model", default="post model", help="post models", type=str)
parser.add_argument("-topic_consensus", default="consensus", help="Consensus driven FL", type=str)
parser.add_argument("-ID", default=0, help="device/learner identifier", type=int)
parser.add_argument('-mu', default=0.00025, help="sets the learning rate for all setups", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFA)", type=float)
parser.add_argument("-local_rounds", default=4, help="number of local rounds", type=int)
parser.add_argument('-target', default=0.1, help="sets the target loss to stop federation", type=float)
parser.add_argument('-N', default=1, help="sets the max. number of neighbors per device per round in CFA", type=int)
parser.add_argument('-samp', default=500, help="sets the number samples per device", type=int)
parser.add_argument('-batches', default=5, help="sets the number of batches per learning round", type=int)
parser.add_argument('-batch_size', default=100, help="sets the batch size per learning round", type=int)
args = parser.parse_args()


target_loss = args.target
# Configuration paramaters for the whole setup
seed = 42
local_rounds = args.local_rounds
# batch_size = 5  # Size of batch taken from replay buffer
batch_size = args.batch_size
number_of_batches = args.batches
training_set_per_device = args.samp # NUMBER OF TRAINING SAMPLES PER DEVICE
validation_train = 60000  # VALIDATION and training DATASET size
validation_test = 10000
number_of_batches_for_validation = int(validation_test/batch_size)

print("Number of batches for learning {}".format(number_of_batches))

max_lag = number_of_batches*2 # consensus max delay 2= 2 epochs max

n_outputs = 10  # 6 classes
max_epochs = 200

validation_start = 1 # start validation in epochs

# Using huber loss for stability
loss_function = keras.losses.Huber()
detObj = {}

def preprocess_observation(obs, batch_size):
    img = obs# crop and downsize
    img = (img).astype(np.float)
    return img.reshape(batch_size, 28, 28, 1)

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
    global model
    global epoch_count, frame_count, training_set_per_device, max_epochs
    global number_of_batches, training_signal, number_of_batches_for_validation
    global layers, data_handle, target_loss, training_end
    global data_history, label_history, validation_start, device_index, epoch_loss_history
    global outfile, outfile_models, checkpointpath1
    model_weights = np.asarray(model.get_weights())
    st = json.loads(message.payload)
    detObj = {}
    local_round = 0
    local_rounds = st['local_rounds']

    # ps
    # st['local_rounds']
    # st['epoch_global']
    # st['global_model']

    print('Frame count {}',format(frame_count))
    if int(st['epoch_global']) >= epoch_count:
        model_global = np.asarray(st['global_model'])
        for k in range(layers):
            # model_weights[k] = model_weights[k]+ 0.5*(model_global[k]-model_weights[k])
            model_weights[k] = model_global[k]
        model.set_weights(model_weights.tolist())

    # validation tool for device 'device_index'
    if epoch_count > validation_start:
        avg_cost = 0.
        for i in range(number_of_batches_for_validation):
            obs_valid, labels_valid = data_handle.getTestData(batch_size, i)
            # obs_valid, labels_valid = data_handle.getRandomTestData(batch_size)
            data_valid = preprocess_observation(np.squeeze(obs_valid), batch_size)
            data_sample = np.array(data_valid)
            label_sample = np.array(labels_valid)
            # Create a mask to calculate loss
            masks = tf.one_hot(label_sample, n_outputs)
            classes = model(data_sample)
            # Apply the masks
            class_v = tf.reduce_sum(tf.multiply(classes, masks), axis=1)
            # Calculate loss
            loss = loss_function(label_sample, class_v)
            avg_cost += loss / number_of_batches_for_validation  # Training loss
        epoch_loss_history.append(avg_cost)
        print("Device {} epoch count {}, validation loss {:.2f}".format(device_index, epoch_count,
                                                                            avg_cost))
        # mean loss for last 5 epochs
        running_loss = np.mean(epoch_loss_history[-5:])

        if running_loss < target_loss or training_signal:  # Condition to consider the task solved
            print("Solved for device {} at epoch {} with average loss {:.2f} !".format(device_index, epoch_count,
                                                                                       running_loss))
            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                     training_end=training_end, epoch_count=epoch_count, loss=running_loss)
            np.save(outfile_models, model_weights)

            dict_1 = {"epoch_loss_history": epoch_loss_history, "batches": number_of_batches,
                      "batch_size": batch_size, "samples": training_set_per_device}

            sio.savemat(
                "results/matlab/Device_{}_samples_{}_batches_{}_size{}.mat".format(
                    device_index, training_set_per_device, number_of_batches, batch_size), dict_1)
            sio.savemat(
                "CFA_device_{}_samples_{}_batches_{}_size{}.mat".format(
                    device_index, training_set_per_device, number_of_batches, batch_size), dict_1)

        if epoch_count > max_epochs:  # stop simulation
            print("Unsolved for device {} at epoch {}!".format(device_index, epoch_count))
            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            # model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                     training_end=training_end, epoch_count=epoch_count, loss=running_loss)
            np.save(outfile_models, model_weights)

            dict_1 = {"epoch_loss_history": epoch_loss_history, "batches": number_of_batches,
                      "batch_size": batch_size, "samples": training_set_per_device}

            sio.savemat(
                "results/matlab/Device_{}_samples_{}_batches_{}_size{}.mat".format(
                    device_index, training_set_per_device, number_of_batches, batch_size), dict_1)
            sio.savemat(
                "CFA_device_{}_samples_{}_batches_{}_size{}.mat".format(
                    device_index, training_set_per_device, number_of_batches, batch_size), dict_1)

    while local_round < local_rounds and not training_end:
        frame_count += 1
        obs, labels = data_handle.getTrainingData(batch_size)
        data_batch = preprocess_observation(obs, batch_size)
        # Save data and labels in the current learning session
        data_history.append(data_batch)
        label_history.append(labels)
        # Local learning update every "number of batches" batches
        data_history = []
        label_history = []
        # Local learning update every "number of batches" batches
        if frame_count % number_of_batches == 0 and not training_signal:
            epoch_count += 1
            local_round += 1
            for i in range(number_of_batches):
                data_sample = np.array(data_history[i])
                label_sample = np.array(label_history[i])

                # Create a mask to calculate loss
                masks = tf.one_hot(label_sample, n_outputs)

                with tf.GradientTape() as tape:
                    # Train the model on data samples
                    classes = model(data_sample)
                    # Apply the masks
                    class_v = tf.reduce_sum(tf.multiply(classes, masks), axis=1)
                    # Calculate loss
                    loss = loss_function(label_sample, class_v)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            del data_history
            del label_history
            data_history = []
            label_history = []

    detObj['model'] = model.get_weights()
    detObj['device'] = device_index
    detObj['framecount'] = frame_count
    detObj['epoch'] = epoch_count
    detObj['training_end'] = training_end

    while mqttc.publish(args.topic_post_model, json.dumps(detObj)):
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
    if args.resume == 0: # clear all files
        # DELETE TEMPORARY CACHE FILES
        fileList = glob.glob('results/*.npy', recursive=False)
        print(fileList)
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")

        fileList = glob.glob('results/*.h5', recursive=False)
        print(fileList)
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")

        fileList = glob.glob('results/*.npz', recursive=False)
        print(fileList)
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")

        fileList = glob.glob('*.mat', recursive=False)
        print(fileList)
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")

    MQTT_broker = args.MQTT
    device_index = args.RID
    client_py = "learner " + str(device_index)
    mqttc = mqtt.Client(client_id=client_py, clean_session=True)
    mqttc.connect(host=MQTT_broker, port=1885, keepalive=60)
    PS_mqtt_topic = args.PS_topic
    mqttc.subscribe(PS_mqtt_topic, qos=0)
    mqttc.message_callback_add(PS_mqtt_topic, PS_callback)
    start_index = device_index*training_set_per_device

    checkpointpath1 = 'results/model{}.h5'.format(device_index)
    outfile = 'results/dump_train_variables{}.npz'.format(device_index)
    outfile_models = 'results/dump_train_model{}.npy'.format(device_index)

    np.random.seed(1)
    tf.random.set_seed(1)  # common initialization

    learning_rate = args.mu
    learning_rate_local = learning_rate

    training_signal = False

    # check for backup variables on start
    if os.path.isfile(checkpointpath1):
        train_start = False

        # backup the model and the model target
        model = models.load_model(checkpointpath1)
        data_history = []
        label_history = []
        local_model_parameters = np.load(outfile_models, allow_pickle=True)
        model.set_weights(local_model_parameters.tolist())

        dump_vars = np.load(outfile, allow_pickle=True)
        frame_count = dump_vars['frame_count']
        epoch_loss_history = dump_vars['epoch_loss_history'].tolist()
        running_loss = np.mean(epoch_loss_history[-5:])
        epoch_count = dump_vars['epoch_count']
    else:
        train_start = True
        model = create_q_model()
        data_history = []
        label_history = []
        frame_count = 0
        # Experience replay buffers
        epoch_loss_history = []
        epoch_count = 0
        running_loss = math.inf

    training_end = False
    # set an arbitrary optimizer, here Adam is used
    optimizer = keras.optimizers.Adam(learning_rate=args.mu, clipnorm=1.0)
    # create a data object (here radar data)
    data_handle = MnistData(device_index, start_index, training_set_per_device, validation_train, 0)

    # create a consensus object (only for consensus)
    # cfa_consensus = CFA_process(device_index, args.N)

    model_weights = np.asarray(model.get_weights())
    layers = model_weights.size

    del model_weights

    while epoch_count < local_rounds:
        frame_count += 1
        obs, labels = data_handle.getTrainingData(batch_size)
        data_batch = preprocess_observation(obs, batch_size)
        # Save data and labels in the current learning session
        data_history.append(data_batch)
        label_history.append(labels)
        # Local learning update every "number of batches" batches

        # Local learning update every "number of batches" batches
        if frame_count % number_of_batches == 0 and not training_signal:
            epoch_count += 1
            for i in range(number_of_batches):
                data_sample = np.array(data_history[i])
                label_sample = np.array(label_history[i])

                # Create a mask to calculate loss
                masks = tf.one_hot(label_sample, n_outputs)

                with tf.GradientTape() as tape:
                    # Train the model on data samples
                    classes = model(data_sample)
                    # Apply the masks
                    class_v = tf.reduce_sum(tf.multiply(classes, masks), axis=1)
                    # Calculate loss
                    loss = loss_function(label_sample, class_v)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            del data_history
            del label_history
            data_history = []
            label_history = []

    detObj['model'] = model.get_weights()
    detObj['device'] = device_index
    detObj['framecount'] = frame_count
    detObj['epoch'] = epoch_count
    detObj['training_end'] = training_end

    while mqttc.publish(args.topic_post_model, json.dumps(detObj)):
        pause(2)
        print("error sending")
    # subscribe to PS and wait
    mqttc.loop_forever()





