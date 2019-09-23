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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mu', default=0.025, help="sets the learning rate for local SGD", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFA)", type=float)
parser.add_argument('-K', default=80, help="sets the number of network devices", type=int)
parser.add_argument('-N', default=2, help="sets the number of neighbors per device", type=int)
parser.add_argument('-T', default=120, help="sets the number of training epochs", type=int)
args = parser.parse_args()

# Parameters for learning rate optimization and batch size ##################

learning_rate = args.mu
training_epochs = args.T
batch_size = 5
display_step = 10

#############################################################################

################################################
# 2NN parameters
input_size = 512
intermediate_nodes = 32
##############################################

# sets neighbor indexes for k-regular networks (number of neighbors is 'neighbors'
def get_connectivity(ii_saved_local, neighbors, devices):
    if (ii_saved_local == 0):
        sets_neighbors_final = np.arange(ii_saved_local + 1, ii_saved_local + neighbors + 1)
    elif (ii_saved_local == devices - 1):
        sets_neighbors_final = np.arange(ii_saved_local - neighbors, ii_saved_local)
    elif (ii_saved_local >= math.ceil(neighbors / 2)) and (ii_saved_local <= devices - math.ceil(neighbors / 2) - 1):
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
def federated_weights_computing2(filename, filename2, ii, ii2, epoch, devices, neighbors):
    saved_epoch = epoch
    b_v = 1 / devices
    # eps_t_control = 1 #from paper
    eps_t_control = args.eps
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
    updated_biases_l1 = biases_current_l1 + eps_t_control * weight_factor * (mathcontent['biases1'] - biases_current_l1)
    updated_weights_l2 = weights_current_l2 + eps_t_control * weight_factor * (
                mathcontent['weights2'] - weights_current_l2)  # see paper section 3
    updated_biases_l2 = biases_current_l2 + eps_t_control * weight_factor * (mathcontent['biases2'] - biases_current_l2)

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


def conv1d(x, W, b, strides=1):
    # Conv1D wrapper, with bias and relu activation
    x = tf.expand_dims(x, 2)
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# CFA
def getFederatedWeight(n_W_l1, n_W_l2, n_b_l1, n_b_l2, federated, devices, ii_saved_local, epoch, v_loss, neighbors):
    if (federated):
        if devices > 1:  # multihop topology
            if epoch == 0:
                sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch,
                    "loss_sample": v_loss})
                W_up_l1 = n_W_l1
                n_up_l1 = n_b_l1
                W_up_l2 = n_W_l2
                n_up_l2 = n_b_l2
            else:
                sio.savemat('temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2, "epoch": epoch,
                    "loss_sample": v_loss})
                # neighbor_vec = [ii_saved_local - 1, ii_saved_local + 1]
                neighbor_vec = get_connectivity(ii_saved_local, neighbors, devices)
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                        'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch - 1))
                        pause(1)
                    [W_up_l1, n_up_l1, W_up_l2, n_up_l2] = federated_weights_computing2(
                        'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1),
                        'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), ii_saved_local,
                        neighbor_vec[neighbor_index],
                        epoch, devices, neighbors)
                    pause(5)
                try:
                    sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2})
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch))
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datamat{}_{}.mat'.format(self.ii_saved_local, epoch), {
                        "weights1": n_W_l1, "biases1": n_b_l1, "weights2": n_W_l2, "biases2": n_b_l2})

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

def processData(samples, iii, federated, tot_devices,fraction_training, neighbors_number):
    # eng = matlab.engine.start_matlab()
    eng = 0
    global learning_rate
    learning_rate_local = learning_rate
    np.random.seed(1)
    tf.set_random_seed(1)  # common initialization
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # MNIST DATABASE USED AS AN ALTERNATIVE
    # mnist2 = input_data.read_data_sets("/tmp/data/", one_hot=True)

    database = sio.loadmat('dati_radar_05-07-2019/data_base_all_sequences_random.mat')

    x_train = database['Data_train_2']
    y_train = database['label_train_2']
    y_train_t = to_categorical(y_train)
    x_train = (x_train.astype('float32') + 140) / 140 # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)
    x_train2 = x_train[iii * samples:((iii + 1) * samples - 1), :] # DATA PARTITION
    y_train2 = y_train_t[iii * samples:((iii + 1) * samples - 1),:]

    x_test = database['Data_test_2']
    y_test = database['label_test_2']
    x_test = (x_test.astype('float32') + 140) / 140
    y_test_t = to_categorical(y_test)

    total_batch2 = int(fraction_training / batch_size)
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, input_size])  # 512 POINT FFT RANGE MEASUREMENTS
    y = tf.placeholder(tf.float32, [None, 8])  # 0-7 HR distances (safe - unsafe)

    W_ext_l1 = tf.placeholder(tf.float32, [input_size, intermediate_nodes])
    b_ext_l1 = tf.placeholder(tf.float32, [intermediate_nodes])
    W_ext_l2 = tf.placeholder(tf.float32, [intermediate_nodes, 8])
    b_ext_l2 = tf.placeholder(tf.float32, [8])

    W2_ext_l1 = tf.placeholder(tf.float32, [input_size, intermediate_nodes])
    b2_ext_l1 = tf.placeholder(tf.float32, [intermediate_nodes])
    W2_ext_l2 = tf.placeholder(tf.float32, [intermediate_nodes, 8])
    b2_ext_l2 = tf.placeholder(tf.float32, [8])

    # Set model weights
    W_l1 = tf.Variable(tf.random_normal([input_size,intermediate_nodes]))
    b_l1 = tf.Variable(tf.random_normal([intermediate_nodes]))
    W_l2 = tf.Variable(tf.zeros([intermediate_nodes, 8]))
    b_l2 = tf.Variable(tf.zeros([8]))

    # Construct model
    hidden0 = tf.nn.relu(tf.matmul(x, W_ext_l1) + b_ext_l1)  # layer 1 example
    pred = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(x, W_ext_l1) + b_ext_l1), W_ext_l2) + b_ext_l2)  # example 2 layers
    hidden20 = tf.nn.relu(tf.matmul(x, W2_ext_l1) + b2_ext_l1)  # layer 1 example
    pred2 = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(x, W2_ext_l1) + b2_ext_l1), W2_ext_l2) + b2_ext_l2)  # example 2 layers

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-15, 0.99)), reduction_indices=1))
    cost2 = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred2, 1e-15, 0.99)), reduction_indices=1))

    #gradients per layer
    grad_W_l1, grad_b_l1, grad_W_l2, grad_b_l2 = tf.gradients(xs=[W_ext_l1, b_ext_l1, W_ext_l2, b_ext_l2], ys=cost)

    new_W_l1 = W_l1.assign(W_ext_l1 - learning_rate * grad_W_l1)
    new_b_l1 = b_l1.assign(b_ext_l1 - learning_rate * grad_b_l1)

    new_W_l2 = W_l2.assign(W_ext_l2 - learning_rate * grad_W_l2)
    new_b_l2 = b_l2.assign(b_ext_l2 - learning_rate * grad_b_l2)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

#    Start training
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(samples / batch_size)
        # PRINTS THE TOTAL NUMBER OF MINI BATCHES
        print(total_batch)

        # Training cycle
        val_loss = np.zeros(training_epochs)
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost_test = 0.

            for i in range(total_batch):
                batch_xs = x_train2[i * batch_size:((i + 1) * batch_size - 1), :]
                batch_ys = y_train2[i * batch_size:((i + 1) * batch_size - 1), :]
                if (i == 0) and (epoch == 0): # initialization
                    # W_val_l1 = np.zeros([512, 32])
                    W_val_l1 = np.random.normal(0.0, 1.0, (input_size, intermediate_nodes))
                    # b_val_l1 = np.zeros([32])
                    b_val_l1 = np.random.normal(0.0, 1.0, intermediate_nodes)
                    W_val_l2 = np.zeros([intermediate_nodes, 8])
                    b_val_l2 = np.zeros([8])
                elif (i > 0):
                    W_val_l1 = n_W_l1 # modify for minibatch updates
                    b_val_l1 = n_b_l1
                    W_val_l2 = n_W_l2  # modify for minibatch updates
                    b_val_l2 = n_b_l2

                # Fit training using batch data
                n_W_l1, n_b_l1, n_W_l2, n_b_l2, c, g_W_l1, g_b_l1, g_W_l2, g_b_l2 = sess.run([new_W_l1, new_b_l1,
                                        new_W_l2, new_b_l2, cost, grad_W_l1, grad_b_l1, grad_W_l2, grad_b_l2], feed_dict={x: batch_xs,
                                        y: batch_ys, W_ext_l1: W_val_l1, b_ext_l1: b_val_l1, W_ext_l2: W_val_l2, b_ext_l2: b_val_l2})
                avg_cost += c / total_batch  # Training loss
            # validation
            with tf.Session() as sess2:
                sess2.run(init)
                for i in range(total_batch2):
                    # Construct model
                    batch_xs = x_test[i * batch_size:((i + 1) * batch_size - 1), :]
                    batch_ys = y_test_t[i * batch_size:((i + 1) * batch_size - 1), :]
                    c = sess2.run(cost2, feed_dict={x: batch_xs,
                                        y: batch_ys, W2_ext_l1: n_W_l1, b2_ext_l1: n_b_l1, W2_ext_l2: n_W_l2, b2_ext_l2: n_b_l2})
                    avg_cost_test += c / total_batch2
            val_loss[epoch] = avg_cost_test
            print('Test Device: ' + str(iii) + " Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost_test))

            ###########################################################
            # CFA: weights exchange (no gradients)
            W_val_l1, b_val_l1, W_val_l2, b_val_l2 = getFederatedWeight(n_W_l1, n_W_l2, n_b_l1, n_b_l2, federated, tot_devices, iii, epoch, val_loss, neighbors_number)
            ##################################################

        print("Optimization Finished!")
        # DUMP RESULTS
        sio.savemat(
            'results/dump_loss_{}_{date:%Y-%m-%d-%H-%M-%S}.mat'.format(iii, date=datetime.datetime.now().time()), {
                "val_acc": val_loss, "device": iii})
        # Test model
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == "__main__":

    # DELETE TEMPORARY CACHE FILES
    fileList = glob.glob('*.mat', recursive=False)
    print(fileList)
    for filePath in fileList:
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file")

    ##################### SETS SIMULATION PARAMETERS ###############################
    devices = args.K  # NUMBER OF DE VICES
    neighbors_number = args.N  # NUMBER OF NEIGHBORS PER DEVICE (K-DEGREE NETWORK)
    ii_saved = 0
    federated = True # ENABLE FEDERATED LEARNING)

    training_set_per_device = 25 # NUMBER OF TRAINING SAMPLES PER DEVICE
    fraction_training = int(devices*training_set_per_device) # total training
    b_v = 1/devices
    balancing_vect = np.ones(devices)*b_v
    samples = np.zeros(devices) # training samples per device
    validation_train = 16000 # VALIDATION DATASET
    ###################################################################################

    # START MULTIPROCESSING
    for id in range(devices):
        samples[id] = math.floor(balancing_vect[id]*fraction_training)
    # samples = int(fraction_training/devices) # training samples per device
    print(samples)
    t = []
    iii = 0
    for ii in range(devices):
        t.append(multiprocessing.Process(target=processData, args=(int(samples[ii]), ii, federated, devices, validation_train, neighbors_number)))
        t[ii].start()
    exit(0)
