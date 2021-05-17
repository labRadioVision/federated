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

# Parameters for learning rate optimization and batch size ##################

learning_rate = 0.025
learning_rate2 = 0.2  # mu_t \times beta (from paper)
training_epochs = 120
batch_size = 5
display_step = 10

#############################################################################

# sets neighbor indexes for k-regular networks (number of neighbors is 'neighbors'
def get_connectivity(ii_saved_local, neighbors, devices):
    if (ii_saved_local == 0):
        sets_neighbors_final = np.arange(ii_saved_local + 1, ii_saved_local + neighbors + 1)
    elif (ii_saved_local == devices - 1):
        sets_neighbors_final = np.arange(ii_saved_local - neighbors, ii_saved_local)
    elif (ii_saved_local >= math.ceil(neighbors / 2)) and (ii_saved_local <= devices - math.ceil(neighbors / 2) - 1):
        sets_neighbors = np.arange(ii_saved_local - math.floor(neighbors / 2), ii_saved_local + math.floor(neighbors / 2) + 1)
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
def federated_weights_computing2(filename, filename2, ii, ii2, epoch, devices,neighbors):
    saved_epoch = epoch
    b_v = 1/devices
    eps_t_control = 1 #from paper
    while not os.path.isfile(filename2):
        print('Waiting..')
        pause(1)

    try:
        mathcontent = sio.loadmat(filename2)
    except:
        print('Detected problem while loading file')
        pause(3)
        mathcontent = sio.loadmat(filename2)

    weights_current = mathcontent['weights']
    biases_current = mathcontent['biases']

    while not os.path.isfile(filename):
        print('Waiting..')
        pause(1)

    try:
        mathcontent = sio.loadmat(filename)
    except:
        print('Detected problem while loading file')
        pause(3)
        mathcontent = sio.loadmat(filename)

    balancing_vect = np.ones(devices)*b_v
    weight_factor = (balancing_vect[ii2]/(balancing_vect[ii2] + (neighbors-1)*balancing_vect[ii]))
    updated_weights = weights_current + eps_t_control*weight_factor*(mathcontent['weights'] - weights_current) # see paper section 3
    updated_biases = biases_current + eps_t_control*weight_factor*(mathcontent['biases'] - biases_current)

    weights = updated_weights
    biases = updated_biases

    try:
        sio.savemat('temp_datamat{}_{}.mat'.format(ii, saved_epoch), {
            "weights": weights, "biases": biases})
        mathcontent = sio.loadmat('temp_datamat{}_{}.mat'.format(ii, saved_epoch))
    except:
        print('Unable to save file .. retrying')
        pause(3)
        print(biases)
        sio.savemat('temp_datamat{}_{}.mat'.format(ii, saved_epoch), {
            "weights": weights, "biases": biases})
    return weights,biases


# CFA-GE 4 stage implementation
def getFederatedWeight_gradients(n_W, n_b, federated, devices, ii_saved_local, epoch, v_loss,eng, x_train2, y_train2, neighbors):
    x_c = tf.placeholder(tf.float32, [None, 512])  # 512 point FFT range measurements
    y_c = tf.placeholder(tf.float32, [None, 8])  # 0-7 HR distances => 8 classes

    W_ext_c = tf.placeholder(tf.float32, [512, 8])
    b_ext_c = tf.placeholder(tf.float32, [8])

    # Set model weights
    # W_c = tf.Variable(tf.zeros([512, 8]))
    # b_c = tf.Variable(tf.zeros([8]))

    # Construct model
    pred_c = tf.nn.softmax(tf.matmul(x_c, W_ext_c) + b_ext_c)  # Softmax use a single layer (other options can be useD)

    # Minimize error using cross entropy
    cost_c = tf.reduce_mean(-tf.reduce_sum(y_c * tf.log(pred_c), reduction_indices=1))

    grad_W_c, grad_b_c = tf.gradients(xs=[W_ext_c, b_ext_c], ys=cost_c)

    # Initialize the variables (i.e. assign their default value)
    init_c = tf.global_variables_initializer()
    if (federated):
        if devices > 1:
            if epoch == 0:
                sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights": n_W, "biases": n_b, "epoch": epoch, "loss_sample": v_loss})
                W_up = n_W
                n_up = n_b
            else:
                sio.savemat('temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights": n_W, "biases": n_b, "epoch": epoch, "loss_sample": v_loss})
                neighbor_vec = get_connectivity(ii_saved_local, neighbors, devices)
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                            'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch - 1))
                        pause(1)
                    [W_up, n_up] = federated_weights_computing2('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1),
                                                 'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), ii_saved_local,
                                                 neighbor_vec[neighbor_index],
                                                 epoch, devices, neighbors)
                    pause(5)
                try:
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                        "weights": W_up, "biases": n_up})
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(ii_saved_local, epoch))
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                        "weights": W_up, "biases": n_up})

                while not os.path.isfile('datamat{}_{}.mat'.format(ii_saved_local, epoch)):
                    # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local, epoch))
                    pause(1)

                # waiting for other updates
                # expanded for gradient exchange
                pause(3)

                g_W_c_vect = np.zeros([512, 8, devices])
                g_b_c_vect = np.zeros([8, devices])

                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch))
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                        W_up_neigh = np.asarray(mathcontent['weights'])
                        n_up_neigh = np.squeeze(np.asarray(mathcontent['biases']))
                    except:
                        pause(5)
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                        W_up_neigh = np.asarray(mathcontent['weights'])
                        n_up_neigh = np.squeeze(np.asarray(mathcontent['biases']))
                    with tf.Session() as sess3:
                        sess3.run(init_c)
                        g_W_c, g_b_c = sess3.run([grad_W_c, grad_b_c],
                                                 feed_dict={x_c: x_train2, y_c: y_train2, W_ext_c: W_up_neigh,
                                                            b_ext_c: n_up_neigh})
                        g_W_c_vect[:, :, neighbor_vec[neighbor_index]] = g_W_c
                        g_b_c_vect[:, neighbor_vec[neighbor_index]] = g_b_c

                # save gradients and upload
                try:
                    sio.savemat('datagrad{}_{}.mat'.format(ii_saved_local, epoch), {
                        "grad_weights": g_W_c_vect, "grad_biases": g_b_c_vect, "epoch": epoch})
                    # waiting for other gradient updates
                    pause(5)
                    mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(ii_saved_local, epoch))
                    test_var = mathcontent['grad_biases']
                    del mathcontent
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datagrad{}_{}.mat'.format(ii_saved_local, epoch), {
                        "grad_weights": g_W_c_vect, "grad_biases": g_b_c_vect, "epoch": epoch})

                # waiting for other gradient updates
                pause(5)
                try:
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(ii_saved_local, epoch))
                    W_up = np.asarray(mathcontent['weights'])
                    n_up = np.squeeze(np.asarray(mathcontent['biases']))
                except:
                    pause(5)
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(ii_saved_local, epoch))
                    W_up = np.asarray(mathcontent['weights'])
                    n_up = np.squeeze(np.asarray(mathcontent['biases']))

                # update local model with neighbor gradients
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch)):
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                    except:
                        pause(3)
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                    gradW_up_neigh = np.asarray(mathcontent['grad_weights'])
                    try:
                        gradn_up_neigh = np.squeeze(np.asarray(mathcontent['grad_biases']))
                    except:
                        pause(5)
                        print('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                        del mathcontent
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch))
                        gradW_up_neigh = np.asarray(mathcontent['grad_weights'])
                        gradn_up_neigh = np.squeeze(np.asarray(mathcontent['grad_biases']))
                    W_up = W_up - learning_rate2 * np.squeeze(gradW_up_neigh[:, :, ii_saved_local])
                    n_up = n_up - learning_rate2 * np.squeeze(gradn_up_neigh[:, ii_saved_local])
        else:
            W_up = n_W
            n_up = n_b
    else:
        W_up = n_W
        n_up = n_b
    return W_up, n_up


# CFA -  GE: 2 stage (or fast) negotiation
def getFederatedWeight_gradients_fast(n_W, n_b, federated, devices, ii_saved_local, epoch, v_loss,eng, x_train2, y_train2, neighbors):
    x_c = tf.placeholder(tf.float32, [None, 512])  # 512 point FFT range measurements
    y_c = tf.placeholder(tf.float32, [None, 8])  # 0-7 HR distances => 8 classes

    W_ext_c = tf.placeholder(tf.float32, [512, 8])
    b_ext_c = tf.placeholder(tf.float32, [8])

    # Set model weights
    # W_c = tf.Variable(tf.zeros([512, 8]))
    # b_c = tf.Variable(tf.zeros([8]))

    # Construct model
    pred_c = tf.nn.softmax(tf.matmul(x_c, W_ext_c) + b_ext_c)  # Softmax

    # Minimize error using cross entropy
    cost_c = tf.reduce_mean(-tf.reduce_sum(y_c * tf.log(pred_c), reduction_indices=1))

    grad_W_c, grad_b_c = tf.gradients(xs=[W_ext_c, b_ext_c], ys=cost_c)

    # Initialize the variables (i.e. assign their default value)
    init_c = tf.global_variables_initializer()
    if (federated):
        if devices > 1:
            if epoch == 0:
                print("Error - exiting")
                exit(1)
            else:
                sio.savemat('temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights": n_W, "biases": n_b, "epoch": epoch, "loss_sample": v_loss})
                # neighbor_vec = [ii_saved_local - 1, ii_saved_local + 1]
                neighbor_vec = get_connectivity(ii_saved_local, neighbors, devices)
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                            'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch - 1))
                        pause(1)
                    [W_up,n_up] = federated_weights_computing2('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1),
                                                 'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), ii_saved_local,
                                                 neighbor_vec[neighbor_index],
                                                 epoch, devices,neighbors)
                    pause(5)
                W_up  = np.asarray(W_up)
                n_up = np.squeeze(np.asarray(n_up))

                pause(3)

                try:
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                            "weights": W_up, "biases": n_up})
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(ii_saved_local, epoch))
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                            "weights": W_up, "biases": n_up})

                g_W_c_vect = np.zeros([512, 8, devices])
                g_b_c_vect = np.zeros([8, devices])

                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch-1)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch))
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch-1))
                        W_up_neigh = np.asarray(mathcontent['weights'])
                        n_up_neigh = np.squeeze(np.asarray(mathcontent['biases']))
                    except:
                        pause(5)
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch-1))
                        W_up_neigh = np.asarray(mathcontent['weights'])
                        n_up_neigh = np.squeeze(np.asarray(mathcontent['biases']))
                    with tf.Session() as sess3:
                        sess3.run(init_c)
                        g_W_c, g_b_c = sess3.run([grad_W_c, grad_b_c],
                                                 feed_dict={x_c: x_train2, y_c: y_train2, W_ext_c: W_up_neigh,
                                                            b_ext_c: n_up_neigh})
                        g_W_c_vect[:, :, neighbor_vec[neighbor_index]] = g_W_c
                        g_b_c_vect[:, neighbor_vec[neighbor_index]] = g_b_c

                # save gradients and upload
                try:
                    sio.savemat('datagrad{}_{}.mat'.format(ii_saved_local, epoch), {
                        "grad_weights": g_W_c_vect, "grad_biases": g_b_c_vect, "epoch": epoch})
                    # waiting for other gradient updates
                    pause(5)
                    mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(ii_saved_local, epoch))
                    test_var = mathcontent['grad_biases']
                    del mathcontent
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datagrad{}_{}.mat'.format(ii_saved_local, epoch), {
                        "grad_weights": g_W_c_vect, "grad_biases": g_b_c_vect, "epoch": epoch})

                pause(5)

                # update local model with neighbor gradients (epoch - 1)
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1)):
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1))
                    except:
                        pause(3)
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1))
                    gradW_up_neigh = np.asarray(mathcontent['grad_weights'])
                    try:
                        gradn_up_neigh = np.squeeze(np.asarray(mathcontent['grad_biases']))
                    except:
                        pause(5)
                        print('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1))
                        del mathcontent
                        mathcontent = sio.loadmat('datagrad{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1))
                        gradW_up_neigh = np.asarray(mathcontent['grad_weights'])
                        gradn_up_neigh = np.squeeze(np.asarray(mathcontent['grad_biases']))
                    W_up = W_up - learning_rate2 * np.squeeze(gradW_up_neigh[:, :, ii_saved_local])
                    n_up = n_up - learning_rate2 * np.squeeze(gradn_up_neigh[:, ii_saved_local])
        else:
            W_up = n_W
            n_up = n_b
    else:
        W_up = n_W
        n_up = n_b
    return W_up, n_up


# CFA
def getFederatedWeight(n_W, n_b, federated, devices, ii_saved_local, epoch, v_loss,eng, neighbors):
    if (federated):
        if devices > 1:  # multihop topology
            if epoch == 0:
                sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights": n_W, "biases": n_b, "epoch": epoch, "loss_sample": v_loss})
                W_up = n_W
                n_up = n_b
            else:
                sio.savemat('temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                    "weights": n_W, "biases": n_b, "epoch": epoch, "loss_sample": v_loss})
                # neighbor_vec = [ii_saved_local - 1, ii_saved_local + 1]
                neighbor_vec = get_connectivity(ii_saved_local, neighbors, devices)
                print(neighbor_vec)
                for neighbor_index in range(neighbor_vec.size):
                    while not os.path.isfile(
                            'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1)) or not os.path.isfile(
                            'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch)):
                        # print('Waiting for datamat{}_{}.mat'.format(ii_saved_local - 1, epoch - 1))
                        pause(1)
                    [W_up, n_up] = federated_weights_computing2(
                        'datamat{}_{}.mat'.format(neighbor_vec[neighbor_index], epoch - 1),
                        'temp_datamat{}_{}.mat'.format(ii_saved_local, epoch), ii_saved_local,
                        neighbor_vec[neighbor_index],
                        epoch, devices, neighbors)
                    pause(5)
                try:
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                        "weights": W_up, "biases": n_up})
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(ii_saved_local, epoch))
                except:
                    print('Unable to save file .. retrying')
                    pause(3)
                    sio.savemat('datamat{}_{}.mat'.format(ii_saved_local, epoch), {
                        "weights": W_up, "biases": n_up})
                W_up = np.asarray(mathcontent['weights'])
                n_up = np.squeeze(np.asarray(mathcontent['biases']))
    else:
        W_up = n_W
        n_up = n_b
    return W_up, n_up


def processData(samples, iii, federated, tot_devices,fraction_training, neighbors_number,EPOCH_THRESHOLD):
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
    x = tf.placeholder(tf.float32, [None, 512])  # 512 POINT FFT RANGE MEASUREMENTS
    y = tf.placeholder(tf.float32, [None, 8])  # 0-7 HR distances (safe - unsafe)

    W_ext = tf.placeholder(tf.float32, [512, 8])
    b_ext = tf.placeholder(tf.float32, [8])

    W2_ext = tf.placeholder(tf.float32, [512, 8])
    b2_ext = tf.placeholder(tf.float32, [8])

    # Set model weights
    W = tf.Variable(tf.zeros([512, 8]))
    b = tf.Variable(tf.zeros([8]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W_ext) + b_ext)  # Softmax
    pred2 = tf.nn.softmax(tf.matmul(x, W2_ext) + b2_ext)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    cost2 = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred2), reduction_indices=1))

    grad_W, grad_b = tf.gradients(xs=[W_ext, b_ext], ys=cost)

    new_W = W.assign(W_ext - learning_rate * grad_W)
    new_b = b.assign(b_ext - learning_rate * grad_b)

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
                    W_val = np.zeros([512, 8])
                    b_val = np.zeros([8])
                elif (i > 0):
                    W_val = n_W # modify for minibatch updates
                    b_val = n_b

                # Fit training using batch data
                n_W, n_b, c, g_W, g_b = sess.run([new_W, new_b, cost, grad_W, grad_b], feed_dict={x: batch_xs,
                                                                y: batch_ys, W_ext: W_val, b_ext: b_val})
                avg_cost += c / total_batch  # Training loss
            # validation
            with tf.Session() as sess2:
                sess2.run(init)
                for i in range(total_batch2):
                    # Construct model
                    batch_xs = x_test[i * batch_size:((i + 1) * batch_size - 1), :]
                    batch_ys = y_test_t[i * batch_size:((i + 1) * batch_size - 1), :]
                    c = sess2.run(cost2, feed_dict={x: batch_xs,
                                                            y: batch_ys, W2_ext: n_W, b2_ext: n_b})
                    avg_cost_test += c / total_batch2
            val_loss[epoch] = avg_cost_test
            print('Test Device: ' + str(iii) + " Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost_test))

            ###########################################################
            # CFA: weights exchange (no gradients)
            # COMMENT BELOW IF CFA-GE IS SELECTED
            # W_val, b_val = getFederatedWeight(n_W, n_b, federated, tot_devices, iii, epoch, val_loss, eng, neighbors_number)
            ##################################################

            ###################################################
            # CFA - GE: 2-stage negotiation after epoch EPOCH_THRESHOLD
            # COMMENT BELOW IF CFA IS SELECTED
            if epoch < EPOCH_THRESHOLD:
                W_val, b_val = getFederatedWeight_gradients(n_W, n_b, federated, tot_devices, iii, epoch, val_loss, eng, x_train2, y_train2, neighbors_number) # method with gradients exchange
            else:
                W_val, b_val = getFederatedWeight_gradients_fast(n_W, n_b, federated, tot_devices, iii, epoch, val_loss, eng, x_train2, y_train2, neighbors_number)  # method with gradients exchange
            ###########################################################
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
    devices = 15 # NUMBER OF DE VICES
    neighbors_number = 2 # NUMBER OF NEIGHBORS PER DEVICE (K-DEGREE NETWORK)
    ii_saved = 0
    EPOCH_THRESHOLD = 4 # STARTING EPOCH FOR CFA-GE (2-STAGE NEGOTIATION)
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
        t.append(multiprocessing.Process(target=processData, args=(int(samples[ii]), ii, federated, devices, validation_train, neighbors_number, EPOCH_THRESHOLD)))
        t[ii].start()
    exit(0)
