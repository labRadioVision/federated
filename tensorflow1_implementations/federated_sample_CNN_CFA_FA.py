from __future__ import absolute_import, division, print_function, unicode_literals
from keras.utils import to_categorical
from consensus.cfa import CFA_process
import numpy as np
import tensorflow as tf
import datetime
import scipy.io as sio
import multiprocessing
from consensus.cfa import CFA_process
import math
from matplotlib.pyplot import pause
import os
import glob
import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-mu', default=0.025, help="sets the learning rate for local SGD", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFA)", type=float)
parser.add_argument('-eps2', default=0.5, help="sets the updated parameters for server-side federated learning", type=float)
parser.add_argument('-K', default=5, help="sets the number of network devices", type=int)
parser.add_argument('-N', default=2, help="sets the number of neighbors per device", type=int)
parser.add_argument('-T', default=60, help="sets the number of training epochs", type=int)
parser.add_argument('-S',default=1, help="sets the frequency of server-side computaion", type=int)
parser.add_argument('-Ser',default=10, help="sets the number of epochs for server-side computation", type=int)
parser.add_argument('-Con',default=10, help="sets the number of epochs for consensus operations", type=int)
args = parser.parse_args()

# Parameters for learning rate optimization and batch size ##################

learning_rate = args.mu
training_epochs = args.T
server_epochs = args.S
consensus_epochs = args.Con
server_duration = args.Ser
batch_size = 5
display_step = 10


# convolutional 1D parameters
filter = 16
number = 8
pooling = 5
stride = 5
multip = 21

#############################################################################

def conv1d(x, W, b, strides=1):
    # Conv1D wrapper, with bias and relu activation
    x = tf.expand_dims(x, 2)
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def processServer(iii, federated, tot_devices,fraction_training):
    # eng = matlab.engine.start_matlab()
    eng = 0
    global learning_rate
    learning_rate_local = learning_rate
    np.random.seed(1)
    tf.set_random_seed(1)  # common initialization

    # Initialize CFA
    # consensus_p = CFA_process(federated, tot_devices, iii, neighbors_number)
    b_v = 1 / tot_devices
    balancing_vect = np.ones(tot_devices) * b_v

    for epoch in range(0, training_epochs, consensus_epochs + server_duration):
        if (epoch == 0):
            # initialization
            server_w1 = np.zeros([filter, 1, number])
            server_b1 = np.zeros([number])
            server_w2 = np.zeros([multip * number, 8])
            server_b2 = np.zeros([8])
            for devices in range(tot_devices):
                while not os.path.isfile('datamat{}_{}.mat'.format(devices, epoch)):
                    pause(1)
                try:
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch))
                except:
                    print('Detected problem while loading file')
                    pause(3)
                    mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch))
                server_w1 = server_w1 + balancing_vect[devices] * mathcontent['weights1']
                server_b1 = server_b1 + balancing_vect[devices] * mathcontent['biases1']
                server_w2 = server_w2 + balancing_vect[devices] * mathcontent['weights2']
                server_b2 = server_b2 + balancing_vect[devices] * mathcontent['biases2']

            for epoch2 in range(epoch+1, server_duration + epoch, server_epochs):
                eps_t_control = args.eps
                for devices in range(tot_devices):
                    while not os.path.isfile('datamat{}_{}.mat'.format(devices, epoch2)):
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch2))
                    except:
                        print('Detected problem while loading file')
                        pause(3)
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch2))

                    server_w1 = server_w1 + eps_t_control * balancing_vect[devices] * (
                            mathcontent['weights1'] - server_w1)  # see paper section 3
                    server_b1 = server_b1 + eps_t_control * balancing_vect[devices] * (
                            mathcontent['biases1'] - server_b1)  # see paper section 3
                    server_w2 = server_w2 + eps_t_control * balancing_vect[devices] * (
                            mathcontent['weights2'] - server_w2)  # see paper section 3
                    server_b2 = server_b2 + eps_t_control * balancing_vect[devices] * (
                            mathcontent['biases2'] - server_b2)  # see paper section 3
                ###########################################################
                sio.savemat('server_datamat{}_{}.mat'.format(iii, epoch2), {
                    "weights1": server_w1, "biases1": server_b1, "weights2": server_w2, "biases2": server_b2,
                    "epoch": epoch2})
            epoch = epoch2

        else:
            for epoch2 in range(epoch, server_duration + epoch, server_epochs):
                eps_t_control = args.eps
                for devices in range(tot_devices):
                    while not os.path.isfile('datamat{}_{}.mat'.format(devices, epoch2)):
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch2))
                    except:
                        print('Detected problem while loading file')
                        pause(3)
                        mathcontent = sio.loadmat('datamat{}_{}.mat'.format(devices, epoch2))

                    server_w1 = server_w1 + eps_t_control * balancing_vect[devices] * (mathcontent['weights1'] - server_w1)  # see paper section 3
                    server_b1 = server_b1 + eps_t_control * balancing_vect[devices] * (mathcontent['biases1'] - server_b1)  # see paper section 3
                    server_w2 = server_w2 + eps_t_control * balancing_vect[devices] * (mathcontent['weights2'] - server_w2)  # see paper section 3
                    server_b2 = server_b2 + eps_t_control * balancing_vect[devices] * (mathcontent['biases2'] - server_b2)  # see paper section 3
                ###########################################################
                sio.savemat('server_datamat{}_{}.mat'.format(iii, epoch2), {
                    "weights1": server_w1, "biases1": server_b1, "weights2": server_w2, "biases2": server_b2, "epoch": epoch2})
            epoch = epoch2

def processData(samples, iii, federated, tot_devices,fraction_training, neighbors_number):
    # eng = matlab.engine.start_matlab()
    eng = 0
    global learning_rate
    learning_rate_local = learning_rate
    np.random.seed(1)
    tf.set_random_seed(1)  # common initialization
    server_federation = False
    server_counter = 0
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

    W_ext_l1 = tf.placeholder(tf.float32, [filter, 1, number])
    b_ext_l1 = tf.placeholder(tf.float32, [number])
    W_ext_l2 = tf.placeholder(tf.float32, [multip * number, 8])
    b_ext_l2 = tf.placeholder(tf.float32, [8])

    W2_ext_l1 = tf.placeholder(tf.float32, [filter, 1, number])
    b2_ext_l1 = tf.placeholder(tf.float32, [number])
    W2_ext_l2 = tf.placeholder(tf.float32, [multip * number, 8])
    b2_ext_l2 = tf.placeholder(tf.float32, [8])

    # Set model weights
    W_l1 = tf.Variable(tf.random_normal([filter, 1, number]))
    b_l1 = tf.Variable(tf.random_normal([number]))
    W_l2 = tf.Variable(tf.zeros([multip * number, 8]))
    b_l2 = tf.Variable(tf.zeros([8]))

    # Construct model Layer #1 CNN 1d, Layer #2 FC
    hidden0 = conv1d(x, W_ext_l1, b_ext_l1)
    hidden01 = tf.layers.max_pooling1d(hidden0, pool_size=stride, strides=stride, padding='SAME')
    fc01 = tf.reshape(hidden01, [-1, multip*number])
    pred = tf.nn.softmax(tf.matmul(fc01, W_ext_l2) + b_ext_l2)  # example 2 layers

    hidden2 = conv1d(x, W2_ext_l1, b2_ext_l1)
    hidden02 = tf.layers.max_pooling1d(hidden2, pool_size=stride, strides=stride, padding='SAME')
    fc02 = tf.reshape(hidden02, [-1, multip*number])
    pred2 = tf.nn.softmax(tf.matmul(fc02, W2_ext_l2) + b2_ext_l2)  # example 2 layers

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

    # Initialize CFA
    consensus_p = CFA_process(federated, tot_devices, iii, neighbors_number)

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
                    W_val_l1 = np.random.normal(0.0, 1.0, (filter, 1, number))
                    # b_val_l1 = np.zeros([32])
                    b_val_l1 = np.random.normal(0.0, 1.0, number)
                    W_val_l2 = np.zeros([multip*number, 8])
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
            eps_t_control2 = args.eps2

            if (epoch % (consensus_epochs + server_duration) == 0)|(server_federation):
                server_federation = True
                server_counter = server_counter + 1
                consensus_p.disable_consensus(False) # disable consensus
                W_val_l1, b_val_l1, W_val_l2, b_val_l2 = consensus_p.getFederatedWeight(n_W_l1, n_W_l2, n_b_l1, n_b_l2, epoch, val_loss, args.eps) # model exchange with the server
                if epoch > 0:
                    while not os.path.isfile('server_datamat{}_{}.mat'.format(tot_devices, epoch)):
                        print('server_datamat{}_{}.mat'.format(tot_devices, epoch))
                        pause(1)
                    try:
                        mathcontent = sio.loadmat('server_datamat{}_{}.mat'.format(tot_devices, epoch))
                    except:
                        print('Detected problem while loading file')
                        pause(3)
                        mathcontent = sio.loadmat('server_datamat{}_{}.mat'.format(tot_devices, epoch))

                    W_val_l1 = W_val_l1 + eps_t_control2 * (np.asarray(mathcontent['weights1']) - W_val_l1)  # see paper section 3
                    b_val_l1 = b_val_l1 + eps_t_control2 * (np.squeeze(np.asarray(mathcontent['biases1'])) - b_val_l1)
                    W_val_l2 = W_val_l2 + eps_t_control2 * (np.array(mathcontent['weights2']) - W_val_l2)
                    b_val_l2 = b_val_l2 + eps_t_control2 * (np.squeeze(np.asarray(mathcontent['biases2'])) - b_val_l2)

                    if (server_counter == server_duration):
                        # stop server federation
                        server_federation = False # restart consensus, disable server federation
                        consensus_p.disable_consensus(True)  # enable consensus
                        server_counter = 0
            else:
                server_counter = 0
                consensus_p.disable_consensus(True) #enable consensus
                W_val_l1, b_val_l1, W_val_l2, b_val_l2 = consensus_p.getFederatedWeight(n_W_l1, n_W_l2, n_b_l1, n_b_l2, epoch, val_loss, args.eps)

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
    devices = args.K  # NUMBER OF DE VICES
    neighbors_number = args.N  # NUMBER OF NEIGHBORS PER DEVICE (K-DEGREE NETWORK)
    ii_saved = 0
    federated = False # ENABLE FEDERATED LEARNING)

    training_set_per_device = 10 # NUMBER OF TRAINING SAMPLES PER DEVICE
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
    # server processing
    t.append(multiprocessing.Process(target=processServer, args=(ii+1, federated, devices, validation_train)))
    t[ii+1].start()
    exit(0)
