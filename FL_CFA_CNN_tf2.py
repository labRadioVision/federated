from __future__ import absolute_import, division, print_function, unicode_literals
from keras.utils import to_categorical
from consensus.cfa_ongraphs import CFA_process
# from consensus.cfa_ge_2stage import CFA_ge_process # for static, k regular networks only, see paper
import numpy as np
# import tensorflow as tf # tf 1.13
import tensorflow.compat.v1 as tf
import datetime
import scipy.io as sio
import multiprocessing
import math
from matplotlib.pyplot import pause
import os
import glob
import argparse
import warnings
import time
import datetime
import random

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-mu', default=0.025, help="sets the learning rate for local SGD", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFA)", type=float)
parser.add_argument('-K', default=5, help="sets the number of network devices (change connectivity matrix in vGraph.mat for increasing the number of devices)", type=int)
parser.add_argument('-N', default=2, help="sets the max. number of neighbors per device per round", type=int)
parser.add_argument('-T', default=60, help="sets the number of training epochs", type=int)
parser.add_argument('-samp', default=15, help="sets the number samples per device", type=int)
parser.add_argument('-input_data', default='dati_mimoradar/data_mmwave_900.mat', help="sets the path to the federated dataset", type=str)
parser.add_argument('-rand', default=1, help="sets static or random choice of the N neighbors on every new round (0 static, 1 random)", type=int)
parser.add_argument('-consensus_mode', default=0, help="0: combine one neighbor at a time and run sgd AFTER every new combination; 1 (faster): combine all neighbors on a single stage, run one sgd after this combination", type=int)
parser.add_argument('-graph', default=6, help="sets the input graph: 0 for default graph, >0 uses the input graph in vGraph.mat, and choose one graph from the available adjacency matrices", type=int)
parser.add_argument('-compression', default=2, help="sets the compression factor for communication: 0 no compression, 1, sparse, 2 sparse + dpcm, 3 sparse (high compression factor), 4 sparse + dpcm (high compression factor)", type=int)
args = parser.parse_args()

# Parameters for learning rate optimization and batch size ##################
tf.disable_v2_behavior() # tf 2
learning_rate = args.mu
training_epochs = args.T
compression = args.compression
if args.rand == 1:
    randomized = True
else:
    randomized = False

batch_size = 5
display_step = 10
training_set_per_device = args.samp # NUMBER OF TRAINING SAMPLES PER DEVICE
validation_train = 900  # VALIDATION and training DATASET size
# validation_train = 450
if (training_set_per_device > validation_train/args.K):
    training_set_per_device = math.floor(validation_train/args.K)
    print(training_set_per_device)


# convolutional 1D parameters
filter = 3
number = 4
pooling = 2
stride = 2
multip = 64*16
input_data = 512
input_data1 = 256
input_data2 = 63
classes = 6

#############################################################################

def conv1d(x, W, b, strides=1):
    # Conv1D wrapper, with bias and relu activation
    x = tf.expand_dims(x, 2)
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv2d_f(x, W, b, strides=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.expand_dims(x, 3)
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


def processData(samples, iii, federated, tot_devices,fraction_training, neighbors_number, EPOCH_THRESHOLD):
    # eng = matlab.engine.start_matlab()
    eng = 0
    global learning_rate
    learning_rate_local = learning_rate
    np.random.seed(1)
    tf.set_random_seed(1)  # common initialization tf 1.13
    # tf.random.set_seed(1)

    # database = sio.loadmat('dati_mimoradar/data_mmwave_900.mat')
    database = sio.loadmat(args.input_data)
    # database = sio.loadmat('dati_mimoradar/data_mmwave_450.mat')
    x_train = database['mmwave_data_train']
    y_train = database['label_train']
    y_train_t = to_categorical(y_train)
    x_train = (x_train.astype('float32').clip(0)) / 1000 # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)
    x_train2 = x_train[iii * samples:((iii + 1) * samples - 1), :, :] # DATA PARTITION
    y_train2 = y_train_t[iii * samples:((iii + 1) * samples - 1), :]

    x_test = database['mmwave_data_test']
    y_test = database['label_test']
    x_test = (x_test.astype('float32').clip(0)) / 1000
    y_test_t = to_categorical(y_test)

    total_batch2 = int(fraction_training / batch_size)
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, input_data1, input_data2])  # 512 POINT FFT RANGE MEASUREMENTS
    y = tf.placeholder(tf.float32, [None, classes])  # 0-7 HR distances (safe - unsafe)

    W_ext_l1 = tf.placeholder(tf.float32, [filter, filter, 1, number])
    b_ext_l1 = tf.placeholder(tf.float32, [number])
    W_ext_l2 = tf.placeholder(tf.float32, [multip * number, classes])
    b_ext_l2 = tf.placeholder(tf.float32, [classes])

    W2_ext_l1 = tf.placeholder(tf.float32, [filter, filter, 1, number])
    b2_ext_l1 = tf.placeholder(tf.float32, [number])
    W2_ext_l2 = tf.placeholder(tf.float32, [multip * number, classes])
    b2_ext_l2 = tf.placeholder(tf.float32, [classes])

    # Set model weights
    W_l1 = tf.Variable(tf.random_normal([filter, filter, 1, number]))
    b_l1 = tf.Variable(tf.random_normal([number]))
    W_l2 = tf.Variable(tf.zeros([multip * number, classes]))
    b_l2 = tf.Variable(tf.zeros([classes]))

    # Construct model Layer #1 CNN 1d, Layer #2 FC
    hidden0 = conv2d_f(x, W_ext_l1, b_ext_l1)
    hidden01 = tf.layers.max_pooling2d(hidden0, pool_size=stride, strides=stride, padding='SAME')
    # print(hidden01) # check hidden01 size
    # hidden01 = tf.nn.max_pool1d(hidden0, ksize=stride, strides=stride, padding='SAME')
    fc01 = tf.reshape(hidden01, [-1, multip*number])
    pred = tf.nn.softmax(tf.matmul(fc01, W_ext_l2) + b_ext_l2)  # example 2 layers

    hidden2 = conv2d_f(x, W2_ext_l1, b2_ext_l1)
    hidden02 = tf.layers.max_pooling2d(hidden2, pool_size=stride, strides=stride, padding='SAME')
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
    consensus_p = CFA_process(federated, tot_devices, iii, neighbors_number, args.graph, compression, args.consensus_mode)
    neighbor_vector = consensus_p.getMobileNetwork_connectivity(iii, neighbors_number, tot_devices, args.graph - 1)
    # print(neighbor_vector.size)

    #    Start training
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(samples / batch_size)
        # PRINTS THE TOTAL NUMBER OF MINI BATCHES
        # print(total_batch)

        # Training cycle
        val_loss = np.zeros(training_epochs)
        param_vector = np.ones(training_epochs)
        timings = np.ones(training_epochs)
        sgd_computational_time = np.ones(training_epochs)
        compression_computational_time = np.ones(training_epochs)
        for epoch in range(training_epochs):
            # changing neighbors on every round if randomized = true
            if randomized:
                neighbor_vector = consensus_p.getMobileNetwork_connectivity(iii, neighbors_number, tot_devices,
                                                                        args.graph - 1)
            for current_neighbor in range(neighbor_vector.size + 1):
                avg_cost = 0.
                avg_cost_test = 0.
                ######## sgd on local data
                start_time = time.time()
                ################
                for i in range(total_batch):
                    batch_xs = x_train2[i * batch_size:((i + 1) * batch_size - 1), :, :]
                    batch_ys = y_train2[i * batch_size:((i + 1) * batch_size - 1), :]
                    if (i == 0) and (epoch == 0):  # initialization
                        # W_val_l1 = np.zeros([512, 32])
                        W_val_l1 = np.random.normal(0.0, 1.0, (filter, filter, 1, number))
                        # b_val_l1 = np.zeros([32])
                        b_val_l1 = np.random.normal(0.0, 1.0, number)
                        W_val_l2 = np.zeros([multip * number, classes])
                        b_val_l2 = np.zeros([classes])
                    elif (i > 0):
                        W_val_l1 = n_W_l1  # modify for minibatch updates
                        b_val_l1 = n_b_l1
                        W_val_l2 = n_W_l2  # modify for minibatch updates
                        b_val_l2 = n_b_l2
                    # Fit training using batch data
                    n_W_l1, n_b_l1, n_W_l2, n_b_l2, c, g_W_l1, g_b_l1, g_W_l2, g_b_l2 = sess.run([new_W_l1, new_b_l1,
                                                                                              new_W_l2, new_b_l2, cost,
                                                                                              grad_W_l1, grad_b_l1,
                                                                                              grad_W_l2, grad_b_l2],
                                                                                             feed_dict={x: batch_xs,
                                                                                                        y: batch_ys,
                                                                                                        W_ext_l1: W_val_l1,
                                                                                                        b_ext_l1: b_val_l1,
                                                                                                        W_ext_l2: W_val_l2,
                                                                                                        b_ext_l2: b_val_l2})
                    avg_cost += c / total_batch  # Training loss
                    #################à
                sgd_computational_time[epoch] = sgd_computational_time[epoch] + time.time() - start_time
                ###################
                # validation
                with tf.Session() as sess2:
                    sess2.run(init)
                    for i in range(total_batch2):
                        # Construct model
                        batch_xs = x_test[i * batch_size:((i + 1) * batch_size - 1), :, :]
                        batch_ys = y_test_t[i * batch_size:((i + 1) * batch_size - 1), :]
                        c = sess2.run(cost2, feed_dict={x: batch_xs,
                                                    y: batch_ys, W2_ext_l1: n_W_l1, b2_ext_l1: n_b_l1,
                                                    W2_ext_l2: n_W_l2, b2_ext_l2: n_b_l2})
                        avg_cost_test += c / total_batch2

                val_loss[epoch] = avg_cost_test
                if epoch == 0:
                    param_vector[epoch] = multip * number * classes
                else:
                    param_vector[epoch] = counter_param

                print('Test Device: ' + str(iii) + ' Neighbor counter: ' + str(current_neighbor) + " Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost_test))

                ###########################################################
                # CFA: weights exchange (no gradients)
                # start_time = time.time()
                if args.consensus_mode == 0:
                    # combine one at a time and run sgd after every combination
                    if current_neighbor < neighbor_vector.size:
                        stop_consensus = False
                        W_val_l1, b_val_l1, W_val_l2, b_val_l2, counter_param, time_info, compression_time = consensus_p.getFederatedWeight(
                            n_W_l1, n_W_l2, n_b_l1, n_b_l2,
                            epoch, val_loss, args.eps, neighbor_vector[current_neighbor], stop_consensus)
                        timings[epoch] = timings[epoch] + time_info
                    else: # transmission of model parameters
                        stop_consensus = True
                        W_val_l1, b_val_l1, W_val_l2, b_val_l2, counter_param, time_info, compression_time = consensus_p.getFederatedWeight(
                            n_W_l1, n_W_l2, n_b_l1, n_b_l2,
                            epoch, val_loss, args.eps, [], stop_consensus)
                elif args.consensus_mode == 1:
                    # sets an alternative implementation, combine all and run one SGD
                    if current_neighbor == 0:
                        stop_consensus = False
                        W_val_l1, b_val_l1, W_val_l2, b_val_l2, counter_param, time_info, compression_time = consensus_p.getFederatedWeight(
                            n_W_l1, n_W_l2, n_b_l1, n_b_l2,
                            epoch, val_loss, args.eps, neighbor_vector, stop_consensus)
                        timings[epoch] = timings[epoch] + time_info
                    else:
                        stop_consensus = True # enable transmission of model only, use as neighbors an empty
                        W_val_l1, b_val_l1, W_val_l2, b_val_l2, counter_param, time_info, compression_time = consensus_p.getFederatedWeight(
                            n_W_l1, n_W_l2, n_b_l1, n_b_l2,
                            epoch, val_loss, args.eps, [], stop_consensus)
                        break
                ###############################################################
            compression_computational_time[epoch] = compression_time
         ###########################################################

        print("Optimization Finished!")
        # DUMP RESULTS %Y-%m-%d-%H-%M-%S
        sio.savemat(
            'results/dump_loss_g{}_n{}_c{}_m{}_con{}_rand{}_{}.mat'.format(args.graph, iii, compression, neighbors_number, args.consensus_mode, args.rand, time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())), {
                "val_acc": val_loss, "device": iii,"T_epochs": training_epochs, "T_set_per_device":training_set_per_device, "samples":samples,
                "param_vector":param_vector, "compression_method":compression, "execution_time":timings, "compression_computational_time":compression_computational_time,
                "sgd_computational_time":sgd_computational_time})
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
    EPOCH_THRESHOLD = 4  # STARTING EPOCH FOR CFA-GE (2-STAGE NEGOTIATION)
    federated = True # ENABLE FEDERATED LEARNING)

    #fraction_training = int(devices*training_set_per_device) # total training
    b_v = 1/devices
    balancing_vect = np.ones(devices)*b_v
    samples = np.zeros(devices) # training samples per device
    ###################################################################################
    w = [random.random() for i in range(0, devices)]
    s = sum(w)
    w = [i / s for i in w]
   # START MULTIPROCESSING
    for id in range(devices):
       # samples[id]=math.floor(w[id]*validation_train)
       # samples[id] = math.floor(balancing_vect[id]*fraction_training)
       samples[id] = training_set_per_device
    # samples = int(fraction_training/devices) # training samples per device
    print(samples)
    t = []
    iii = 0
    for ii in range(devices):
        t.append(multiprocessing.Process(target=processData, args=(int(samples[ii]), ii, federated, devices, validation_train, neighbors_number, EPOCH_THRESHOLD)))
        t[ii].start()
    exit(0)
