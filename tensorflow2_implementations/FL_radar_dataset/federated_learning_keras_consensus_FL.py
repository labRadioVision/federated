from DataSets import RadarData
from DataSets_tasks import RadarData_tasks
from consensus.consensus_v3 import CFA_process
from consensus.parameter_server_v2 import Parameter_Server
# use only for consensus , PS only for energy efficiency
# from ReplayMemory import ReplayMemory
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import argparse
import warnings
import glob
import datetime
import scipy.io as sio
# import multiprocessing
import threading
import math
from matplotlib.pyplot import pause
import time



warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-resume', default=0, help="set 1 to resume from a previous simulation, 0 to start from the beginning", type=float)
parser.add_argument('-PS', default=0, help="set 1 to enable PS server and FedAvg, set 0 to disable PS", type=float)
parser.add_argument('-consensus', default=1, help="set 1 to enable consensus, set 0 to disable", type=float)
parser.add_argument('-mu', default=0.00025, help="sets the learning rate for all setups", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFA)", type=float)
parser.add_argument('-target', default=0.1, help="sets the target loss to stop federation", type=float)
parser.add_argument('-K', default=9, help="sets the number of network devices", type=int)
parser.add_argument('-Ka', default=4, help="sets the number of active devices per round in FA (<= K)", type=int)
parser.add_argument('-N', default=1, help="sets the max. number of neighbors per device per round in CFA", type=int)
parser.add_argument('-Ka_consensus', default=8, help="sets the number of active devices for consensus", type=int)
parser.add_argument('-samp', default=15, help="sets the number samples per device", type=int)
parser.add_argument('-noniid_assignment', default=0, help=" set 0 for iid assignment, 1 for non-iid random", type=int)
parser.add_argument('-run', default=0, help=" set the run id", type=int)
parser.add_argument('-random_data_distribution', default=0, help=" set 0 for fixed distribution, 1 for time-varying", type=int)
parser.add_argument('-batches', default=3, help="sets the number of batches per learning round", type=int)
parser.add_argument('-batch_size', default=5, help="sets the batch size per learning round", type=int)
parser.add_argument('-input_data', default='data_mimoradar/data_mmwave_900.mat', help="sets the path to the federated dataset", type=str)
parser.add_argument('-graph', default=6, help="sets the input graph: 0 for default graph, >0 uses the input graph in vGraph.mat, and choose one graph from the available adjacency matrices", type=int)
args = parser.parse_args()

devices = args.K  # NUMBER OF DEVICES
active_devices_per_round = args.Ka
max_epochs = 400


if args.consensus == 1:
    federated = True
    parameter_server = False
elif args.PS == 1:
    federated = False
    parameter_server = True
else: # CL: CENTRALIZED LEARNING ON DEVICE 0 (DATA CENTER)
    federated = False
    parameter_server = False

################# consensus, create the scheduling function ################
scheduling_tx = np.zeros((devices, max_epochs*2), dtype=int)
if parameter_server and not federated:
    indexes_tx = np.zeros((args.Ka, max_epochs*2), dtype=int)
    for k in range(max_epochs*2):
        # inds = np.random.choice(devices, args.Ka, replace=False)
        sr = devices - args.Ka + 1
        sr2 = k % sr
        inds = np.arange(sr2, args.Ka + sr2)
        scheduling_tx[inds, k] = 1
        indexes_tx[:,k] = inds
elif not parameter_server and federated:
    indexes_tx = np.zeros((args.Ka_consensus, max_epochs*2), dtype=int)
    for k in range(max_epochs*2):
        # inds = np.random.choice(devices, args.Ka_consensus, replace=False)
        sr = devices - args.Ka_consensus + 1
        sr2 = k % sr
        inds = np.arange(sr2, args.Ka_consensus + sr2)
        scheduling_tx[inds, k] = 1
        indexes_tx[:, k] = inds
###########################################################################

if active_devices_per_round > devices:
    active_devices_per_round = devices

filepath = args.input_data
target_loss = args.target
# Configuration paramaters for the whole setup
seed = 42

# batch_size = 5  # Size of batch taken from replay buffer
batch_size = args.batch_size
number_of_batches = args.batches
training_set_per_device = args.samp # NUMBER OF TRAINING SAMPLES PER DEVICE
validation_train = 900  # VALIDATION and training DATASET size

if (training_set_per_device > validation_train/args.K):
    training_set_per_device = math.floor(validation_train/args.K)
    print(training_set_per_device)

if batch_size > training_set_per_device:
    batch_size = training_set_per_device

# if batch_size*number_of_batches > training_set_per_device:
#     number_of_batches = math.floor(training_set_per_device/batch_size)

# number_of_batches = int(training_set_per_device/batch_size)
# number_of_batches = args.batches

number_of_batches_for_validation = int(validation_train/batch_size)

print("Number of batches for learning {}".format(number_of_batches))

max_lag = 1 # consensus max delay 2= 2 epochs max
refresh_server = 1 # refresh server updates (in sec)

n_outputs = 6  # 6 classes

validation_start = 1 # start validation in epochs

# Using huber loss for stability
# loss_function = keras.losses.Huber()
# Using crossentropy
loss_function = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy",
)
# save scheduling format
# dict_0 = {"scheduling": scheduling_tx, "devices_scheduling": indexes_tx}
# sio.savemat("results/matlab/CFA_scheduling_devices_{}_neighbors_{}_batches_{}_size{}_noniid{}_run{}.mat".format(devices, args.N, number_of_batches, batch_size, args.noniid_assignment, args.run), dict_0)


# def get_noniid_data(total_training_size, devices, batch_size):
#     samples = np.random.random_integers(batch_size, total_training_size - batch_size * (devices - 1),
#                                         devices)  # create random numbers
#     samples = samples / np.sum(samples, axis=0) * total_training_size  # force them to sum to totals
#     # Ignore the following if you don't need integers
#     samples = np.round(samples)  # transform them into integers
#     remainings = total_training_size - np.sum(samples, axis=0)  # check if there are corrections to be done
#     step = 1 if remainings > 0 else -1
#     while remainings != 0:
#         i = np.random.randint(devices)
#         if samples[i] + step >= 0:
#             samples[i] += step
#             remainings -= step
#     return samples
####

def preprocess_observation(obs, batch_size):
    img = obs# crop and downsize
    img = (img).astype(np.float)
    return img.reshape(batch_size, 256, 63, 1)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(256, 63, 1,))

    # Convolutions
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    classification = layers.Dense(n_outputs, activation="softmax")(layer5)

    return keras.Model(inputs=inputs, outputs=classification)

def processParameterServer(devices, active_devices_per_round, federated, refresh_server=1):
    model_global = create_q_model()
    model_parameters_initial = np.asarray(model_global.get_weights())
    parameter_server = Parameter_Server(devices, model_parameters_initial, active_devices_per_round, indexes_tx)
    global_target_model = 'results/model_global.npy'
    global_epoch = 'results/epoch_global.npy'
    epoch_count = 0
    np.save(global_target_model, model_parameters_initial)
    np.save(global_epoch, epoch_count)
    pause(2) # wait for neighbors
    while True:
        pause(refresh_server) # refresh global model on every xx seconds
        fileList = glob.glob('*.mat', recursive=False)
        if len(fileList) == devices:
            # stop the server
            break
        else:
            np.save(global_target_model, parameter_server.federated_target_weights_aggregation(epoch_count, aggregation_type=0))
            epoch_count += 1
            np.save(global_epoch, epoch_count)


# execute for each deployed device
def processData(device_index, start_samples, samples, federated, full_data_size, number_of_batches, parameter_server, sample_distribution):
    pause(5) # PS server (if any) starts first
    checkpointpath1 = 'results/model{}.h5'.format(device_index)
    outfile = 'results/dump_train_variables{}.npz'.format(device_index)
    outfile_models = 'results/dump_train_model{}.npy'.format(device_index)
    global_model = 'results/model_global.npy'
    global_epoch = 'results/epoch_global.npy'

    # np.random.seed(1)
    # tf.random.set_seed(1)  # common initialization

    learning_rate = args.mu
    learning_rate_local = learning_rate

    B = np.ones((devices, devices)) - tf.one_hot(np.arange(devices), devices)
    Probabilities = B[device_index, :]/(devices - 1)
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

    if parameter_server:
        epoch_global = 0

    training_end = False

    a = model.get_weights()
    # set an arbitrary optimizer, here Adam is used
    optimizer = keras.optimizers.Adam(learning_rate=args.mu, clipnorm=1.0)
    # create a data object (here radar data)
    #start = time.time()
    if args.noniid_assignment == 1:
        data_handle = RadarData_tasks(filepath, device_index, start_samples, samples, full_data_size)
    else:
        data_handle = RadarData(filepath, device_index, start_samples, samples, full_data_size,
                                args.random_data_distribution)
    #end = time.time()
    #time_count = (end - start)
    #print(time_count)
    # create a consensus object
    cfa_consensus = CFA_process(devices, device_index, args.N)

    while True:  # Run until solved
        # collect 1 batch
        frame_count += 1
        obs, labels = data_handle.getTrainingData(batch_size)
        data_batch = preprocess_observation(obs, batch_size)

        # Save data and labels in the current learning session
        data_history.append(data_batch)
        label_history.append(labels)


        if frame_count % number_of_batches == 0:
            if not parameter_server:
                epoch_count += 1
            # check scheduling for federated
            if federated:
                if epoch_count == 1 or scheduling_tx[device_index, epoch_count] == 1:
                    training_signal = False
                else:
                    # stop all computing, just save the previous model
                    training_signal = True
                    model_weights = np.asarray(model.get_weights())
                    model.save(checkpointpath1, include_optimizer=True, save_format='h5')
                    np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                         training_end=training_end, epoch_count=epoch_count, loss=running_loss)
                    np.save(outfile_models, model_weights)
            # check scheduling for parameter server
            if parameter_server:
                while not os.path.isfile(global_epoch):
                    # implementing consensus
                    print("waiting")
                    pause(1)
                try:
                    epoch_global = np.load(global_epoch, allow_pickle=True)
                except:
                    pause(5)
                    print("retrying opening global epoch counter")
                    try:
                        epoch_global = np.load(global_epoch, allow_pickle=True)
                    except:
                        print("failed reading global epoch")

                if epoch_global == 0:
                    training_signal = False

                elif scheduling_tx[device_index, epoch_global] == 1:
                    if epoch_global > epoch_count:
                        epoch_count = epoch_global
                        training_signal = False
                    else:
                        training_signal = True
                else:
                    # stop all computing, just save the previous model
                    training_signal = True

                # always refresh the local model using the PS one
                stop_aggregation = False
                while not os.path.isfile(global_model):
                    # implementing consensus
                    print("waiting")
                    pause(1)
                try:
                    model_global = np.load(global_model, allow_pickle=True)
                except:
                    pause(5)
                    print("retrying opening global model")
                    try:
                        model_global = np.load(global_model, allow_pickle=True)
                    except:
                        print("halting aggregation")
                        stop_aggregation = True

                if not stop_aggregation:
                    model.set_weights(model_global.tolist())

                if training_signal:
                    model_weights = np.asarray(model.get_weights())
                    model.save(checkpointpath1, include_optimizer=True, save_format='h5')
                    np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                             training_end=training_end, epoch_count=epoch_count, loss=running_loss)
                    np.save(outfile_models, model_weights)
            # check schedulting for parameter server

        # Local learning update every "number of batches" batches
        time_count = 0
        if frame_count % number_of_batches == 0 and not training_signal:
            # run local batches
            for i in range(number_of_batches):
                start = time.time()
                data_sample = np.array(data_history[i])
                label_sample = np.array(label_history[i])

                # Create a mask to calculate loss
                masks = tf.one_hot(label_sample, n_outputs)

                with tf.GradientTape() as tape:
                    # Train the model on data samples
                    classes = model(data_sample, training=False)
                    # Apply the masks
                    # for k in range(batch_size):
                    #     class_v[k] = tf.argmax(classes[k])
                    # class_v = tf.reduce_sum(tf.multiply(classes, masks), axis=1)
                    # Take best action

                    # Calculate loss
                    loss = loss_function(masks, classes)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                end = time.time()
                time_count = time_count + (end-start)/number_of_batches
            if not parameter_server and not federated:
                print('Average batch training time {:.2f}'.format(time_count))
            del data_history
            del label_history
            data_history = []
            label_history = []

            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                     training_end=training_end, epoch_count=epoch_count, loss=running_loss)
            np.save(outfile_models, model_weights)


            #  Consensus round
            # update local model

            cfa_consensus.update_local_model(model_weights)
            # neighbor = cfa_consensus.get_connectivity(device_index, args.N, devices) # fixed neighbor
            np.random.seed(1)
            tf.random.set_seed(1)  # common initialization
            if not train_start:
                if federated and not training_signal:
                    eps_c = 1 / (args.N + 1)
                    # apply consensus for model parameter
                    # neighbor = np.random.choice(np.arange(devices), args.N, p=Probabilities, replace=False) # choose neighbor
                    neighbor = np.random.choice(indexes_tx[:, epoch_count - 1], args.N, replace=False) # choose neighbor
                    while neighbor == device_index:
                        neighbor = np.random.choice(indexes_tx[:, epoch_count - 1], args.N,
                                                    replace=False)  # choose neighbor
                    print("Consensus from neighbor {} for device {}, local loss {:.2f}".format(neighbor, device_index,
                                                                                               loss.numpy()))

                    model.set_weights(cfa_consensus.federated_weights_computing(neighbor, args.N, epoch_count, eps_c, max_lag))
                    if cfa_consensus.getTrainingStatusFromNeightbor():
                        # a neighbor completed the training, with loss < target, transfer learning is thus applied (the device will copy and reuse the same model)
                        training_signal = True # stop local learning, just do validation
            else:
                print("Consensus warm up")
                train_start = False

            # check if parameter server is enabled
            # stop_aggregation = False

            # if parameter_server:
            #     # pause(refresh_server)
            #     while not os.path.isfile(global_model):
            #         # implementing consensus
            #         print("waiting")
            #         pause(1)
            #     try:
            #         model_global = np.load(global_model, allow_pickle=True)
            #     except:
            #         pause(5)
            #         print("retrying opening global model")
            #         try:
            #             model_global = np.load(global_model, allow_pickle=True)
            #         except:
            #             print("halting aggregation")
            #             stop_aggregation = True
            #
            #     if not stop_aggregation:
            #         # print("updating from global model inside the parmeter server")
            #         for k in range(cfa_consensus.layers):
            #             # model_weights[k] = model_weights[k]+ 0.5*(model_global[k]-model_weights[k])
            #             model_weights[k] = model_global[k]
            #         model.set_weights(model_weights.tolist())
            #
            #     while not os.path.isfile(global_epoch):
            #         # implementing consensus
            #         print("waiting")
            #         pause(1)
            #     try:
            #         epoch_global = np.load(global_epoch, allow_pickle=True)
            #     except:
            #         pause(5)
            #         print("retrying opening global epoch counter")
            #         try:
            #             epoch_global = np.load(global_epoch, allow_pickle=True)
            #         except:
            #             print("halting aggregation")

            del model_weights


        #start = time.time()
        # validation tool for device 'device_index'
        if epoch_count > validation_start and frame_count % number_of_batches == 0:
            avg_cost = 0.
            for i in range(number_of_batches_for_validation):
                obs_valid, labels_valid = data_handle.getTestData(batch_size, i)
                # obs_valid, labels_valid = data_handle.getRandomTestData(batch_size)
                data_valid = preprocess_observation(np.squeeze(obs_valid), batch_size)
                data_sample = np.array(data_valid)
                label_sample = np.array(labels_valid)
                # Create a mask to calculate loss
                masks = tf.one_hot(label_sample, n_outputs)
                classes = model(data_sample, training=False)
                # Apply the masks
                # class_v = tf.reduce_sum(tf.multiply(classes, masks), axis=1)
                # class_v = np.zeros(batch_size, dtype=int)
                # for k in range(batch_size):
                #     class_v[k] = tf.argmax(classes[k]).numpy()
                # Calculate loss
                # loss = loss_function(label_sample, classes)
                loss = loss_function(masks, classes).numpy()
                avg_cost += loss / number_of_batches_for_validation  # Training loss
            epoch_loss_history.append(avg_cost)
            print("Device {} epoch count {}, validation loss {:.2f}".format(device_index, epoch_count,
                                                                                 avg_cost))
            # mean loss for last 5 epochs
            running_loss = np.mean(epoch_loss_history[-1:])
        #end = time.time()
        #time_count = (end - start)
        #print(time_count)

        if running_loss < target_loss:  # Condition to consider the task solved
            print("Solved for device {} at epoch {} with average loss {:.2f} !".format(device_index, epoch_count, running_loss))
            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            # model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                     training_end=training_end, epoch_count=epoch_count, loss=running_loss)
            np.save(outfile_models, model_weights)

            if federated:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                      "parameter_server": parameter_server, "devices": devices, "neighbors": args.N,
                      "active_devices": args.Ka_consensus,
                      "batches": number_of_batches, "batch_size": batch_size, "samples": samples, "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}
            elif parameter_server:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                          "parameter_server": parameter_server, "devices": devices,
                          "active_devices": active_devices_per_round,
                          "batches": number_of_batches, "batch_size": batch_size, "samples": samples,
                          "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}
            else:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                          "parameter_server": parameter_server, "devices": devices,
                          "batches": number_of_batches, "batch_size": batch_size, "samples": samples,
                          "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}

            if federated:
                sio.savemat(
                    "results/matlab/CFA_device_{}_samples_{}_devices_{}_active_{}_neighbors_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(
                        device_index, samples, devices, args.Ka_consensus, args.N, number_of_batches, batch_size, args.noniid_assignment, args.run, args.random_data_distribution), dict_1)
                sio.savemat(
                    "CFA_device_{}_samples_{}_devices_{}_neighbors_{}_batches_{}_size{}.mat".format(
                        device_index, samples, devices, args.N, number_of_batches, batch_size), dict_1)
            elif parameter_server:
                sio.savemat(
                    "results/matlab/FA_device_{}_samples_{}_devices_{}_active_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(
                        device_index, samples, devices, active_devices_per_round, number_of_batches, batch_size, args.noniid_assignment,args.run, args.random_data_distribution), dict_1)
                sio.savemat(
                    "FA_device_{}_samples_{}_devices_{}_active_{}_batches_{}_size{}.mat".format(
                        device_index, samples, devices, active_devices_per_round, number_of_batches, batch_size), dict_1)
            else: # CL
                sio.savemat(
                    "results/matlab/CL_samples_{}_devices_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(samples, devices, number_of_batches, batch_size,
                                                                                                                         args.noniid_assignment, args.run, args.random_data_distribution), dict_1)
            break

        if epoch_count > max_epochs:  # stop simulation
            print("Unsolved for device {} at epoch {}!".format(device_index, epoch_count))
            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            # model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, epoch_loss_history=epoch_loss_history,
                     training_end=training_end, epoch_count=epoch_count, loss=running_loss)
            np.save(outfile_models, model_weights)

            if federated:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                          "parameter_server": parameter_server, "devices": devices, "neighbors": args.N,
                          "active_devices": args.Ka_consensus,
                          "batches": number_of_batches, "batch_size": batch_size, "samples": samples,
                          "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}
            elif parameter_server:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                          "parameter_server": parameter_server, "devices": devices,
                          "active_devices": active_devices_per_round,
                          "batches": number_of_batches, "batch_size": batch_size, "samples": samples,
                          "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}
            else:
                dict_1 = {"epoch_loss_history": epoch_loss_history, "federated": federated,
                          "parameter_server": parameter_server, "devices": devices,
                          "batches": number_of_batches, "batch_size": batch_size, "samples": samples,
                          "noniid": args.noniid_assignment, "data_distribution": args.random_data_distribution}

            if federated:
                sio.savemat(
                    "results/matlab/CFA_device_{}_samples_{}_devices_{}_active_{}_neighbors_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(
                        device_index, samples, devices, args.Ka_consensus, args.N, number_of_batches, batch_size,
                        args.noniid_assignment, args.run, args.random_data_distribution), dict_1)
                sio.savemat(
                    "CFA_device_{}_samples_{}_devices_{}_neighbors_{}_batches_{}_size{}.mat".format(
                        device_index, samples, devices, args.N, number_of_batches, batch_size), dict_1)
            elif parameter_server:
                sio.savemat(
                    "results/matlab/FA_device_{}_samples_{}_devices_{}_active_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(
                        device_index, samples, devices, active_devices_per_round, number_of_batches, batch_size,
                        args.noniid_assignment, args.run, args.random_data_distribution), dict_1)
                sio.savemat(
                    "FA_device_{}_samples_{}_devices_{}_active_{}_batches_{}_size{}.mat".format(
                        device_index, samples, devices, active_devices_per_round, number_of_batches, batch_size),
                    dict_1)
            else:  # CL
                sio.savemat(
                    "results/matlab/CL_samples_{}_devices_{}_batches_{}_size{}_noniid{}_run{}_distribution{}.mat".format(
                        samples, devices, number_of_batches, batch_size,
                        args.noniid_assignment, args.run, args.random_data_distribution), dict_1)
            break


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

    # main loop for multiprocessing
    t = []

    ############# enable consensus based federation #######################
    # federated = False
    # federated = True
    ########################################################

    ##################### enable parameter server ##############
    # parameter_server = False
    server_index = devices
    # parameter_server = True
    #########################################################

    samples = np.zeros(devices)  # training samples per device
    for id in range(devices):
       # samples[id]=math.floor(w[id]*validation_train)
       # samples[id] = math.floor(balancing_vect[id]*fraction_training)
       samples[id] = training_set_per_device
    # samples = int(fraction_training/devices) # training samples per device

    ######################### Create a non-iid assignment  ##########################
    # if args.noniid_assignment == 1:
    #     total_training_size = training_set_per_device * devices
    #     samples = get_noniid_data(total_training_size, devices, batch_size)
    #     while np.min(samples) < batch_size:
    #         samples = get_noniid_data(total_training_size, devices, batch_size)
    #############################################################################
    print(samples)

    ####################################   code testing CL learning (0: data center)
    # federated = False
    # parameter_server = False
    # processData(0, validation_train, federated, validation_train, number_of_batches, parameter_server)
    ######################################################################################

    if federated or parameter_server:
        for ii in range(devices):
            # position start
            if ii == 0:
                start_index = 0
            else:
                start_index = start_index + int(samples[ii-1])
            t.append(threading.Thread(target=processData, args=(ii, start_index, int(samples[ii]), federated, validation_train, number_of_batches, parameter_server, samples)))
            t[ii].start()

        # last process is for the target server
        if parameter_server:
            print("Target server starting with active devices {}".format(active_devices_per_round))
            t.append(threading.Thread(target=processParameterServer, args=(devices, active_devices_per_round, federated)))
            t[devices].start()
    else: # run centralized learning on device 0 (data center)
        processData(0, 0, training_set_per_device*devices, federated, validation_train, number_of_batches, parameter_server, samples)

    exit(0)
