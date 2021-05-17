from RobotTrajectory import RobotTrajectory
from consensus.consensus_v2 import CFA_process
from consensus.target_server_v2 import Target_Server

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
import time
import datetime
import scipy.io as sio
import multiprocessing
import math
from matplotlib.pyplot import pause



warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-resume', default=0, help="set 1 to resume from a previous simulation, 0 to start from the beginning (NOT WORKING resume 1)", type=float)
parser.add_argument('-consensus', default=1, help="sets FRL using consensus", type=int)
parser.add_argument('-PS', default=0, help="sets FRL with target model server", type=int)
parser.add_argument('-isolated', default=0, help="disable FRL ", type=int)
parser.add_argument('-centralized', default=0, help="centralized RL", type=int)
parser.add_argument('-update_federation', default=20, help="counts the number of frames (robot movements) per epoch", type=int)
parser.add_argument('-run', default=0, help="run number", type=int)
parser.add_argument('-target_reward', default=50, help="run number", type=int)
parser.add_argument('-mu', default=0.00025, help="sets the learning rate for DQL", type=float)
parser.add_argument('-eps', default=1, help="sets the mixing parameters for model averaging (CFL)", type=float)
parser.add_argument('-K', default=5, help="sets the number of network devices", type=int)
parser.add_argument('-N', default=1, help="sets the max. number of neighbors per device per round", type=int)
parser.add_argument('-pos', default=100, help="sets the maximum total number of explorable positions: pos/K gives the number of explored positions per device", type=int)
parser.add_argument('-true_pos', default=35, help="sets the number of explorable positions in the workspace", type=int)
parser.add_argument('-input_data', default='dataset_trajectories/data_robots2.mat', help="sets the path to the federated dataset, to compute new observations and rewards for input actions ", type=str)
parser.add_argument('-input_table', default='dataset_trajectories/lookuptab2.mat', help="sets the path to the lookup table to compute robot trajectories", type=str)
parser.add_argument('-input_rewards', default='dataset_trajectories/rewards2.mat', help="sets the path to the input rewards per robot position", type=str)
parser.add_argument('-rand', default=1, help="sets static or random choice of the N neighbors on every new round (0 static, 1 random)", type=int)
#parser.add_argument('-consensus_mode', default=0, help="0: combine one neighbor at a time and run sgd AFTER every new combination; 1 (faster): combine all neighbors on a single stage, run one sgd after this combination", type=int)
#parser.add_argument('-graph', default=6, help="sets the input graph: 0 for default graph, >0 uses the input graph in vGraph.mat, and choose one graph from the available adjacency matrices", type=int)
args = parser.parse_args()

devices = args.K  # NUMBER OF DEVICES
number_positions = args.pos

filepath = args.input_data
lookuptab = args.input_table
filerewards = args.input_rewards

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards

batch_size = 32  # Size of batch taken from replay buffer

n_outputs = 4  # 4 robot movements are available: 1 backward, 2 forward, 3 turn right, 4 turn left
num_actions = n_outputs

# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 2000000.0
# Train the model after 4 actions
update_after_actions = 4 # other option 2
# How often to update the target network
update_target_network = 2000 # other option 500
# How often to update the target server
update_target_server = 500
target_reward = args.target_reward
# How often to update the consensus process
update_consensus = args.update_federation
#update_consensus = 100 # PREVIOUSLY 8 (upate consensus or plain FL with PS
# max number of episodes
max_episodes = 10000
# Maximum replay length
max_memory_length = 50000 # previous 100000
# Using huber loss for stability
loss_function = keras.losses.Huber()
# max reward for last frames
max_reward = 80



def preprocess_observation(obs):
    img = obs# crop and downsize
    img = (img).astype(np.float)
    return img.reshape(80, 88, 1)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(80, 88, 1,))

    # Convolutions on the camera frames
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(n_outputs, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

# frame_count_global = np.zeros(devices, dtype=int)

# The model makes the predictions for Q-values which are used to make a action.
# Target model is used for the prediction of future rewards.
# Consensus and FL are applied on the model parameters

def processTargetServer(devices, federated):
    model_target_global = create_q_model()
    model_target_parameters_initial = 0 * np.asarray(model_target_global.get_weights()) # overwrite zero initialization
    target_server = Target_Server(devices, model_target_parameters_initial)
    global_target_model = 'results/model_target_global.npy'
    np.save(global_target_model, model_target_parameters_initial)
    pause(5)
    while True:
        pause(1)
        fileList = glob.glob('*.mat', recursive=False)
        if len(fileList) == devices:
            # stop the server
            break
        else:
            np.save(global_target_model, target_server.federated_target_weights_aggregation(epoch=0))


# execute for each deployed device
def processData(device_index, number_positions_devices, federated, target_server, initialization, update_after_actions):
    pause(5) # server start first
    checkpointpath1 = 'results/model{}.h5'.format(device_index)
    checkpointpath2 = 'results/model_target{}.h5'.format(device_index)
    outfile = 'results/dump_train_variables{}.npz'.format(device_index)
    outfile_models = 'results/dump_train_model{}.npy'.format(device_index)
    global_target_model = 'results/model_target_global.npy'

    if args.centralized == 0:
        max_steps_per_episode = number_positions_devices  # other option 1000
    else:
        max_steps_per_episode = number_positions_devices * devices  # other option 1000

    n_file_cfa = "models_saved/CFA_robot_{}_number_{}_neighbors_{}_explored_pos_{}_update_{}_run_{}.mat".format(device_index,
                                                                                                   devices,
                                                                                                   args.N,
                                                                                                   number_positions_devices,
                                                                                                   update_consensus,
                                                                                                   args.run)
    n_file_cfa_h5 = "models_saved/CFA_robot_{}_number_{}_neighbors_{}_explored_pos_{}_update_{}_run_{}.h5".format(device_index,
                                                                                                   devices,
                                                                                                   args.N,
                                                                                                   number_positions_devices,
                                                                                                   update_consensus,
                                                                                                   args.run)
    n_file_fa = "models_saved/FA_robot_{}_number_{}_explored_pos_{}_update_{}_run_{}.mat".format(device_index, devices,
                                                                                                 number_positions_devices,
                                                                                                 update_consensus,
                                                                                                 args.run)
    n_file_fa_h5 = "models_saved/FA_robot_{}_number_{}_explored_pos_{}_update_{}_run_{}.h5".format(device_index, devices,
                                                                                                 number_positions_devices,
                                                                                                 update_consensus,
                                                                                                 args.run)
    n_file_cl = "models_saved/CL_datacenter_{}_number_{}_explored_pos_{}_update_{}_run{}.mat".format(device_index, devices,
                                                                                              number_positions_devices,
                                                                                              update_after_actions,
                                                                                              args.run)
    n_file_cl_h5 = "models_saved/CL_datacenter_{}_number_{}_explored_pos_{}_update_{}_run{}.h5".format(device_index, devices,
                                                                                              number_positions_devices,
                                                                                              update_after_actions,
                                                                                              args.run)
    n_file_isolated = "models_saved/Isolated_robot_{}_number_{}_explored_pos_{}_update_{}_run_{}.mat".format(
        device_index, devices,
        number_positions_devices,
        update_after_actions,
        args.run)

    n_file_isolated_h5 = "models_saved/Isolated_robot_{}_number_{}_explored_pos_{}_update_{}_run_{}.h5".format(
        device_index, devices,
        number_positions_devices,
        update_after_actions,
        args.run)

    #np.random.seed(1)
    #tf.random.set_seed(1)  # common initialization


    B = np.ones((devices, devices)) - tf.one_hot(np.arange(devices), devices)
    Probabilities = B[device_index, :]/(devices - 1)
    training_signal = False

    # check for backup variables
    if os.path.isfile(checkpointpath1):
        train_start = False

        # backup the model and the model target
        model = models.load_model(checkpointpath1)
        model_target = models.load_model(checkpointpath2)
        local_model_parameters = np.load(outfile_models, allow_pickle=True)
        model.set_weights(local_model_parameters.tolist())

        dump_vars = np.load(outfile, allow_pickle=True)
        frame_count = dump_vars['frame_count']
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = dump_vars['episode_reward_history'].tolist()
        running_reward = dump_vars['running_reward']
        episode_count = dump_vars['episode_count']
        epsilon = dump_vars['epsilon']
    else:
        train_start = True
        model = create_q_model()
        model_target = create_q_model()
        frame_count = 0
        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        epsilon = 1.0  # Epsilon greedy parameter

    training_end = False
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_min_validation = 0.001 # for validation only
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken

    print("End loading file")
    print("Set epsilon to {} for device {}".format(epsilon, device_index))

    optimizer = keras.optimizers.Adam(learning_rate=args.mu, clipnorm=1.0)
    file_size = number_positions
    robot_trajectory = RobotTrajectory(filepath, lookuptab, filerewards, file_size, number_positions_devices)
    robot_trajectory_validation = RobotTrajectory(filepath, lookuptab, filerewards, file_size, number_positions)
    cfa_consensus = CFA_process(devices, device_index, args.N)
    inizialization_index = 0

    while True:  # Run until solved
        # state = np.array(env.reset())

        if args.centralized  == 0:
            [obs, reward, done] = robot_trajectory.initialize(position_initial=initialization)
        else: # CL learning
            [obs, reward, done] = robot_trajectory.initialize(position_initial=initialization[inizialization_index])
            #inizialization_index += 1
            #if inizialization_index % devices == 0:
            #    inizialization_index = 0

        state = preprocess_observation(np.squeeze(obs))
        # episode_reward = reward
        # neighbor = cfa_consensus.get_connectivity(device_index, args.N, devices)
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1
            if args.centralized == 1: # check data obtained from multiple devices
                if timestep % number_positions_devices == 0: # reinitialize
                    inizialization_index += 1
                    if inizialization_index % devices == 0:
                        inizialization_index = 0
                    [obs, reward, done] = robot_trajectory.initialize(position_initial=initialization[inizialization_index])
                    state = preprocess_observation(np.squeeze(obs))


            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(n_outputs)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            dev = 0
            [obs, reward, done] = robot_trajectory.implement(action, dev)
            state_next = preprocess_observation(np.squeeze(obs))
            state_next = np.array(state_next)

            # episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)

            # Let's memorize what happened
            # replay_memory.append((state, action, reward, state_next, 1.0 - done))

            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size and not training_signal:
                # start = time.time()
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)

                # Bellman equation
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1 or max_reward
                # updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                updated_q_values = updated_q_values * (1 - done_sample) + done_sample*max_reward # test
                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                #end = time.time()
                #print("Time for 1 minibatch (32 observations): {}".format(end-start))


            if frame_count % update_consensus == 0 and len(done_history) > batch_size:
                model_weights = np.asarray(model.get_weights())
                # update local model
                cfa_consensus.update_local_model(model_weights)
                # neighbor = cfa_consensus.get_connectivity(device_index, args.N, devices) # fixed neighbor
                # neighbor = np.random.choice(np.arange(devices), args.N, p=Probabilities) # choose neighbor
                neighbor = np.random.choice(np.arange(devices), args.N)
                while device_index == neighbor:
                    neighbor = np.random.choice(np.arange(devices), args.N)

                if not train_start:
                    print("Episode {}, frame count {}, running reward: {:.2f}, loss {:.2f}".format(episode_count,
                                                                                                   frame_count,
                                                                                                   running_reward,
                                                                                                   loss.numpy()))
                    if federated and not training_signal:
                        print(
                            "Neighbor {} for device {} at episode {} and frame_count {}".format(
                                neighbor, device_index, episode_count, frame_count))

                        eps_c = 1 / (args.N + 1)
                        # apply consensus for model parameter
                        model.set_weights(cfa_consensus.federated_weights_computing(neighbor, args.N, frame_count, eps_c, update_consensus))
                        if cfa_consensus.getTrainingStatusFromNeightbor():
                            training_signal = True
                    elif target_server:
                        stop_aggregation = False
                        while not os.path.isfile(global_target_model):
                         # implementing consensus
                         print("waiting")
                         pause(1)
                        try:
                             model_target_global = np.load(global_target_model, allow_pickle=True)
                        except:
                            pause(5)
                            print("retrying opening target model")
                            try:
                                model_target_global = np.load(global_target_model, allow_pickle=True)
                            except:
                                print("halting aggregation")
                                stop_aggregation = True

                        if not stop_aggregation:
                            print("Device {} at episode {}".format(device_index, episode_count))
                            model.set_weights(model_target_global.tolist())
                else:
                    print("Warm up")
                    train_start = False

                # model.save(checkpointpath1, include_optimizer=True, save_format='h5')
                np.savez(outfile, frame_count=frame_count, episode_reward_history=episode_reward_history,
                                    running_reward=running_reward, episode_count=episode_count, epsilon=epsilon, training_end=training_end)
                np.save(outfile_models, model_weights)
                del model_weights

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model.save(checkpointpath1, include_optimizer=True, save_format='h5')
                model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
                # np.savez_compressed(outfile, frame_count=frame_count, episode_reward_history=episode_reward_history,
                # running_reward=running_reward, episode_count=episode_count, epsilon=epsilon, training_end=training_end)
                stop_aggregation = False

                model_target.set_weights(model.get_weights())

                # model_loaded = models.load_model('model.h5')
                # model_loaded._make_predict_function() # unclear
                # Log details



            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                # print("initializing")
                break

        # validation tool for device 'device_index'
        trajectory = np.zeros(number_positions, dtype=int)
        [obs, reward, done] = robot_trajectory_validation.initialize() # initialize in position 0 (entrance)
        state = preprocess_observation(np.squeeze(obs))
        episode_reward = reward
        tr_count = 0
        # print(training_signal)
        for timestep_v in range(1, number_positions): # validate overall the full position set (number of positions)
            trajectory[tr_count] = robot_trajectory_validation.getPosition()
            tr_count += 1
            # wait for epsilon_random_frames before validating
            if  epsilon_min_validation > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(n_outputs)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
            dev = 0
            [obs, reward, done] = robot_trajectory_validation.implement(action, dev)
            state_next = preprocess_observation(np.squeeze(obs))
            state_next = np.array(state_next)
            episode_reward += reward
            state = state_next
            if done:
                print("found an exit with reward {}".format(reward))
                break
        trajectory[tr_count] = robot_trajectory_validation.getPosition()

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > max_episodes:  # check memory
            del episode_reward_history[:1]
        # mean reward for last 10 episodes (change depending on application)
        running_reward = np.mean(episode_reward_history[-10:])

        episode_count += 1

        if running_reward > target_reward:  # Condition to consider the task solved
            print("Solved for device {} at episode {} with running reward {:.2f} !".format(device_index, episode_count, running_reward))
            print(trajectory)
            print("Reward {:.2f} for trajectory".format(episode_reward))

            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            # model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, episode_reward_history=episode_reward_history,
                                running_reward=running_reward, episode_count=episode_count, epsilon=epsilon,
                                training_end=training_end)
            np.save(outfile_models, model_weights)
            dict_1 = {"episode_reward_history": episode_reward_history}
            if federated:
                sio.savemat(n_file_cfa, dict_1)
                model.save(n_file_cfa_h5, include_optimizer=True, save_format='h5')
            elif target_server:
                sio.savemat(n_file_fa, dict_1)
                sio.savemat("FA_device_{}.mat".format(device_index), dict_1)
                model.save(n_file_fa_h5, include_optimizer=True, save_format='h5')
            elif args.centralized == 1:
                sio.savemat(n_file_cl, dict_1)
                model.save(n_file_cl_h5, include_optimizer=True, save_format='h5')
            else:
                sio.savemat(n_file_isolated, dict_1)
                model.save(n_file_isolated_h5, include_optimizer=True, save_format='h5')
            break

        if episode_count > max_episodes:  # stop simulation
            print("Unsolved for device {} at episode {}!".format(device_index, episode_count))
            training_end = True
            model_weights = np.asarray(model.get_weights())
            model.save(checkpointpath1, include_optimizer=True, save_format='h5')
            # model_target.save(checkpointpath2, include_optimizer=True, save_format='h5')
            np.savez(outfile, frame_count=frame_count, episode_reward_history=episode_reward_history,
                                running_reward=running_reward, episode_count=episode_count, epsilon=epsilon,
                                training_end=training_end)
            np.save(outfile_models, model_weights)
            dict_1 = {"episode_reward_history": episode_reward_history}
            if federated:
                sio.savemat(n_file_cfa, dict_1)
                model.save(n_file_cfa_h5, include_optimizer=True, save_format='h5')
            elif target_server:
                sio.savemat(n_file_fa, dict_1)
                sio.savemat("FA_device_{}.mat".format(device_index), dict_1)
                model.save(n_file_fa_h5, include_optimizer=True, save_format='h5')
            elif args.centralized == 1:
                sio.savemat(n_file_cl, dict_1)
                model.save(n_file_cl_h5, include_optimizer=True, save_format='h5')
            else:
                sio.savemat(n_file_isolated, dict_1)
                model.save(n_file_isolated_h5, include_optimizer=True, save_format='h5')
            break


if __name__ == "__main__":
    
    # DELETE TEMPORARY CACHE FILES
    # fileList = glob.glob('*.mat', recursive=False)
    # print(fileList)
    # for filePath in fileList:
    #     try:
    #         os.remove(filePath)
    #     except OSError:
    #         print("Error while deleting file")
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

    number_positions_devices = int(number_positions/devices) # max explorable positions per device

    print("Explored positions per device {}".format(number_positions_devices))

    # main loop for multiprocessing
    t = []

    ############# enable federation #######################
    # federated = False
    if args.consensus == 1:
        federated = True
        target_server = False
    else:
        federated = False
    ########################################################

    ##################### enable target server ##############
    # target_server = False
    target_index = devices
    if args.PS == 1:
        target_server = True
        federated = False
    else:
        target_server = False
    #########################################################

    if args.isolated == 1:
        federated = False
        target_server = False

    if args.centralized == 0:
        for ii in range(devices):
            # position start
            initialization = ii*int(args.true_pos/devices) # device 0 starts at position 0, device 1 starts at 1*true_pos/number_of_devices etc...
            print("Device {} starting at position {}".format(ii, initialization))
            t.append(multiprocessing.Process(target=processData, args=(ii, number_positions_devices, federated, target_server, initialization, update_after_actions)))
            t[ii].start()

        # last process is for the target server
        if target_server:
            print("Target server starting")
            t.append(multiprocessing.Process(target=processTargetServer, args=(devices, federated)))
            t[devices].start()
    else:
        update_after_actions = update_after_actions * devices
        processData(0, number_positions_devices, False, False, np.arange(devices), update_after_actions)

    exit(0)

    # Use the Baseline Atari environment because of Deepmind helper functions
    # env = make_atari("BreakoutNoFrameskip-v4")
    # replay_memory_size = 100000
    # replay_memory = ReplayMemory(replay_memory_size)

    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    # env = wrap_deepmind(env, frame_stack=True, scale=True)
    # env.seed(seed)

    # def sample_memories(batch_size):
    #     cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    #     for memory in replay_memory.sample(batch_size):
    #         for col, value in zip(cols, memory):
    #             col.append(value)
    #     cols = [np.array(col) for col in cols]
    #     return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
