# For further details please check the following articles: 

https://arxiv.org/pdf/1912.13163.pdf or https://ieeexplore.ieee.org/abstract/document/8950073/

https://ieee-dataport.org/open-access/federated-learning-mmwave-mimo-radar-dataset-testing

https://ieeexplore.ieee.org/abstract/document/9054055/

# Usage example for federated_sample_XXX_YYY.py.

- XXX refers to the ML model. Options: CNN, 2NN

- YYY refers to the consensus-based federated learning method. Options: CFA, CFA-GE

Note: the code is written for tensorflow 1.13.1. To use the code with tensorflow 2.1.0 installed, please use the following workaround:

Replace:

import tensorflow as tf

with:

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

Please check https://www.tensorflow.org/guide/migrate

Run 

python federated_sample_XXX_YYY.py -h 

for help 

# CFA-GE: example Python script
federated_sample_XXX_CFA-GE.py [-h] [-l1 L1] [-l2 L2] [-mu MU]
                                [-eps EPS] [-K K] [-N N] [-T T]
                                [-ro RO]

optional arguments:

  -h, --help  show this help message and exit
  
  -l1 L1      sets the learning rate (gradient exchange) for convolutional
              layer
              
  -l2 L2      sets the learning rate (gradient exchange) for FC layer
  
  -mu MU      sets the learning rate for local SGD
  
  -eps EPS    sets the mixing parameters for model averaging (CFA)
  
  -K K        sets the number of network devices
  
  -N N        sets the number of neighbors per device
  
  -T T        sets the number of training epochs
  
  -ro RO      sets the hyperparameter for MEWMA

# CFA: example Python script
federated_sample_XXX_CFA.py [-h] [-mu MU]
                                [-eps EPS] [-K K] [-N N] [-T T]

optional arguments:

  -h, --help  show this help message and exit
  
  -mu MU      sets the learning rate for local SGD
  
  -eps EPS    sets the mixing parameters for model averaging (CFA)
  
  -K K        sets the number of network devices
  
  -N N        sets the number of neighbors per device
  
  -T T        sets the number of training epochs

# Alternating federated averaging and consensus: example Python script
federated_sample_CNN_CFA_FA.py [-h] [-mu MU] [-eps EPS] [-eps2 EPS2]
                                      [-K K] [-N N] [-T T] [-S S] [-Ser SER]
                                      [-Con CON]

optional arguments:
  -h, --help  show this help message and exit
  
  -mu MU      sets the learning rate for local SGD
  
  -eps EPS    sets the mixing parameters for model averaging (CFA)
  
  -eps2 EPS2  sets the updated parameters for server-side federated learning
  
  -K K        sets the number of network devices
  
  -N N        sets the number of neighbors per device
  
  -T T        sets the number of training epochs
  
  -S S        sets the frequency of server-side computation
  
  -Ser SER    sets the number of epochs for server-side computation
  
  -Con CON    sets the number of epochs for consensus operations


# Example 1 

python federated_sample_CNN_CFA-GE.py -l1 0.025 -l2 0.02 -K 40 -N 2 -T 40 -ro 0.99

Use convolutional layers followed by a FC layer (CNN model, see paper) and CFA-GE federated learning algorithm. 

Sets gradient learning rate for hidden layer to 0.025, for output layer to 0.02, K=40 devices, N=2 neighbors per device, MEWMA parameter 0.99 (see paper), number of training epochs to T = 40


# Example 2

python federated_sample_2NN_CFA.py - K 30 -N 2

Use FC layers (2NN model, see paper) and CFA federated learning algorithm. Sets K=30 devices, N=2 neighbors per device, number of training epoch is set to default T = 120


# Decentralized Federated Learning simulator: FL_CFA_CNN_tf2.py (UPDATE 13/05/2020)

New version of the FL simulator: supports tensorflow 2. 

CFA and gossip implemented:

usage: FL_CFA_CNN_tf2.py [-h] [-mu MU] [-eps EPS] [-K K] [-N N] [-T T]
                         [-samp SAMP] [-input_data INPUT_DATA] [-rand RAND]
                         [-consensus_mode CONSENSUS_MODE] [-graph GRAPH]
                         [-compression COMPRESSION]


optional arguments:

  -h, --help            show this help message and exit
  
  -mu MU                sets the learning rate for local SGD
  
  -eps EPS              sets the mixing parameters for model averaging (CFA)
  
  -K K                  sets the number of network devices
  
  -N N                  sets the max. number of neighbors per device per round
  
  -T T                  sets the number of training epochs
  
  -samp SAMP            sets the number samples per device
  
  -input_data INPUT_DATA
                        sets the path to the federated dataset
                        
  -rand RAND            sets static or random choice of the N neighbors on
                        every new round (0 static, 1 random)
                        
  -consensus_mode CONSENSUS_MODE
                        0: combine one neighbor at a time and run sgd AFTER
                        every new combination; 1 (faster): combine all
                        neighbors on a single stage, run one sgd after this
                        combination
                        
  -graph GRAPH          sets the input graph: 0 for default graph, >0 uses the
                        input graph in vGraph.mat, and choose one graph from
                        the available adjacency matrices
                        
  -compression COMPRESSION
                        sets the compression factor for communication: 0 no
                        compression, 1, sparse, 2 sparse + dpcm, 3 sparse
                        (high compression factor), 4 sparse + dpcm (high
                        compression factor)

