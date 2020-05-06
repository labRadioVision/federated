Available also on 
https://test.pypi.org/project/consensus-stefano/0.3/

# Usage example for federated_sample_XXX_YYY.py.

- XXX refers to the ML model. Options: CNN, 2NN

- YYY refers to the consensus-based federated learning method. Options: CFA, CFA-GE

Note: the code is written for tensorflow 1.13.1. To use the code with tensorflow 2.1.0 installed, please use the following workaround:

Replace
import tensorflow as tf

with
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # tf 2

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


# PYTHON PACKAGE (SEE ALSO THE CODE SCRIPTS FOR FURTHER DETAILS)

# CFA

To initialize CFA use constructor:
    consensus_p = CFA_process(federated, tot_devices, device_id, neighbors_number)
    
To apply/update Federated weights use:
    consensus_p.getFederatedWeight( ... )		

To enable/disable consensus (dynamically)
    consensus_p.disable_consensus( ... True/False ... )

# CFA-GE

To initialize CFA-GE:
    consensus_p = CFA_ge_process(federated, tot_devices, iii, neighbors_number, args.ro)
    
Set ML model parameters (CNN model):
    consensus_p.setCNNparameters(filter, number, pooling, stride, multip, classes, input_data)
    
Alternatively 2NN model can be used:
    consensus_p.set2NNparameters(intermediate_nodes, classes, input_data)
    
To apply/update Federated weights use (4 stage CFA-GE):
    consensus_p.getFederatedWeight_gradients( ... )		
    
To apply/update Federated weights use (2 stage CFA-GE):
    consensus_p.getFederatedWeight_gradients_fast( ... )	
