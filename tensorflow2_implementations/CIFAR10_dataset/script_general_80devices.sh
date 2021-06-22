#!/usr/bin/sudo /bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/stefano/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/stefano/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/stefano/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/stefano/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate reinforcement

#python federated_learning_keras_PS_CIFAR100.py -PS 0 -consensus 0 -K 60 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 0 -consensus 0 -K 30 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 0 -consensus 0 -K 80 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 40 -Ka 30 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 60 -Ka 40 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 30 -Ka 20 -run 0 -target 0.5
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 40 -Ka 30 -run 1 -target 1
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 60 -Ka 40 -run 1 -target 1
#python federated_learning_keras_PS_CIFAR100.py -PS 1 -consensus 0 -K 30 -Ka 20 -run 1 -target 1
#python federated_learning_keras_consensus_FL_CIFAR100.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -run 0 -target 0.5
#python federated_learning_keras_consensus_FL_CIFAR100.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -run 0 -target 0.5
#python federated_learning_keras_consensus_FL_CIFAR100.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -run 0 -target 0.5
#python federated_learning_keras_consensus_FL_CIFAR100.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -run 1 -target 1
#python federated_learning_keras_consensus_FL_CIFAR100.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -run 1 -target 1
#python federated_learning_keras_consensus_FL_threads_CIFAR100.py -PS 0 -consensus 1 -K 100 -N 1 -Ka_consensus 50 -run 0 -target 0.5 -noniid_assignment 1
#python federated_learning_keras_consensus_FL_threads_CIFAR100.py -PS 0 -consensus 1 -K 60 -Ka_consensus 40 -run 0 -target 0.5 -noniid_assignment 1
#python federated_learning_keras_consensus_FL_threads_CIFAR100.py -PS 0 -consensus 1 -K 30 -Ka_consensus 20 -run 0 -target 0.5 -noniid_assignment 1
python federated_learning_keras_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 100 -Ka 50 -run 0 -target 0.5 -noniid_assignment 1
python federated_learning_keras_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 60	-Ka 40 -run 0 -target 0.5 -noniid_assignment 1
python federated_learning_keras_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 30	-Ka 20 -run 0 -target 0.5 -noniid_assignment 1
#python federated_learning_keras_low_power_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 100 -Ka 60 -run 0 -target 0.5 -random_data_distribution 0
#python federated_learning_keras_low_power_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 80 -Ka 60 -run 0 -target 0.5 -random_data_distribution 0
#python federated_learning_keras_low_power_PS_threads_CIFAR100.py -PS 1 -consensus 0 -K 60 -Ka 60 -run 0 -target 0.5 -random_data_distribution 0
#python federated_learning_keras_low_power_PS_CIFAR100.py -PS 1 -consensus 0 -K 60 -Ka 40 -run 1 -target 1
#python federated_learning_keras_low_power_PS_CIFAR100.py -PS 1 -consensus 0 -K 30 -Ka 20 -run 1 -target 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#for i in {0..4}
#do
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 5 -N 1 -Ka 5 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 5 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 10 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 10 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 15 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 15 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 20 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 20 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 30 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 30 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches $batch -random_data_distribution 0 -noniid_assignment $noniid -run $i
#done
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches $batch 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 1
#
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches $batch -random_data_distribution 0 -noniid_assignment 1 -run 2
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
##python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 3
#
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 5 -N 1 -Ka_consensus 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 10 -N 1 -Ka_consensus 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 20 -N 1 -Ka_consensus 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 30 -N 1 -Ka_consensus 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 40 -N 1 -Ka_consensus 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
#python federated_learning_keras_v2.py -PS 0 -consensus 1 -K 60 -N 1 -Ka_consensus 40 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4

#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 5 -N 1 -Ka 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 0



#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 5 -N 1 -Ka 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 1
#
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 5 -N 1 -Ka 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 5 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 10 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 15 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 20 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 30 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 2
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 5 -N 1 -Ka 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 5 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 10 -N 1 -Ka 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 10 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 20 -N 1 -Ka 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 15 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 30 -N 1 -Ka 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 20 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 40 -N 1 -Ka 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 30 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches 1 -random_data_distribution 0 -noniid_assignment 1 -run 1
#python federated_learning_keras_v2.py -PS 1 -consensus 0 -K 60 -N 1 -Ka 40 -batches 5 -random_data_distribution 0 -noniid_assignment 1 -run 4
