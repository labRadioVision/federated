@echo off
cd "D:\federated_git\tensorflow2_implementations\CIFAR10_dataset"
"D:\anaconda\envs\tensorflow25\python.exe" "D:\federated_git\tensorflow2_implementations\CIFAR10_dataset\federated_learning_keras_consensus_FL_threads_CIFAR100.py" -PS 0 -consensus 1 -K 120 -target 0.5 -Ka_consensus 40 -random_data_distribution 1 -run 0
pause