# Sokoban-gym
Solution for Sokoban using RL

## Installation
The Environment: https://github.com/mpSchrader/gym-sokoban
Our code is based on Keras on TensorFlow. We also have torch-based version, but our best results achieved using Keras-based code.

### Ex1
See at notebooks/Ex1_final.ipynb
Agent will converge after around 120 episodes

### Ex2
See at notebooks/Ex2_final.ipynb

For train agent from scratch one can use Ex2.py, as follows:
```
Python3 Ex2.py
```

There are many hyper parameters and more arguments, for example:
```
Python3 Ex2.py --learning_rate 0.001 --replay_buffer_size 1000 --success_before_train 20
```
Please find the entire list in the python file itself.

For Inference our best model (solve 96/100) run:
```
Python3 Ex2.py --inference --load_model Ex2_weights.h5
```
Videos for all 100 tests are stored in test_videos

### Ex3
See at notebooks/Ex3_final.ipynb
Our best agent reaches XXX
