# Sokoban-gym
Solution for Sokoban using RL

## Installation
The Environment: https://github.com/mpSchrader/gym-sokoban
Our code is based on Keras on TensorFlow. We also have torch-based version, but our best results achieved using Keras-based code.

All our notebooks are stored under [Notebook](https://github.com/AvrahamRaviv/Sokoban-gym/tree/main/Notebooks) folder.
### Ex1
See at [Ex1 Notebook](https://colab.research.google.com/drive/1eIVgC8H8Ftmv0AliiLAv0U_GQgRugi28)

Agent will converge after around 120 episodes.

### Ex2
See at [Ex2 Notebook](https://colab.research.google.com/drive/1lqsRouX3752jFCJv9M5hb25z7row8kEw?usp=sharing) (at the first part of the notebook).

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
Python3 Ex2.py --inference --load_model models/Q2_end.h5
```
Videos for all 100 tests are stored in test_videos/Ex2.

### Ex3
See at [Ex3 Notebook](https://colab.research.google.com/drive/1lqsRouX3752jFCJv9M5hb25z7row8kEw?usp=sharing) (at the second part of the notebook).

For Inference our best model (solve 73/100) run:
```
Python3 Ex3.py --inference --load_model models/Q3_end.h5
```
Videos for all 100 tests are stored in test_videos/Ex3.
