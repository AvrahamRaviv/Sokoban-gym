import numpy as np
from collections import deque
import torch


def get_distances(room_state):
    target = None

    for i in range(room_state.shape[0]):
        for j in range(room_state.shape[1]):
            if room_state[i][j] == 2:
                target = (i, j)

    distances = np.zeros(shape=room_state.shape)
    visited_cells = set()
    cell_queue = deque()

    visited_cells.add(target)
    cell_queue.appendleft(target)

    while len(cell_queue) != 0:
        cell = cell_queue.pop()
        distance = distances[cell[0]][cell[1]]
        for x, y in ((1, 0), (-1, -0), (0, 1), (0, -1)):
            next_cell_x, next_cell_y = cell[0] + x, cell[1] + y
            if room_state[next_cell_x][next_cell_y] != 0 and not (next_cell_x, next_cell_y) in visited_cells:
                distances[next_cell_x][next_cell_y] = distance + 1
                visited_cells.add((next_cell_x, next_cell_y))
                cell_queue.appendleft((next_cell_x, next_cell_y))

    return distances


def calc_distances(room_state, distances):
    box = None
    mover = None
    for i in range(room_state.shape[0]):
        for j in range(room_state.shape[1]):
            if room_state[i][j] == 4:
                box = (i, j)

            if room_state[i][j] == 5:
                mover = (i, j)

    return mover, box, distances[box[0]][box[1]]


def box2target_change_reward(room_state, next_room_state, distances):
    if np.array_equal(room_state, next_room_state):
        return -1.0

    mover, box, t2b = calc_distances(room_state, distances)
    n_mover, n_box, n_t2b = calc_distances(next_room_state, distances)

    change_reward = 0.0
    if n_t2b < t2b:
        change_reward += 5.0
    elif n_t2b > t2b:
        change_reward -= 5.0

    m2b = np.sqrt((mover[0] - box[0]) ** 2 + (mover[1] - box[1]) ** 2)
    n_m2b = np.sqrt((n_mover[0] - n_box[0]) ** 2 + (n_mover[1] - n_box[1]) ** 2)

    if n_m2b < m2b and m2b >= 2:
        change_reward += 1.0
    elif n_m2b > m2b and n_m2b >= 2:
        change_reward -= 1.0

    return change_reward


def process_frame(frame):
    f = frame.mean(axis=2)
    f = f / 255
    return np.expand_dims(f, axis=0)


action_rotation_map = {
    0: 2,
    1: 3,
    2: 1,
    3: 0,
    4: 6,
    5: 7,
    6: 5,
    7: 4
    }


def calc_rot_action(org_action, aug):
    # given an action and an augmentation, return the action that is equivalent to the augmentation
    # action is number from range 0-action_size, and aug is number from range 1-3
    # for single augmentation step, the action is change by using action_rotation_map
    action = org_action
    for i in range(aug):
        action = action_rotation_map[action]
    return action