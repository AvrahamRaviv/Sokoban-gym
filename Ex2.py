import time
import numpy as np
import base64
from IPython.display import HTML
from soko_pap import *
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from collections import defaultdict
import sys
import argparse
import torch

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_rate', type=int, default=10, help='How many steps between each replay')
    parser.add_argument('--replay_times', type=int, default=10, help='How many times to replay')
    parser.add_argument('--max_episodes', type=int, default=50000, help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum number of steps per episode')
    parser.add_argument('--epsilon', type=float, default=1, help='Epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.1, help='Epsilon minimum value')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--action_size', type=int, default=8, help='Action size')
    parser.add_argument('--replay_buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--prioritized_replay_buffer_size', type=int, default=1500,
                        help='Prioritized replay buffer size')
    parser.add_argument('--prioritized_replay_batch', type=int, default=20, help='Prioritized replay batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--decay_steps', type=int, default=5000, help='update_lr_rate')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay_lr')
    parser.add_argument('--update_beta', type=float, default=0.999, help='Update beta')
    parser.add_argument('--positive_reward', type=float, default=10, help='Positive reward')
    parser.add_argument('--NonMovePenalty', type=float, default=-1, help='NonMovePenalty')
    parser.add_argument('--CR', type=float, default=5, help='Change Reward')
    parser.add_argument('--aug_size', type=int, default=4, help='Augmentation size')
    parser.add_argument('--success_before_train', type=int, default=0, help='Success before train')
    parser.add_argument('--load_model', type=str, default='', help='Load model')
    parser.add_argument('--test_rate', type=int, default=100, help='Test rate')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune')
    parser.add_argument('--inference', action='store_true', help='Inference')
    parser.add_argument('--exp_header', type=str, default=None, help='Experiment header')
    return parser.parse_args()


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return HTML(tag)


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


def box2target_change_reward(room_state, next_room_state, distances, NMP=-1, CR=5):
    if np.array_equal(room_state, next_room_state):
        return NMP

    mover, box, t2b = calc_distances(room_state, distances)
    n_mover, n_box, n_t2b = calc_distances(next_room_state, distances)

    change_reward = 0.0
    if n_t2b < t2b:
        change_reward += CR
    elif n_t2b > t2b:
        change_reward -= CR

    m2b = np.sqrt((mover[0] - box[0]) ** 2 + (mover[1] - box[1]) ** 2)
    n_m2b = np.sqrt((n_mover[0] - n_box[0]) ** 2 + (n_mover[1] - n_box[1]) ** 2)

    if n_m2b < m2b and m2b >= 2:
        change_reward += CR / 5
    elif n_m2b > m2b and n_m2b >= 2:
        change_reward -= (CR / 5)

    return change_reward


class SOK_Agent:
    def __init__(self, sok_args):
        # Construct DQN models
        self.state_size = (112, 112, 1)
        self.action_size = sok_args.action_size
        self.lr = sok_args.learning_rate
        self.decay_rate = sok_args.decay_rate
        self.decay_steps = sok_args.decay_steps
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.batch_size = sok_args.batch_size
        self.load_model = sok_args.load_model

        # Replay buffers
        self.replay_buffer_size = sok_args.replay_buffer_size
        self.prioritized_replay_buffer_size = sok_args.prioritized_replay_buffer_size
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.prioritized_replay_buffer = deque(maxlen=self.prioritized_replay_buffer_size)

        # Hyperparameters
        self.gamma = sok_args.gamma
        self.epsilon = sok_args.epsilon
        self.epsilon_min = sok_args.epsilon_min
        self.epsilon_decay = sok_args.epsilon_decay
        self.replay_rate = sok_args.replay_rate
        self.update_beta = sok_args.update_beta
        self.exp_header = sok_args.exp_header
        self.NonMovePenalty = sok_args.NonMovePenalty
        self.positive_reward = sok_args.positive_reward
        self.CR = sok_args.CR
        self.max_episodes = sok_args.max_episodes
        self.success_before_train = sok_args.success_before_train
        self.test_rate = sok_args.test_rate
        self.max_steps = sok_args.max_steps
        self.inference = agent_args.inference

        self.action_rotation_map = {
            0: 2,
            1: 3,
            2: 1,
            3: 0,
            4: 6,
            5: 7,
            6: 5,
            7: 4
        }

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (16, 16), strides=(16, 16), input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        lr_schedule = ExponentialDecay(self.lr, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=False)
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append([state, action, reward, next_state, done])

    def copy_to_prioritized_buffer(self, n):
        for i in range(n):
            self.prioritized_replay_buffer.append(self.replay_buffer[-1 - i])

    def act(self, state, stochastic=False):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state, verbose=0)[0]

        if stochastic:
            act_probs = np.exp(act_values) / np.exp(act_values).sum()
            return np.random.choice(np.arange(self.action_size), size=1, p=act_probs)[0]

        return np.argmax(act_values)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        if len(self.prioritized_replay_buffer) < self.batch_size // 2:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size // 2)
            minibatch.extend(random.sample(self.prioritized_replay_buffer, self.batch_size // 2))

        states = np.zeros((self.batch_size * 4, self.state_size[0], self.state_size[1]))
        actions = np.zeros(self.batch_size * 4, dtype=int)
        rewards = np.zeros(self.batch_size * 4)
        next_states = np.zeros((self.batch_size * 4, self.state_size[0], self.state_size[1]))
        statuses = np.zeros(self.batch_size * 4)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            for rot in range(4):
                ind = i * 4 + rot
                if rot != 0:
                    state = np.rot90(state, axes=(1, 2))
                    next_state = np.rot90(next_state, axes=(1, 2))
                    action = self.action_rotation_map.get(action)

                states[ind] = state.copy()
                actions[ind] = action
                rewards[ind] = reward
                next_states[ind] = next_state.copy()
                statuses[ind] = 1 if done else 0

        targets = self.model.predict(states, verbose=0)
        max_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_rewards = self.target_model.predict(next_states, verbose=0)

        ind = 0
        for action, reward, next_reward, max_action, done in zip(actions, rewards, next_rewards, max_actions, statuses):
            if not done:
                reward += self.gamma * next_reward[max_action]
            targets[ind][action] = reward
            ind += 1

        self.model.fit(states, targets, epochs=10, verbose=0)

        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def update_target_model(self):
        model_w = self.model.get_weights()
        target_model_w = self.target_model.get_weights()
        updated_target_model_w = []
        for i in range(len(model_w)):
            updated_target_model_w.append(self.update_beta * target_model_w[i] + (1 - self.update_beta) * model_w[i])
        self.target_model.set_weights(updated_target_model_w)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def test_agent(self, e, cur_record, stochastic=False):
        current_epsilon = self.epsilon
        self.epsilon = 0.0
        num_solved = 0
        solved_in_steps = defaultdict(int)
        t_solved = []
        t_unsolved = []
        images = []

        for t in range(100):
            random.seed(t)
            sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
            sok.set_maxsteps(20)
            steps = 0

            state = sok.get_image('rgb_array')
            done = False
            while not done:
                images.append(sok.get_image('rgb_array'))
                steps += 1
                action = self.act(process_frame(state), stochastic)
                if action < 4:
                    action += 1
                else:
                    action += 5
                state, reward, done, info = sok.step(action)

            if 3 in sok.room_state:
                num_solved += 1
                solved_in_steps[steps] += 1
                t_solved.append(t)
            else:
                t_unsolved.append(t)
                
            if self.inference:
                height, width, layers = images[0].shape
                size = (width, height)
                out = cv2.VideoWriter(f'test_videos/test_{t}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
                for i in range(len(images)):
                    out.write(images[i])
                out.release()

        self.epsilon = current_epsilon
        print(f"Episode {e + 1} Epsilon {self.epsilon} Learning Rate {np.round(self.model.optimizer.lr.numpy(), 6)} Solved: {num_solved}")

        if num_solved > cur_record:
            self.save(f"models/Q2_03A_{num_solved}_{self.exp_header}.h5")
            cur_record = num_solved

        if len(t_solved) > 0:
            print("Solved: ", t_solved)
        if len(t_unsolved) > 0:
            print("Unsolved: ", t_unsolved)

        # if solved_in_steps isn't empty - sort it by keys
        if solved_in_steps:
            solved_in_steps = dict(sorted(solved_in_steps.items()))

        print("*" * 30)
        print("Stochastic" if stochastic else "Deterministic")
        print("*" * 30)
        print("Solved: %d" % num_solved)
        print("=" * 30)
        print(solved_in_steps)
        print("*" * 30)

        return num_solved, cur_record

    def init_sok(self, r):
        random.seed(r + 100)
        sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
        sok.set_maxsteps(self.max_steps)
        return sok


def process_frame(frame):
    f = frame.mean(axis=2)
    f = f / 255
    return np.expand_dims(f, axis=0)


def inner_main(argv):
    running_puzzles = 0
    running_solved = 0
    record = 0
    num_success = 0
    solved_tests = []

    agent = SOK_Agent(argv)
    max_episodes = agent.max_episodes

    if argv.load_model != '':
        agent.model.load_weights(f"models/{agent.load_model}")
        agent.target_model.load_weights(f"models/{agent.load_model}")

    for e in range(max_episodes):
        print("Episode %", e)
        sok = agent.init_sok(e)
        random.seed(e)
        running_puzzles += 1

        state = process_frame(sok.get_image('rgb_array'))
        room_state = sok.room_state.copy()
        distances = get_distances(room_state)

        if e % agent.test_rate == 0:
            num_solved, record = agent.test_agent(e, record, stochastic=False)
            solved_tests.append(num_solved)
            if agent.inference:
                exit(0)

        for step in range(sok.max_steps):
            action = agent.act(state)
            if action < 4:
                next_state, reward, done, _ = sok.step(action + 1)
            else:
                next_state, reward, done, _ = sok.step(action + 5)

            next_state = process_frame(next_state)
            next_room_state = sok.room_state

            if not done:
                reward += box2target_change_reward(room_state, next_room_state, distances, agent.NonMovePenalty, agent.CR)
            else:
                reward = agent.positive_reward

            agent.remember(state, action, reward, next_state, done)

            state = next_state.copy()
            room_state = next_room_state.copy()

            if (step + 1) % agent.replay_rate == 0:
                agent.replay()

            if done:
                if 3 in sok.room_state:
                    agent.copy_to_prioritized_buffer(step + 1)
                    running_solved += 1
                    num_success += 1

                if (e + 1) % 20 == 0 and num_success > agent.success_before_train:
                    print(f"{running_solved} | {running_puzzles}")

                    if (e + 1) % 100 == 0:
                        running_puzzles = 0
                        running_solved = 0

                break


def main(argv):
    t0 = time.time()
    args = parse_arguments()
    inner_main(args)
    print("Finished in %.4f seconds\n" % (time.time() - t0))
