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
import cv2


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_rate', type=int, default=10, help='How many steps between each replay')
    parser.add_argument('--replay_times', type=int, default=10, help='How many times to replay')
    parser.add_argument('--max_episodes', type=int, default=50000, help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum number of steps per episode')
    parser.add_argument('--epsilon', type=float, default=1, help='Epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.2, help='Epsilon minimum value')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--action_size', type=int, default=8, help='Action size')
    parser.add_argument('--replay_buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--prioritized_replay_buffer_size', type=int, default=1500,
                        help='Prioritized replay buffer size')
    parser.add_argument('--prioritized_replay_batch', type=int, default=20, help='Prioritized replay batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--decay_steps', type=int, default=2000, help='update_lr_rate')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay_lr')
    parser.add_argument('--update_beta', type=float, default=0.999, help='Update beta')
    parser.add_argument('--positive_reward', type=float, default=10, help='Positive reward')
    parser.add_argument('--NonMovePenalty', type=float, default=-1, help='NonMovePenalty')
    parser.add_argument('--CR', type=float, default=5, help='Change Reward')
    parser.add_argument('--aug_size', type=int, default=4, help='Augmentation size')
    parser.add_argument('--success_before_train', type=int, default=0, help='Success before train')
    parser.add_argument('--load_model', type=str, default='', help='Load model')
    parser.add_argument('--DNN_type', type=str, default='M', help='DNN size')
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


# # Utils

# In[8]:


def get_distances_for_target(room_state, target):
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


def get_maze_info(room_state):
    targets = []
    for i in range(room_state.shape[0]):
        for j in range(room_state.shape[1]):
            if room_state[i][j] == 2:
                targets.append((i, j))

    distances0 = get_distances_for_target(room_state, targets[0])
    distances1 = get_distances_for_target(room_state, targets[1])
    common_distances = np.minimum(distances0, distances1)

    maze_info = {}
    maze_info['target0'] = targets[0]
    maze_info['target1'] = targets[1]
    maze_info['distances0'] = distances0
    maze_info['distances1'] = distances1
    maze_info['coomon_distances'] = common_distances
    return maze_info


def calc_distances(room_state, distances):
    boxes = []
    for i in range(room_state.shape[0]):
        for j in range(room_state.shape[1]):
            if room_state[i][j] == 4:
                boxes.append((i, j))
    if len(boxes) == 2:
        return distances[boxes[0][0]][boxes[0][1]] + distances[boxes[1][0]][boxes[1][1]]

    return distances[boxes[0][0]][boxes[0][1]]


def box2target_change_reward(room_state, next_room_state, maze_info):
    if np.array_equal(room_state, next_room_state):
        return -10.0

    target0 = maze_info['target0']
    target1 = maze_info['target1']
    distances0 = maze_info['distances0']
    distances1 = maze_info['distances1']
    common_distances = maze_info['coomon_distances']

    relevant_distances = common_distances

    if room_state[target0[0]][target0[1]] == 3:
        relevant_distances = distances1
    elif room_state[target1[0]][target1[1]] == 3:
        relevant_distances = distances0

    change_reward = 0.0
    t2b = calc_distances(room_state, relevant_distances)
    n_t2b = calc_distances(next_room_state, relevant_distances)
    if n_t2b < t2b:
        change_reward += 10.0
    elif n_t2b > t2b:
        change_reward -= 10.0

    return change_reward


class SOK_Agent:
    def __init__(self, sok_args):
        # Construct DQN models
        self.state_size = (112, 112, 1)
        self.action_size = sok_args.action_size
        self.learning_rate = sok_args.learning_rate
        self.decay_rate = sok_args.decay_rate
        self.decay_steps = sok_args.decay_steps
        self.DNN_type = sok_args.DNN_type
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.batch_size = 8
        self.inference = sok_args.inference

        # Replay buffers
        self.replay_buffer = deque(maxlen=10000)
        self.prioritized_replay_buffer = deque(maxlen=1000)

        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = sok_args.epsilon
        self.epsilon_min = sok_args.epsilon_min
        self.epsilon_decay = 0.9995
        self.replay_rate = 10
        self.update_beta = 0.999
        self.success_before_train = sok_args.success_before_train
        self.exp_header = sok_args.exp_header

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
        if self.DNN_type == 'M':
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        else:
            model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        if self.DNN_type == 'L':
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        lr_schedule = ExponentialDecay(self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                       staircase=False)
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
        targets = np.zeros((self.batch_size * 4, self.action_size))

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

        for t in range(100):
            sok = init_sok(t)
            steps = 0
            images = []
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

            if sok.boxes_on_target == 2:
                images.append(sok.get_image('rgb_array'))
                num_solved += 1
                solved_in_steps[steps] += 1
                t_solved.append(t)
            else:
                t_unsolved.append(t)

            if self.inference:
                height, width, layers = images[0].shape
                size = (width, height)
                out = cv2.VideoWriter(f'test_videos/Ex3_2/test_{t}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
                for i in range(len(images)):
                    out.write(images[i])
                out.release()

        self.epsilon = current_epsilon
        # print(f"Episode {e + 1} Epsilon {self.epsilon} Learning Rate {np.round(self.model.optimizer.lr.numpy(), 6)} Solved: {num_solved}")
        print(f"Episode {e + 1} Epsilon {self.epsilon}, Solved: {num_solved}")

        if num_solved > cur_record:
            self.save(f"models/Q2_03A_{self.exp_header}_{num_solved}_{e}.h5")
            cur_record = num_solved

        if len(t_solved) > 0:
            print("Solved: ", t_solved)
        if len(t_unsolved) > 0:
            print("Unsolved: ", t_unsolved)

        # if solved_in_steps isn't empty - sort it by keys
        if solved_in_steps:
            solved_in_steps = dict(sorted(solved_in_steps.items()))

        print("*" * 30)
        print("Solved: %d" % num_solved)
        print("=" * 30)
        print(solved_in_steps)
        print("*" * 30)

        return num_solved, cur_record


def process_frame(frame):
    f = frame.mean(axis=2)
    f = f / 255
    return np.expand_dims(f, axis=0)


max_episodes = 50000
max_steps = 30


def init_sok(r):
    random.seed(r)
    sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=2)
    sok.set_maxsteps(max_steps)
    return sok

def inner_main(args):
    agent = SOK_Agent(args)
    running_puzzles = 0
    running_solved = 0
    num_success = 0
    solved_in_train = []
    solved_in_test = []
    test_record = -1
    agent.exp_header = args.exp_header
    print(args)

    if args.load_model != '':
        agent.model.load_weights(f"models/{args.load_model}.h5")
        agent.target_model.load_weights(f"models/{args.load_model}.h5")
        agent.epsilon = agent.epsilon_min
        agent.success_before_train = 65

    for e in range(max_episodes):
        print(f"episode {e}")
        if e % 100 == 0:
            solved_tests, test_record = agent.test_agent(e, test_record, stochastic=False)
            solved_in_test.append(solved_tests)
            if agent.inference:
                exit(0)
        sok = init_sok(e + 100)
        random.seed(e)
        running_puzzles += 1

        state = process_frame(sok.get_image('rgb_array'))
        room_state = sok.room_state.copy()
        maze_info = get_maze_info(room_state)

        for step in range(sok.max_steps):
            action = agent.act(state, stochastic=True)
            if action < 4:
                next_state, reward, done, _ = sok.step(action + 1)
            else:
                next_state, reward, done, _ = sok.step(action + 5)

            next_state = process_frame(next_state)
            next_room_state = sok.room_state

            if not done:
                reward += box2target_change_reward(room_state, next_room_state, maze_info)

            agent.remember(state, action, reward, next_state, done)

            state = next_state.copy()
            room_state = next_room_state.copy()

            if (step + 1) % agent.replay_rate == 0 and num_success > agent.success_before_train:
                agent.replay()

            if done:
                if sok.boxes_on_target == 2:
                    agent.copy_to_prioritized_buffer(step + 1)
                    running_solved += 1
                    num_success += 1

                if (e + 1) % 10 == 0 and e > 0:
                    print(f"{running_solved} | {running_puzzles}")

                    if (e + 1) % 100 == 0:
                        solved_in_train.append(running_solved)
                        running_puzzles = 0
                        running_solved = 0

                break


if __name__ == '__main__':
    # call inner_main with arguments
    t0 = time.time()
    args = parse_arguments()
    inner_main(args)
    print("Finished in %.4f seconds\n" % (time.time() - t0))
