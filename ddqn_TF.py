import base64
from IPython.display import HTML
from soko_pap import *

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from tqdm.notebook import tqdm
from collections import defaultdict

import tensorflow as tf


# Define the function to decay the learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=100000,
    decay_rate=0.75,
    staircase=True)


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
        return -5.0

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

class SOK_Agent:
    def __init__(self):
        # Construct DQN models
        self.state_size = (112, 112, 1)
        self.action_size = 8
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.batch_size = 8

        # Replay buffers
        self.replay_buffer = deque(maxlen=10000)
        self.prioritized_replay_buffer = deque(maxlen=5000)

        # Hyperparameters
        self.gamma = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        self.replay_rate = 10
        self.update_beta = 0.9999

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
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
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


def process_frame(frame):
    f = frame.mean(axis=2)
    f = f / 255
    return np.expand_dims(f, axis=0)


def test_agent(e, stochastic=False):
    current_epsilon = agent.epsilon
    agent.epsilon = 0.0
    num_solved = 0
    solved_in_steps = defaultdict(int)
    t_solved = []
    t_unsolved = []

    for t in tqdm(range(100)):
        random.seed(t)
        sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
        sok.set_maxsteps(20)
        steps = 0

        state = sok.get_image('rgb_array')
        done = False
        while not done:
            steps += 1
            action = agent.act(process_frame(state), stochastic)
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

    if len(t_solved) > 0:
        print("Solved: ", t_solved)
    if len(t_unsolved) > 0:
        print("Unsolved: ", t_unsolved)

    # if solved_in_steps isn't empty - sort it by keys
    if solved_in_steps:
        solved_in_steps = dict(sorted(solved_in_steps.items()))

    agent.epsilon = current_epsilon
    print("*" * 30)
    print("Stochastic" if stochastic else "Deterministic")
    print("*" * 30)
    print("Solved: %d" % num_solved)
    print("=" * 30)
    print(solved_in_steps)
    print("*" * 30)

    agent.epsilon = current_epsilon
    print("Episode %d Solved: %d" % (e + 1, num_solved))

    return num_solved


max_episodes = 50000
max_steps = 20
count_epochs = 0


def init_sok(e):
    random.seed(e + 100)
    sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
    sok.set_maxsteps(max_steps)
    return sok

agent = SOK_Agent()

running_puzzles = 0
running_solved = 0
best_solved = 0

for e in range(max_episodes):
    # print episode number each 10 episodes
    if e % 200 == 0:
        print("Episode %d" % (e + 1))
    sok = init_sok(e)
    random.seed(e)
    running_puzzles += 1

    state = process_frame(sok.get_image('rgb_array'))
    room_state = sok.room_state.copy()
    distances = get_distances(room_state)

    for step in range(sok.max_steps):
        action = agent.act(state, stochastic=True)
        if action < 4:
            next_state, reward, done, _ = sok.step(action + 1)
        else:
            next_state, reward, done, _ = sok.step(action + 5)

        next_state = process_frame(next_state)
        next_room_state = sok.room_state

        if not done:
            reward += box2target_change_reward(room_state, next_room_state, distances)

        agent.remember(state, action, reward, next_state, done)

        state = next_state.copy()
        room_state = next_room_state.copy()

        if (step + 1) % agent.replay_rate == 0:
            agent.replay()
            count_epochs += 10

        if done:
            if 3 in sok.room_state:
                agent.copy_to_prioritized_buffer(step + 1)
                running_solved += 1

            if (e + 1) % 20 == 0 and e > 0:
                print(f"{running_solved} | {running_puzzles}")
                # print count_epochs
                print(f"count_epochs: {count_epochs}")

                if (e + 1) % 100 == 0:
                    running_puzzles = 0
                    running_solved = 0

            break

    if (e + 1) % 100 == 0 and e > 0:
        num_s = test_agent(e, stochastic=False)
        if num_s > best_solved:
            best_solved = num_s
            # save the model in models folder
            print(f"Solved {num_s} puzzles in episode number {e + 1}\n. Saving model...")
            agent.save(f"models/model_{num_s}_{e + 1}.h5")



