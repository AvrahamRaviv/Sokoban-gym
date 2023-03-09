import matplotlib.pyplot as plt
import torch

from soko_pap import *
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
from utils2 import *
from copy import deepcopy


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_rate', type=int, default=10, help='How many steps between each replay')
    parser.add_argument('--replay_times', type=int, default=10, help='How many times to replay')
    parser.add_argument('--max_episodes', type=int, default=50000, help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum number of steps per episode')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.3, help='Epsilon minimum value')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--action_size', type=int, default=8, help='Action size')
    parser.add_argument('--replay_buffer_size', type=int, default=1500, help='Replay buffer size')
    parser.add_argument('--prioritized_replay_buffer_size', type=int, default=500,
                        help='Prioritized replay buffer size')
    parser.add_argument('--prioritized_replay_batch', type=int, default=20, help='Prioritized replay batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--update_lr_rate', type=int, default=2500, help='update_lr_rate')
    parser.add_argument('--decay_lr', type=float, default=0.25, help='decay_lr')
    parser.add_argument('--update_beta', type=float, default=0.999, help='Update beta')
    parser.add_argument('--positive_reward', type=float, default=10, help='Positive reward')
    parser.add_argument('--NonMovePenalty', type=float, default=-1, help='NonMovePenalty')
    parser.add_argument('--CR', type=float, default=1, help='Change Reward')
    parser.add_argument('--epsilon', type=float, default=1, help='Epsilon')
    parser.add_argument('--aug_size', type=int, default=4, help='Augmentation size')
    parser.add_argument('--success_before_train', type=int, default=10, help='Success before train')
    parser.add_argument('--load_model', type=str, default='', help='Load model')
    parser.add_argument('--test_rate', type=int, default=100, help='Test rate')
    parser.add_argument('--DNN_type', type=str, default='', help='DNN type')
    parser.add_argument('--use_BN', action='store_true', help='Use BN')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune')
    parser.add_argument('--inference', action='store_true', help='Inference')
    parser.add_argument('--exp_header', type=str, default=None, help='Experiment header')
    return parser.parse_args()


class Sokoban_DNN_L_Model(nn.Module):
    def __init__(self, state_size, action_size, use_BN):
        super(Sokoban_DNN_L_Model, self).__init__()
        self.conv1 = nn.Conv2d(state_size[2], 32, kernel_size=16, stride=16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, action_size)
        self.ReLU = nn.ReLU()
        self.use_BN = use_BN

    def forward(self, x):
        x = self.conv1(x)
        if self.use_BN:
            x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        if self.use_BN:
            x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        if self.use_BN:
            x = self.bn3(x)
        x = self.ReLU(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        x = self.ReLU(x)
        x = self.fc4(x)
        return x


class Sokoban_DNN_Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Sokoban_DNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(state_size[2], 32, kernel_size=16, stride=16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.ReLU(self.conv2(x))
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x


class SOK_Agent:
    def __init__(self, agent_args):
        # Construct DQN models
        self.state_size = (112, 112, 1)
        self.action_size = agent_args.action_size
        self.DNN_type = agent_args.DNN_type
        self.use_BN = agent_args.use_BN
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.batch_size = agent_args.batch_size

        # Replay buffers
        self.replay_buffer = deque(maxlen=agent_args.replay_buffer_size)
        self.prioritized_replay_buffer = deque(maxlen=agent_args.prioritized_replay_buffer_size)
        self.prioritized_replay_batch = agent_args.prioritized_replay_batch

        # Hyper parameters
        self.gamma = agent_args.gamma
        self.epsilon = agent_args.epsilon
        self.epsilon_min = agent_args.epsilon_min
        self.epsilon_decay = agent_args.epsilon_decay
        self.replay_rate = agent_args.replay_rate
        self.replay_times = agent_args.replay_times
        self.learning_rate = agent_args.learning_rate
        self.decay_lr = agent_args.decay_lr
        self.weight_decay = agent_args.weight_decay
        self.update_lr_rate = agent_args.update_lr_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss()
        self.max_steps = agent_args.max_steps
        self.aug_size = agent_args.aug_size
        self.update_beta = agent_args.update_beta
        self.success_before_train = agent_args.success_before_train
        self.fine_tune = agent_args.fine_tune
        self.inference = agent_args.inference
        self.NonMovePenalty = agent_args.NonMovePenalty
        self.CR = agent_args.CR

        # info
        self.solved = 0
        self.exp_header = agent_args.exp_header
        self.max_episodes = agent_args.max_episodes
        self.test_rate = agent_args.test_rate

    def _build_model(self):
        model = None
        if self.DNN_type == 'M':
            model = Sokoban_DNN_Model(self.state_size, self.action_size)
        elif self.DNN_type == 'L':
            model = Sokoban_DNN_L_Model(self.state_size, self.action_size, self.use_BN)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append([state, action, reward, next_state, done])

    def copy_to_prioritized_buffer(self, n):
        for i in range(n):
            self.prioritized_replay_buffer.append(self.replay_buffer[-1 - i])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # if sok is not None:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state_tensor.unsqueeze(0)).detach()[0]

        return act_values.argmax().item()

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            raise Exception('Replay buffer size is not enough')

        if len(self.prioritized_replay_buffer) < self.batch_size // 2:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size // 2)
            minibatch.extend(random.sample(self.prioritized_replay_buffer, self.batch_size // 2))

        aug_size = self.aug_size
        states = torch.zeros((self.batch_size * aug_size, 1, self.state_size[0], self.state_size[1]))
        actions = torch.zeros(self.batch_size * aug_size, dtype=torch.long)
        rewards = torch.zeros(self.batch_size * aug_size)
        next_states = torch.zeros((self.batch_size * aug_size, 1, self.state_size[0], self.state_size[1]))
        statuses = torch.zeros(self.batch_size * aug_size)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            for aug in range(aug_size):
                if aug == 0:
                    states[i * aug_size + aug] = state_tensor
                    actions[i * aug_size + aug] = action
                    rewards[i * aug_size + aug] = reward
                    next_states[i * aug_size + aug] = next_state_tensor
                    statuses[i * aug_size + aug] = 1 if done else 0
                else:
                    org_state = state_tensor.clone().detach()
                    states[i * aug_size + aug] = torch.rot90(org_state, k=aug, dims=[1, 2])
                    actions[i * aug_size + aug] = calc_rot_action(action, aug)
                    rewards[i * aug_size + aug] = reward
                    org_next_state = next_state_tensor.clone().detach()
                    next_states[i * aug_size + aug] = torch.rot90(org_next_state, k=aug, dims=[1, 2])
                    statuses[i * aug_size + aug] = 1 if done else 0

        targets = self.model(states).detach()
        max_actions = torch.argmax(self.model(next_states).detach(), dim=1)
        next_rewards = self.target_model(next_states).detach()

        ind = 0
        for action, reward, next_reward, max_action, done in zip(actions, rewards, next_rewards, max_actions, statuses):
            if not done:
                reward += self.gamma * next_reward[max_action]
            targets[ind][action.int().item()] = reward.long()
            ind += 1

        loss = None
        for e in range(self.replay_times):
            self.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.model(states), targets)
            loss.backward()
            self.optimizer.step()

        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        return loss

    def update_target_model(self):
        # take target model weights and model weights, and update model weights with temperature:
        # model_weights = self.update_beta*target_model_weights + (1-self.update_beta)*model_weights
        model_dict = self.model.state_dict()
        target_dict = self.target_model.state_dict()
        for k in model_dict.keys():
            target_dict[k] = self.update_beta * target_dict[k] + (1 - self.update_beta) * model_dict[k]

    def update_learning_rate(self, lr):
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def test_agent(self, stochastic=False):
        current_epsilon = self.epsilon
        self.epsilon = 0.0
        num_solved = 0
        solved_in_steps = defaultdict(int)
        t_solved = []
        t_unsolved = []

        for t in range(100):
            random.seed(t)
            sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
            sok.set_maxsteps(20)
            steps = 0
            images = []

            state = sok.get_image('rgb_array')
            done = False
            while not done:
                images.append(sok.get_image('rgb_array'))
                steps += 1
                action = self.act(process_frame(state))
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

        if len(t_solved) > 0:
            print("Solved: ", t_solved)
        if len(t_unsolved) > 0:
            print("Unsolved: ", t_unsolved)

        # if solved_in_steps isn't empty - sort it by keys
        if solved_in_steps:
            solved_in_steps = dict(sorted(solved_in_steps.items()))

        self.epsilon = current_epsilon
        print("*" * 30)
        print("Stochastic" if stochastic else "Deterministic")
        print("*" * 30)
        print("Solved: %d" % num_solved)
        print("=" * 30)
        print(solved_in_steps)
        print("*" * 30)

        return num_solved

    def init_sok(self, r):
        random.seed(r + 100)
        Sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1)
        Sok.set_maxsteps(self.max_steps)
        return Sok


def inner_main(main_args):
    agent = SOK_Agent(main_args)
    steps_per_episode = []
    loss_value = np.asarray([])
    losses = []
    evaluation = []
    success_episodes = 0
    total_steps = 0
    running_puzzles = 0
    running_solved = 0

    print('Starting training')
    print('exp name: %s' % agent.exp_header)

    # load model if exists
    if main_args.load_model != '':
        state_dict = torch.load(f"models/{main_args.load_model}.pth")
        agent.model.load_state_dict(state_dict)
        agent.target_model.load_state_dict(state_dict)

    for e in range(agent.max_episodes):

        if e % agent.test_rate == 0 and (main_args.load_model != '' or e > 0):
            print('Testing Agent')
            num_s = agent.test_agent(stochastic=False)
            evaluation.append(num_s)
            if num_s > agent.solved:
                agent.solved = num_s
                # save the model in models folder
                torch.save(agent.model.state_dict(), f"models/{agent.exp_header}_{num_s}_{e}.pth")
            if num_s == 100:
                print("Solved all 100 tests!!!")
                exit(0)
            if agent.inference:
                exit(0)

        if e % 10 == 0 and e > 0:
            print(f"Episode: {e}")

        rand_seed = e
        if agent.fine_tune:
            cases = with_1_step[:250]
            rand_seed = cases[e % len(cases)]

        sok = agent.init_sok(rand_seed)
        random.seed(e)
        running_puzzles += 1

        state = process_frame(sok.get_image('rgb_array'))
        room_state = sok.room_state.copy()
        distances = get_distances(room_state)

        # update learning rate every 1000 episodes
        if e % agent.update_lr_rate == 0 and e > 0:
            agent.update_learning_rate(agent.learning_rate * agent.decay_lr)

        # save plot of loss every 100 episodes
        if e % 100 == 0 and e > 0:
            x_loss = np.arange(0, len(losses))
            plt.plot(x_loss, losses)
            plt.grid()
            plt.savefig(f"Graphs/{agent.exp_header}_loss.png")
            plt.close()

        # save plot of evaluation every 10000 episodes
        if e % 10000 == 0 and e > 0:
            x_eval = np.arange(0, len(evaluation))
            plt.plot(x_eval, evaluation)
            plt.grid()
            plt.savefig(f"Graphs/{agent.exp_header}_eval.png")
            plt.close()

        for step in range(sok.max_steps):
            total_steps += 1
            action = agent.act(state)
            if action < 4:
                next_state, reward, done, _ = sok.step(action + 1)
            else:
                next_state, reward, done, _ = sok.step(action + 5)

            next_state = process_frame(next_state)
            next_room_state = sok.room_state

            if not done:
                reward += box2target_change_reward(room_state, next_room_state, distances, agent.NonMovePenalty, agent.CR)

            agent.remember(state, action, reward, next_state, done)

            state = next_state.copy()
            room_state = next_room_state.copy()

            if total_steps % agent.replay_rate == 0 and success_episodes > agent.success_before_train:
                loss_value_step = agent.replay()
                loss_value = np.append(loss_value, loss_value_step.item())

            if done:
                steps_per_episode.append(step + 1)

                if 3 in sok.room_state:
                    print("SOLVED! Episode %d Steps: %d Epsilon %.4f" % (e, step + 1, agent.epsilon))
                    agent.copy_to_prioritized_buffer(step + 1)
                    success_episodes += 1
                    running_solved += 1

                if (e + 1) % 20 == 0 and e > 0:
                    print(f"{running_solved} | {running_puzzles}")

                    if (e + 1) % 100 == 0:
                        running_puzzles = 0
                        running_solved = 0

                if len(loss_value) > 0:
                    # print('Loss value: %f' % loss_value.mean())
                    losses.append(loss_value.mean())
                break


if __name__ == '__main__':
    # call inner_main with arguments
    args = parse_arguments()
    if args.exp_header is None:
        # find the latest experiment in models folder
        exps = [f for f in os.listdir('models') if os.path.isdir(os.path.join('models', f))]
        args.exp_header = f'exp_{len(exps) + 1}'
    inner_main(args)
