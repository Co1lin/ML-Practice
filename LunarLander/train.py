import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from tensorboardX import SummaryWriter

# torch.autograd.set_detect_anomaly(True)

def random_play(env):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print(reward, end=', ')

class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.linears = nn.Sequential(
            nn.Linear(8, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, 4),
        )

    def forward(self, state):
        out = self.linears(state)
        return out

class Experience:

    def __init__(self, cur_state, action, reward, next_state, done):
        self.cur_state = torch.Tensor([cur_state]).to(device)
        self.action = torch.LongTensor([action]).to(device)
        self.reward = torch.Tensor([reward]).to(device)
        self.next_state = torch.Tensor([next_state]).to(device)
        self.done = torch.LongTensor([done]).to(device)

class Buffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_states = torch.Tensor()
        self.action = torch.LongTensor()
        self.reward = torch.Tensor()
        self.next_state = torch.Tensor()
        self.done = torch.LongTensor()

    def push(self, exp: Experience):
        if self.cur_states.shape[0] > 0:
            self.cur_states = torch.cat([self.cur_states, exp.cur_state])
            self.action = torch.cat([self.action, exp.action])
            self.reward = torch.cat([self.reward, exp.reward])
            self.next_state = torch.cat([self.next_state, exp.next_state])
            self.done = torch.cat([self.done, exp.done])
        else:
            self.cur_states = exp.cur_state
            self.action = exp.action
            self.reward = exp.reward
            self.next_state = exp.next_state
            self.done = exp.done


        if self.cur_states.shape[0] > self.max_size:
            self.cur_states = self.cur_states[-self.max_size:]
            self.action = self.action[-self.max_size:]
            self.reward = self.reward[-self.max_size:]
            self.next_state = self.next_state[-self.max_size:]
            self.done = self.done[-self.max_size:]

    def sample(self, num):
        indices = random.sample(list(range(self.cur_states.shape[0])), num)
        return self.cur_states[indices], self.action[indices], self.reward[indices], self.next_state[indices], self.done[indices]

    def __len__(self):
        return self.cur_states.shape[0]

class Agent():

    def __init__(self, e_net, num_actions=4):
        self.e_net = e_net
        self.num_actions = num_actions
        # self.greedy_eps = greedy_eps

    def act(self, state, greedy_eps):
        state = torch.Tensor(state).to(device)
        if random.random() < greedy_eps:
            q_values = self.e_net(state)
            action = torch.argmax(q_values, dim=-1).item()
        else:
            action = random.randint(0, self.num_actions - 1)
        return action

class Trainer():

    def __init__(self, env, buf_size=10000):
        self.env = env
        self.buffer = Buffer(buf_size)
        self.e_net = QNetwork().to(device)
        self.t_net = QNetwork().to(device)
        self.agent = Agent(self.e_net)
        self.sample_size = 64
        self.decay = 0.99

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.e_net.parameters(), lr=5e-4)

        self.writer = SummaryWriter('log')


    def sample_and_optimize(self):
        cur_s, action, reward, next_s, done = self.buffer.sample(self.sample_size)
        e_values = self.e_net(cur_s)
        e_values = torch.gather(e_values, dim=1, index=action.unsqueeze(-1))
        e_values = e_values.flatten()

        with torch.no_grad():
            next_s_values = self.t_net(next_s)
            next_s_max_val = torch.max(next_s_values, dim=-1)[0]
            t_values = reward + (1 - done) * self.decay * next_s_max_val

        self.optimizer.zero_grad()
        loss = self.loss_fn(e_values, t_values)
        loss.backward()
        self.optimizer.step()

        return loss

    def sync_network(self):
        self.t_net.load_state_dict(self.e_net.state_dict())

    def train(self, episodes=10000, max_time=1000, update_interval=4, save_path='ckpts'):

        self.sync_network()

        greedy_eps = 0.01
        max_greedy_eps = 0.99

        bar = tqdm(range(episodes))

        update_time = 0
        test_time = 0

        for eps in bar:
            cur_s = env.reset()

            tot_reward = []

            for time in range(max_time):
                action = self.agent.act(cur_s, greedy_eps)
                next_s, reward, done, _ = env.step(action)

                tot_reward.append(reward)
                self.buffer.push(Experience(cur_s, action, reward, next_s, done))

                if time % update_interval == 0 and len(self.buffer) > self.sample_size:
                    loss = self.sample_and_optimize()
                    self.writer.add_scalar('loss', loss, update_time)
                    self.sync_network()
                    update_time += 1

                if done or time == max_time - 1:
                    self.writer.add_scalar('tot_reward', sum(tot_reward), eps)
                    break

                cur_s = next_s
            # end for time

            if greedy_eps < max_greedy_eps:
                greedy_eps += 0.005

            if eps % 100 == 0:
                file_path = f'{save_path}/{eps}.pkl'
                print(file_path)
                self.save(file_path)
                test_res = self.test(file_path)
                print(test_res)
                self.writer.add_scalar('test_res', test_res, test_time)
                test_time += 1

        # end for eps


    def save(self, path):
        d = {
            'e_net': self.e_net.state_dict(),
            't_net': self.t_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(d, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=torch.device(device))
        self.e_net.load_state_dict(ckpt['e_net'])
        self.t_net.load_state_dict(ckpt['t_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def test(self, path, episodes=5, rendering=False):
        self.load(path)

        tot_rewards = []

        for eps in range(episodes):
            cur_s = env.reset()

            tot_reward = []

            while True:
                action = self.agent.act(cur_s, 1)
                next_s, reward, done, _ = env.step(action)

                if rendering:
                    img = env.render(mode='rgb_array')

                tot_reward.append(reward)

                if done:
                    tot_rewards.append(sum(tot_reward))
                    break

                cur_s = next_s
            # end for time
        # end for eps
        return sum(tot_rewards) / len(tot_rewards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LunarLanding',
        description='Train Script',
        allow_abbrev=True,
    )

    parser.add_argument('-m', '--model', dest='model', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    env = gym.make('LunarLander-v2')
    random_play(env)

    trainer = Trainer(env)

    if args.model is not None:
        trainer.load(args.model)

    trainer.train()
