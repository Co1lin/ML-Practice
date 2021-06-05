
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
 
import torch
 
torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

writer = SummaryWriter('log_ex')
 
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99  # discount factor
LR = 5e-4
UPDATE_PERIOD = 4
EPS_ED = 0.01
EPS_DECAY = 0.99
SLIDE_LEN = 20
MAX_TIME = 1000
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
env = gym.make('LunarLander-v2')
env.seed(0)
random.seed(0)
 
class Net(nn.Module):
    def __init__(self, h1=128, h2=64):
        super(Net, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(8, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 4)
 
    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.fc3(t)
        return t
 
 
class Experience:
    def __init__(self, cur_state, action, reward, nxt_state, done):
        self.cur_state = cur_state
        self.action = action
        self.reward = reward
        self.nxt_state = nxt_state
        self.done = done
 
 
class Buffer:
    def __init__(self):
        # random.seed(0)
        self.n = BUFFER_SIZE
        self.memory = [None for _ in range(BUFFER_SIZE)]
        self.pt = 0
        self.flag = 0  # to indicate whether the buffer can provide a batch of data
 
    def push(self, experience):
        self.memory[self.pt] = experience
        self.pt = (self.pt + 1) % self.n
        self.flag = min(self.flag + 1, self.n)
 
    def sample(self, sample_size):
        return random.sample(self.memory[:self.flag], sample_size)
 
 
class Agent:
    def __init__(self):
        # random.seed(0)
        self.eps = 1.0
        self.buff = Buffer()
 
        self.policy_net = Net()
        self.target_net = Net()
        self.optim = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.update_networks()
 
        self.total_rewards = []
        self.avg_rewards = []
 
    def update_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
 
    def update_experiences(self, cur_state, action, reward, nxt_state, done):
        experience = Experience(cur_state, action, reward, nxt_state, done)
        self.buff.push(experience)
 
    def sample_experiences(self):
        samples = self.buff.sample(BATCH_SIZE)
        for _, ele in enumerate(samples):
            if _ == 0:
                cur_states = ele.cur_state.unsqueeze(0)
                actions = ele.action
                rewards = ele.reward
                nxt_states = ele.nxt_state.unsqueeze(0)
                dones = ele.done
            else:
                cur_states = torch.cat((cur_states, ele.cur_state.unsqueeze(0)), dim=0)
                actions = torch.cat((actions, ele.action), dim=0)
                rewards = torch.cat((rewards, ele.reward), dim=0)
                nxt_states = torch.cat((nxt_states, ele.nxt_state.unsqueeze(0)), dim=0)
                dones = torch.cat((dones, ele.done), dim=0)
        return cur_states, actions, rewards, nxt_states, dones
 
    def get_action(self, state):
        rnd = random.random()
        if rnd > self.eps:  # exploit
            values = self.policy_net(state)
            act = torch.argmax(values, dim=0).item()
        else:
            act = random.randint(0, 3)
        return act
 
    def optimize_policy(self):
        criterion = nn.MSELoss()
        cur_states, actions, rewards, nxt_states, dones = self.sample_experiences()
 
        cur_states = cur_states.to(device).float()
        actions = actions.to(device).long()
        rewards = rewards.to(device).float()
        nxt_states = nxt_states.to(device).float()
        dones = dones.to(device)
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
 
        # for i in range(10):
        policy_values = torch.gather(self.policy_net(cur_states), dim=1, index=actions.unsqueeze(-1))
        with torch.no_grad():
            next_values = torch.max(self.target_net(nxt_states), dim=1)[0]
            target_values = rewards + GAMMA * next_values * (1 - dones)
 
        target_values = target_values.unsqueeze(1)
 
        self.optim.zero_grad()
        loss = criterion(policy_values, target_values)
 
        loss.backward()
        # print("Loss:", loss.item())
        self.optim.step()
 
        self.policy_net = self.policy_net.cpu()
        self.target_net = self.target_net.cpu()
        return loss.item()
 
    def train(self, episodes):
        loss_index = 0
        for episode in range(episodes):
            total_reward = 0
            cur_state = env.reset()
            cur_state = torch.from_numpy(cur_state)
            for tim in count():
                action = self.get_action(cur_state)
                # img = env.render(mode='rgb_array')
                nxt_state, reward, done, _ = env.step(action)
                nxt_state = torch.from_numpy(nxt_state)
                action = torch.tensor(action).unsqueeze(-1)
                reward = torch.tensor(reward).unsqueeze(-1)
                done = torch.tensor(1 if done else 0).unsqueeze(-1)
 
                self.buff.push(Experience(cur_state, action, reward, nxt_state, done))
                cur_state = nxt_state  # !!!
 
                if self.buff.flag >= BATCH_SIZE and self.buff.pt % UPDATE_PERIOD == 0:
                    self.update_networks()
                    loss = self.optimize_policy()
                    writer.add_scalar('loss', loss, loss_index)
                    loss_index += 1
 
                total_reward += reward.item()
                if done or tim >= MAX_TIME:
                    self.update_rewards(total_reward)
                    writer.add_scalar('tot_reward', total_reward, len(self.total_rewards))
                    break
 
            self.plot_rewards()
 
            if self.eps > EPS_ED:
                self.eps *= EPS_DECAY
 
        torch.save(self.policy_net.state_dict(), 'policy_net.pkl')
 
    def update_rewards(self, total_reward):
        self.total_rewards.append(total_reward)
        cur = len(self.total_rewards) - 1
        rewards = 0
        for i in range(cur, max(-1, cur - SLIDE_LEN), -1):
            rewards += self.total_rewards[i]
        avg = rewards / min(SLIDE_LEN, len(self.total_rewards))
        self.avg_rewards.append(avg)
 
    def plot_rewards(self):
        plt.clf()
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(self.total_rewards, color='r', label='Current')
 
        plt.plot(self.avg_rewards, color='b', label='Average')
        plt.legend()
        plt.pause(0.001)
        print("Episode", len(self.total_rewards))
        print("Current reward", self.total_rewards[-1])
        print("Average reward", self.avg_rewards[-1])
        print("Epsilon", self.eps)
        plt.savefig('Train.jpg')
 
    def test(self, episodes):
        self.eps = 0
        ret = 0
        for episode in range(episodes):
            total_reward = 0
            cur_state = env.reset()
            cur_state = torch.from_numpy(cur_state)
            for tim in count():
                action = self.get_action(cur_state)
                img = env.render(mode='rgb_array')
                nxt_state, reward, done, _ = env.step(action)
                cur_state = torch.from_numpy(nxt_state)
                total_reward += reward
                if done or tim >= MAX_TIME:
                    break
            print("Episode", episode+1)
            print("Current reward", total_reward)
            ret += total_reward
        print("Average reward of", episodes, "episodes:", ret / episodes)
 
agent = Agent()
agent.train(700)
agent.test(100)
 
 
env.close()
