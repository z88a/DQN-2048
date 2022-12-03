import random

from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import logging
import logger


class DQN_NET(nn.Module):
    def __init__(self):
        super(DQN_NET, self).__init__()
        self.fc1 = nn.Linear(16,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,4)

    def forward(self,x):
        x = x.reshape(-1,16).float()
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        y = self.fc3(x)
        return y


class DQN_CNN(nn.Module):
    def __init__(self):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,128,(2,2))
        self.conv2 = nn.Conv2d(128,128,(2,2))
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,4)

    def forward(self,x):
        x = x.view(-1,1,4,4).float()
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x= x.view(-1,512)
        x = F.tanh(self.fc1(x))
        y = self.fc2(x)
        return y


class DQN_CNN_onehot(nn.Module):
    def __init__(self):
        super(DQN_CNN_onehot, self).__init__()
        self.conv1 = nn.Conv2d(16,64,(2,2))
        self.conv2 = nn.Conv2d(64,128,(2,2))
        self.conv3 = nn.Conv2d(128,32,(2,2))
        self.fc1 = nn.Linear(32,4)

    def forward(self, x):
        x = x.view(-1,16,4,4).float()
        x = F.relu(self.conv1(x))  # -1,64,3,3
        x = F.relu(self.conv2(x))  # -1,128,2,2
        x = F.relu(self.conv3(x))  # -1,32,1,1
        x= x.view(-1,32)
        y = self.fc1(x)
        return y


class DQN:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-4
        self.epsilon = 1
        self.epsilon_start = 0.9
        self.epsilon_decay = 0.9999995
        self.epsilon_min = 0.00001
        self.gamma = 0.9
        self.gamma_max = 0.99
        self.gamma_up = 1.00000005  #1.000001
        self.policy_model = DQN_CNN_onehot().to(self.device)
        self.target_model = DQN_CNN_onehot().to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())  # 使两个网络初始权重一致
        # Q_target parameters are frozen.
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.action_space = 4
        self.memory = deque(maxlen=5000)
        self.memory2 = deque(maxlen=2000)
        self.batch_size = 256
        self.min_memory = 2000
        # self.loss_func = torch.nn.SmoothL1Loss()
        self.loss_func = F.mse_loss
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(),lr=self.learning_rate)
        self.sync_every = 50
        self.learn_every = 10
        self.curr_step = 0
        self.learn_step = 0

    def select_action(self, state, ramdom_flag=True, force_rand=False):
        if force_rand or (ramdom_flag and random.random() < self.epsilon):
            action = random.randint(0, self.action_space-1)
        else:
            state = torch.from_numpy(state).to(self.device)
            action_values = self.policy_model.forward(state)
            action = torch.argmax(action_values, axis=1).item()

        # adjust epsilon
        # self.epsilon *= self.epsilon_decay
        self.curr_step += 1
        return action

    def cache(self,state,action,new_state, reward, done):
        state = torch.tensor(state)
        action = torch.tensor(action)
        new_state = torch.tensor(new_state)
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        self.memory.append((state, new_state, action, reward, done,))
        if 1 in state[11:,:,:]:
            self.memory2.append((state, new_state, action, reward, done,))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, new_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, new_state, action, reward, done

    def recall2(self):
        batch = random.sample(self.memory2, self.batch_size)
        state, new_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, new_state, action, reward, done

    def learn(self):
        def get_curr_q(state, action):
            return self.policy_model(state.to(self.device))[np.arange(0,self.batch_size),action]

        def get_target_q(reward, new_state, done):
            new_state_q = self.policy_model(new_state.to(self.device))
            best_action = torch.argmax(new_state_q,axis=1)
            new_q = self.target_model(new_state.to(self.device))[np.arange(0,self.batch_size), best_action]
            return reward.to(self.device)+(1-done.float().to(self.device))*self.gamma*new_q

        def sync_param():
            self.target_model.load_state_dict(self.policy_model.state_dict())

        if self.learn_step % self.sync_every == 0:
            sync_param()

        if len(self.memory) <= self.batch_size:
            return None

        if self.curr_step % self.learn_every != 0:
            return None

        self.learn_step += 1
        state, new_state, action, reward, done = self.recall()
        c_q = get_curr_q(state,action).float()
        t_q = get_target_q(reward,new_state,done).float()
        loss = self.loss_func(c_q,t_q.to(self.device))
        self.optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 更新梯度
        self.optimizer.step()  # 更新参数

        if len(self.memory2) > self.batch_size:
            # logging.info('!!!learn from memory2!!!')
            state, new_state, action, reward, done = self.recall2()
            c_q = get_curr_q(state,action).float()
            t_q = get_target_q(reward,new_state,done).float()
            loss = self.loss_func(c_q,t_q.to(self.device))
            self.optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 更新梯度
            self.optimizer.step()  # 更新参数

        self.epsilon = self.epsilon_start**(self.curr_step/40000)
        self.epsilon = max(self.epsilon_min, self.epsilon)
        self.gamma = min(self.gamma_up*self.gamma,self.gamma_max)
        return loss.item()
