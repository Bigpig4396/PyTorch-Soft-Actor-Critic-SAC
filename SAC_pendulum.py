import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = -20
        self.log_std_max = 2


        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, self.action_dim)
        self.log_std = nn.Linear(32, self.action_dim)

    def sample_action(self, ):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()

    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        x = F.relu(self.fc3(h2))
        x = F.relu(self.fc4(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action_0 = torch.tanh(mean + std * z)
        action = action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        z = Normal(0, 1).sample(mean.shape)
        action = torch.tanh(mean + std * z)
        action = action.detach().numpy()
        return action[0][0]

class SAC():
    def __init__(self, state_dim, action_dim):
        self.soft_q_net1 = QNetwork(state_dim, action_dim)
        self.soft_q_net2 = QNetwork(state_dim, action_dim)
        self.target_soft_q_net1 = QNetwork(state_dim, action_dim)
        self.target_soft_q_net2 = QNetwork(state_dim, action_dim)
        self.policy_net = Actor(state_dim, action_dim)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.reward_scale = 10.0
        self.target_entropy = -1. * action_dim

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=3e-4)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def get_action(self, state):
        a = self.policy_net.get_action(state)
        return a

    def train(self, batch):
        state, action, reward, next_state, done = batch

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(-1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1)

        predicted_q_value1 = self.soft_q_net1.forward(state, action)
        predicted_q_value2 = self.soft_q_net2.forward(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # Updating alpha wrt entropy
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Training Q Function
        predict_target_q1 = self.target_soft_q_net1.forward(next_state, new_next_action)
        predict_target_q2 = self.target_soft_q_net2.forward(next_state, new_next_action)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + self.gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predict_q1 = self.soft_q_net1.forward(state, new_action)
        predict_q2 = self.soft_q_net2.forward(state, new_action)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim)
    max_epi_iter = 1000
    max_MC_iter = 500
    batch_size = 64
    replay_buffer = ReplayBuffer(50000)
    train_curve = []
    for epi in range(max_epi_iter):
        state = env.reset()
        acc_reward = 0
        for MC_iter in range(max_MC_iter):
            # print("MC= ", MC_iter)
            env.render()
            action1 = agent.get_action(state)
            next_state, reward, done, info = env.step(action1*2)
            acc_reward = acc_reward + reward
            replay_buffer.push(state, action1, reward, next_state, done)
            state = next_state
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer.sample(batch_size))
            if done:
                break
        print('Episode', epi, 'reward', acc_reward)
        train_curve.append(acc_reward)
    plt.plot(train_curve, linewidth=1, label='SAC')
    plt.show()