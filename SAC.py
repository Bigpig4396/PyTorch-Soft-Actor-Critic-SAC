from env_GoTogether import EnvGoTogether
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

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

class P_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(P_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_score = self.fc3(x)
        return F.softmax(action_score, dim=-1)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.fc4 = nn.Linear(state_dim, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, action_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q1 = self.fc3(h)

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q2 = self.fc3(h)
        return q1, q2

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SAC():
    def __init__(self, state_dim, action_dim):
        super(SAC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_net = P_net(state_dim, action_dim)
        self.q_net = Q_net(state_dim, action_dim)
        self.q_target_net = Q_net(state_dim, action_dim)
        self.gamma = 0.99
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-5)
        hard_update(self.q_target_net, self.q_net)
        self.update_interval = 100
        self.update_count = 0

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.p_net.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        # print('q', q.detach().numpy())
        # print('a_prob', action_prob.detach().numpy())
        return action.item()

    def train(self, batch):
        state = batch[0]
        action = batch[1]
        reward = batch[2]
        next_state = batch[3]
        state = torch.from_numpy(state).float().squeeze(1)
        next_state = torch.from_numpy(next_state).float().squeeze(1)
        T = state.size()[0]

        # calculate V
        self.alpha = self.log_alpha.exp()
        next_q1, next_q2 = self.q_target_net.forward(next_state)
        next_q = torch.min(next_q1, next_q2)
        next_a_prob = self.p_net.forward(next_state)
        next_v = next_a_prob*(next_q-self.alpha*torch.log(next_a_prob))
        next_v = torch.sum(next_v, 1)

        # train Q
        q1, q2 = self.q_net.forward(state)
        expect_q = q1.clone()
        for i in range(T):
            expect_q[i, action[i]] = reward[i] + self.gamma * next_v[i]
        loss1 = self.loss_fn(q1, expect_q.detach())
        loss2 = self.loss_fn(q2, expect_q.detach())
        loss = loss1 + loss2
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # train Actor
        q1, q2 = self.q_net.forward(state)
        q = torch.min(q1, q2)
        a_prob = self.p_net.forward(state)
        ploss = a_prob*(self.alpha*torch.log(a_prob)-q)
        ploss = torch.sum(ploss)
        ploss = ploss / T
        self.p_optimizer.zero_grad()
        ploss.backward()
        self.p_optimizer.step()

        # Updating alpha wrt entropy
        self.alpha = self.log_alpha.exp()
        a_prob = self.p_net.forward(state)
        H = - a_prob*torch.log(a_prob)
        alpha_loss = -a_prob*(self.alpha*torch.log(a_prob) + self.alpha * H)
        alpha_loss = torch.sum(alpha_loss)
        alpha_loss = alpha_loss / T
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # print('q_loss', loss.detach().item(), 'p_loss', ploss.detach().item(), 'alpha_loss', alpha_loss.detach().item())
        self.update_count += 1
        if self.update_count % self.update_interval == 0:
            soft_update(self.q_target_net, self.q_net, 0.7)

    def load_model(self):
        self.q_net = torch.load('SAC_q_net.pkl')
        self.p_net = torch.load('SAC_p_net.pkl')

    def save_model(self):
        torch.save(self.q_net, 'SAC_q_net.pkl')
        torch.save(self.p_net, 'SAC_p_net.pkl')

if __name__ == '__main__':
    state_dim = 4
    action_dim = 4
    max_epi = 500
    max_mc = 1000
    epi_iter = 0
    mc_iter = 0
    acc_reward = 0
    reward_curve = []
    batch_size = 32
    replay_buffer = ReplayBuffer(50000)
    env = EnvGoTogether(13)
    agent = SAC(state_dim, action_dim)
    for epi_iter in range(max_epi):
        for mc_iter in range(max_mc):
            # env.render()
            state = env.get_state()
            action = agent.get_action(state)
            group_list = [action, 2]
            reward, done = env.step(group_list)
            next_state = env.get_state()
            acc_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer.sample(batch_size))
            if done:
                break
        print('epi', epi_iter, 'reward', acc_reward / mc_iter, 'MC', mc_iter, 'alpha', agent.alpha)
        env.reset()
        acc_reward = 0