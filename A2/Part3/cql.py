import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from models import QNetwork, GaussianPolicy, DQN
from utils import soft_update
import numpy as np

class CQLDQN(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 hidden_size=256,
                 tau=1e-02,
                 gamma=0.99,
                 lr=1e-03,
                 optim_class=optim.Adam,
                 alpha=1.0,):
        super(CQLDQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.gamma = gamma
        assert alpha > 0
        self.alpha = alpha

        self.q_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_q_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim_class(self.q_net.parameters(), lr=lr)
        self.q_criterion = nn.MSELoss()

    def get_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                action = np.argmax(q_values.cpu().data.numpy())
        return action

    def learn(self, batch):
        """
        Performs a learning step.
        :param batch: (tuple) contains states, actions, rewards, next_states, dones(terimal flag)
        :return: total loss, cql_loss, bellman_error
        cql_loss = E_{s ~ d^D, a ~ pi}[Q(s, a)] - E_{(s, a) ~ d^D}[Q(s, a)]
        bellman_error = E_{(s, a) ~ d^D, s' ~ P(s, a), a' ~ pi}[(r + Q_target(s', a') - Q(s, a))^2]
        total_loss = alpha * cql_loss + bellman_error
        """
        states, actions, rewards, next_states, dones = batch
        
        return 0, 0, 0


class DeepQN(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 hidden_size=256,
                 tau=1e-02,
                 gamma=0.99,
                 lr=1e-03,
                 optim_class=optim.Adam,):
        super(DeepQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.gamma = gamma

        self.q_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_q_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim_class(self.q_net.parameters(), lr=lr)
        self.q_criterion = nn.MSELoss()

    def get_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                action = np.argmax(q_values.cpu().data.numpy())
        return action

    def learn(self, batch):
        """
        Performs a learning step.
        :param batch:(tuple) contains states, actions, rewards, next_states, dones(terimal flag)
        :return: total_loss, cql_loss, bellman_error
        """
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            q_targets_next = self.target_q_net(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (1 - dones) * self.gamma * q_targets_next
        q_values = self.q_net(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1))
        q_pi = q_values.max(1)[0]

        bellman_error = self.q_criterion(q_pred, q_targets)
        self.q_optimizer.zero_grad()
        bellman_error.backward()
        self.q_optimizer.step()
        
        soft_update(self.q_net, self.target_q_net, self.tau)
        return bellman_error.item(), 0, bellman_error.item()