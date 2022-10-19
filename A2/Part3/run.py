import gym
import argparse
import torch
import numpy as np
import random
from utils import save
from cql import CQLDQN, DeepQN
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0', help='environment name, defalult: CartPole-v0')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--cql_alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1e-02)

    args = parser.parse_args()
    return args

def create_dataloader_env(batch_size=256):
    with open('./datasets/cartPole_pure_0.0_0.pkl', 'rb') as f:
        dataset = pickle.load(f)
    tensors = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'dones': []}
    help_list = ['observations', 'actions', 'rewards', 'next_observations', 'dones']
    for i in range(len(dataset)):
        for k, v in dataset[i].items():
            if k in help_list:
                if k != 'dones' and k != 'actions':
                    tensors[k].append(torch.from_numpy(v).float())
                else:
                    tensors[k].append(torch.from_numpy(v).long())

    for k in help_list:
        tensors[k] = torch.cat(tensors[k])

    tensor_dataset = TensorDataset(tensors['observations'],
                                    tensors['actions'],
                                    tensors['rewards'][:, None],
                                    tensors['next_observations'],
                                    tensors['dones'][:, None],)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    eval_env = gym.make('CartPole-v0')
    return dataloader, eval_env

def evaluate(env, agent, seed, eval_runs=5):
    avg_rewards = []
    for i in range(eval_runs):
        state = env.reset(seed=seed)[0]
        cumulative_reward = 0
        done = False
        while not done:
            action = agent.get_action(state, 0.0)
            state, reward, done, trunc, __ = env.step(action)
            cumulative_reward += reward
            done = done | trunc
        avg_rewards.append(cumulative_reward)
    return np.mean(avg_rewards)

def train(config, seed, algo='cqldqn'):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    last10_stats = []
    import os
    path = './datasets'
    if not os.path.exists(path):
        raise Exception('Download datasets first please!')
    dataloader, env = create_dataloader_env(batch_size=config.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_last10 = deque(maxlen=10)

    if algo == 'cqldqn':
        agent = CQLDQN(state_size=env.observation_space.shape[0],
                       action_size=env.action_space.n,
                       hidden_size=config.hidden_size,
                       alpha=config.cql_alpha,
                       device=device,
                       lr=config.lr,
                       tau=config.tau,)
    elif algo == 'dqn':
        agent = DeepQN(state_size=env.observation_space.shape[0],
                       action_size=env.action_space.n,
                       hidden_size=config.hidden_size,
                       device=device,
                       lr=config.lr,
                       tau=config.tau,)
    else:
        raise Exception('Only supports cqldqn and dqn.')
    
    returns = evaluate(env, agent, seed=seed)
    avg_last10.append(returns)

    for i in tqdm(range(1, config.episodes + 1)):
        for states, actions, rewards, next_states, dones in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)   
            dones = dones.to(device)
            batch = (states, actions, rewards, next_states, dones)
            total_loss, cql_loss, bellman_error = agent.learn(batch)
            
        if i % config.eval_every == 0:
            returns = evaluate(env, agent, seed=seed)
            avg_last10.append(returns)
            print('Episode: {}, Test Returns: {}'.format(i, returns))
        
        last10_stats.append(np.mean(avg_last10))

        if i % config.save_every == 0:
            save(agent.q_net, filename='cqldqn_cartpole')

    return last10_stats

def plot_curves(data, colors, labels, path='./results.png'):
    for i, v in enumerate(data):
        mean = np.mean(v, 0)
        std = np.std(v, 0)
        x = np.arange(v.shape[1])
        plt.plot(x, mean, color=colors[i], label=labels[i])
        plt.fill_between(x, np.maximum(mean - std, 0), np.minimum(mean + std, 200), color=colors[i], alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('# Episodes')
    plt.ylabel('Avg Last 10 Returns')
    plt.title('CQLDQN vs DQN')
    plt.grid()
    plt.savefig(path)

def main():
    config = set_config()
    seeds = [1]
    # seeds = [1, 2, 3, 4, 5]
    algos = ['cqldqn', 'dqn']
    plot_data = []
    for algo in algos:
        results = []
        for seed in seeds:
            avg_last10_stats = train(config, seed=seed, algo=algo)
            avg_last10 = np.array(avg_last10_stats).reshape(1, -1)
            results.append(avg_last10)
        data = np.concatenate(results, axis=0)
        plot_data.append(data)

    colors = ['gold', 'deepskyblue']
    labels = ['CQLDQN', 'DQN']
    plot_curves(plot_data, colors, labels)


if __name__ == '__main__':
    main()
