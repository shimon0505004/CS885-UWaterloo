import torch
import numpy as np
import os

def soft_update(source, target, tau):
    for params, target_params in zip(source.parameters(), target.parameters()):
        target_params.data.copy_(tau * params.data + (1.0 - tau) * target_params.data)

def hard_update(source, target):
    for params, target_params in zip(source.parameters(), target.parameters()):
        target_params.data.copy_(params)

def save(model, filename):
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_dir + filename + '.pth')

def evaluate(env, agent, eavl_runs=5):
    avg_rewards = []
    for i in range(eavl_runs):
        state = env.reset()
        cumulative_reward = 0
        done = False
        while not done:
            action = agent.get_action(state, eval=True)
            action = action.detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
        avg_rewards.append(cumulative_reward)
    return np.mean(avg_rewards)