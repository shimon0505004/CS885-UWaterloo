from turtle import update
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    env.reset(seed=seed)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    test_env.reset(seed=seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor

    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## use Z to compute greedy action
        z_min = ZRANGE[0]
        z_max = ZRANGE[1]
        z_i = torch.arange(z_min, z_max, ((z_max - z_min) / ATOMS)).unsqueeze(0)


        ## Q(s,a) = weighted ensamble of returns
        qvalues = (torch.t(torch.nn.functional.softmax(Z(obs).view(-1, ACT_N), dim=1))*z_i).sum(dim=-1)
        action = torch.argmax(qvalues).item()

    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    
    #loss = 0.
    ## TODO: Implement this function

    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)

    z_min = ZRANGE[0]
    z_max = ZRANGE[1]
    delta_z = ((z_max - z_min) / ATOMS)
    z_i = torch.arange(z_min, z_max, delta_z).unsqueeze(0)

    p_wbar_Z_s2_a2 = torch.nn.functional.softmax(Zt(S2).view(len(S2), ATOMS, ACT_N), dim=2)
    values = (torch.transpose(p_wbar_Z_s2_a2,1,2)*z_i).sum(-1)
    greedy_actions = torch.argmax(values,dim=-1)

    probabilities = torch.zeros((ATOMS, len(S)))


    for i_prime in range(ATOMS):
        z_i_prime = z_i.squeeze()[i_prime]
        tau_z_i_prime = torch.clamp((R + (GAMMA* (1-D)*z_i_prime)), min=z_min,max=z_max)
        real_index = (tau_z_i_prime - z_min) / delta_z
        l = torch.floor(real_index)
        u = torch.ceil(real_index)

        gatheredValue = torch.nn.functional.softmax(Zt(S2).view(len(S2), ATOMS, ACT_N)[:,i_prime,:], dim=1).gather(1, greedy_actions.view(-1, 1)).squeeze()

        lowerBound = gatheredValue * (1-D) * (u-real_index)
        upperBound = gatheredValue * (1-D) * (real_index-l)
        probabilities[i_prime] += lowerBound
        probabilities[i_prime] += upperBound

        probabilities[i_prime] /= 2

    probabilities = torch.t(probabilities)

    #For doing vector operation, selecting actions later
    logs = torch.log((torch.nn.functional.softmax(Zt(S).view(len(S), ATOMS, ACT_N), dim=2))+ 1e-08)
    loss = torch.mean(-1*torch.transpose(logs*probabilities.view(len(S), ATOMS,1),1,2).sum(-1).gather(1, A.view(-1, 1)).squeeze())

    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Zt.load_state_dict(Z.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.savefig("C51.png")
    plt.show()