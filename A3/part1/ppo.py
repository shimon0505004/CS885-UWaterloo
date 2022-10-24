import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore")

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATES = {      # Learning rate for optimizers
    "Pi": 1e-4,
    "V": 1e-3,
}
EPOCHS = 300            # Total number of episodes to learn over
EPISODES_PER_EPOCH = 20 # Epsides per epoch
TEST_EPISODES = 10      # Test episodes
HIDDEN = 64             # Hidden size
BUFSIZE = 10000         # Buffer size
CLIP_PARAM = 0.1        # Clip parameter
MINIBATCH_SIZE = 64     # Minibatch size
TRAIN_EPOCHS = 50       # Training epochs
OBS_N = None
ACT_N = None
Pi = None

# Create environment
# Create replay buffer
# Create networks
# Create optimizers
def create_everything(seed):

    global OBS_N, ACT_N, penalty_param, Pi
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.ConstrainedCartPole(), 200)
    env.reset(seed=seed)
    test_env = utils.envs.TimeLimit(utils.envs.ConstrainedCartPole(), 200)
    test_env.reset(seed=10+seed)
    OBS_N = env.observation_space.shape[0]  # State space size
    ACT_N = env.action_space.n              # Action space size
    buf = utils.buffers.ReplayBuffer(BUFSIZE, OBS_N, t)
    Pi = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    V = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1)
    ).to(DEVICE)
    OPTPi = torch.optim.Adam(Pi.parameters(), lr = 1e-4)
    OPTV = torch.optim.Adam(V.parameters(), lr = 1e-3)
    return env, test_env, buf, Pi, V, OPTPi, OPTV

# Policy
def policy(env, obs):

    probs = torch.nn.Softmax(dim=-1)(Pi(t.f(obs)))
    return np.random.choice(ACT_N, p = probs.cpu().detach().numpy())

# Training function
def update_networks(epi, buf, Pi, V, OPTPi, OPTV):
    
    # Sample from buffer
    S, A, returns, old_log_probs = buf.sample(MINIBATCH_SIZE)
    log_probs = torch.nn.LogSoftmax(dim=-1)(Pi(S)).gather(1, A.view(-1, 1)).view(-1)

    # Critic update
    OPTV.zero_grad()
    objective1 = (returns - V(S)).pow(2).mean()
    objective1.backward()
    OPTV.step()

    # Actor update
    OPTPi.zero_grad()
    advantages = returns - V(S)
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-CLIP_PARAM, 1+CLIP_PARAM)
    ppo_obj = torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    objective2 = -ppo_obj
    objective2.backward()
    OPTPi.step()

# Play episodes
# Training function
def train(seed):

    global penalty_param
    print("Seed=%d" % seed)

    # Create environments, buffer, networks, optimizers
    env, test_env, buf, Pi, V, OPTPi, OPTV = create_everything(seed)

    # Train PPO for EPOCH times
    testRs = []
    testRcs = []
    last25testRs = []
    last25testRcs = []
    print("Training:")
    pbar = tqdm.trange(EPOCHS)
    for epi in pbar:

        # Collect experience
        all_S, all_A = [], []
        all_returns = []
        for epj in range(EPISODES_PER_EPOCH):
            
            # Play an episode and log episodic reward
            S, A, R, Rc = utils.envs.play_episode(env, policy, constraint=True)
            all_S += S[:-1] # ignore last state
            all_A += A
            
            # Create returns 
            discounted_rewards = copy.deepcopy(R)
            for i in range(len(R)-1)[::-1]:
                discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
            discounted_rewards = t.f(discounted_rewards)
            all_returns += [discounted_rewards]
                       
        S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
        returns = torch.cat(all_returns, dim=0).flatten()

        # add to replay buffer
        log_probs = torch.nn.LogSoftmax(dim=-1)(Pi(S)).gather(1, A.view(-1, 1)).view(-1)
        buf.add(S, A, returns, log_probs.detach())

        # update networks
        for i in range(TRAIN_EPOCHS):
            update_networks(epi, buf, Pi, V, OPTPi, OPTV)

        # evaluate
        Rews = []
        Rewcs = []
        for epj in range(TEST_EPISODES):
            S, A, R, Rc = utils.envs.play_episode(test_env, policy, constraint=True)
            Rews += [sum(R)]
            Rewcs += [sum(Rc)]
        testRs += [sum(Rews)/TEST_EPISODES]
        testRcs += [sum(Rewcs)/TEST_EPISODES]

        # Show mean episodic test reward over last 25 episodes
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        last25testRcs += [sum(testRcs[-25:])/len(testRcs[-25:])]
        pbar.set_description("R25(%g), Rc25(%g)" % (last25testRs[-1], last25testRcs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()
    
    return last25testRs, last25testRcs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(ax, vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    ax.plot(range(len(mean)), mean, color=color, label=label)
    ax.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # Train for different seeds
    curves = []
    curvesc = []
    for seed in SEEDS:
        R, Rc = train(seed)
        curves += [R]
        curvesc += [Rc]

    # Plot the curve for the given seeds
    plot_arrays(ax[0], curves, 'b', 'ppo')
    plot_arrays(ax[1], curvesc, 'b', 'ppo')
    plt.legend(loc='best')
    plt.show()