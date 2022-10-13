import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean:
        return 1
    else:
        return 0


# Multi-arm bandit problems (3 arms with probabilities 0.25, 0.5 and 0.75)
T = np.array([[[1]], [[1]], [[1]]])
R = np.array([[0.25], [0.5], [0.75]])
discount = 0.999
mdp = MDP.MDP(T, R, discount)
banditProblem = RL2.RL2(mdp, sampleBernoulli)


fig = plt.figure()
plt.title("Graph 1: UCB, Epsilon Greedy Bandit and Thompson Sampling")
plt.xlabel("Iteration #")
plt.ylabel("Average Rewards Earned at each iteration")

#Epsilon greedy strategy Graph Generation
TRIALS = 1000
N_ITERATIONS = 200

average_rewards = None
for i in range(1,(TRIALS+1)):
    empiricalMeans, rewards, V = banditProblem.epsilonGreedyBandit(nIterations=N_ITERATIONS)
    if average_rewards is None:
        average_rewards = rewards
    else:
        average_rewards = (((average_rewards * (i-1)) + rewards) / i)

plt.plot(range(len(average_rewards)), average_rewards, label= "Epsilon Greedy Bandit")

#thompsonSamplingBandit Graph generation

average_rewards = None
for i in range(1,(TRIALS+1)):
    empiricalMeans, rewards, V = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=N_ITERATIONS)
    if average_rewards is None:
        average_rewards = rewards
    else:
        average_rewards = (((average_rewards * (i-1)) + rewards) / i)

plt.plot(range(len(average_rewards)), average_rewards, label= "Thompson Sampling")

#UC Bandit Graph generation

average_rewards = None
for i in range(1,(TRIALS+1)):
    empiricalMeans, rewards, V = banditProblem.UCBbandit(nIterations=N_ITERATIONS)
    if average_rewards is None:
        average_rewards = rewards
    else:
        average_rewards = (((average_rewards * (i-1)) + rewards) / i)

#Generates log curve
plt.plot(range(len(average_rewards)), average_rewards, label= "UC Bandit")


plt.legend(loc='best')
plt.savefig('A2_Part1.png', bbox_inches='tight')
plt.show()
plt.close()
plt.cla()
plt.clf()
