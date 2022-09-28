import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt

''' Construct simple MDP as described in Lecture 1b Slides 17-18'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
plt.title("Graph 1: Varying Exploration Probability")
plt.xlabel("Episode #")
plt.ylabel("Average Cumulative Discounted Rewards")
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(20)
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.1)
print("Epsilon : 0.1")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.3)
print("Epsilon : 0.3")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.5)
print("Epsilon : 0.5")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('TestRL_Graph1.png', bbox_inches='tight')
plt.close()
plt.cla()
plt.clf()


# Test Q-learning
plt.title("Graph 2: Varying Boltzman Temperature")
plt.xlabel("Episode #")
plt.ylabel("Average Cumulative Discounted Rewards")
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(20)

[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,temperature=0)
print("temperature : 0")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")

[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,temperature=10)
print("temperature : 10")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")

[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,temperature=20)
print("temperature : 20")
print("Q : " + str(Q))
print("Policy : " + str(policy))
print("\n")


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('TestRL_Graph2.png', bbox_inches='tight')
plt.close()
plt.cla()
plt.clf()
