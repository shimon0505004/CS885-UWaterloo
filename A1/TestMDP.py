from MDP import *

''' Construct simple MDP as described in Lecture 1b Slides 17-18'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
policy = mdp.extractPolicy(V)
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
