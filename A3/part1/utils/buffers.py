import collections
import numpy as np
import random
import torch

# Replay Buffer
class ReplayBuffer:

    # create replay buffer of size N
    def __init__(self, N, OBS_N, t):
        self.S = t.f(torch.zeros((N, OBS_N)))
        self.A = t.l(torch.zeros((N)))
        self.Ret = t.f(torch.zeros((N)))
        self.LP = t.f(torch.zeros((N)))
        self.i = 0
        self.N = N
        self.t = t
        self.filled = 0
    
    # add states, actions, returns, log_probs
    def add(self, states, actions, returns, log_probs):
        M = states.shape[0]
        self.filled = min(self.filled+M, self.N)
        assert(M <= self.N)
        for j in range(M):
            self.S[self.i] = self.t.f(states[j, :])
            self.A[self.i] = self.t.l(actions[j])
            self.Ret[self.i] = self.t.f(returns[j])
            self.LP[self.i] = self.t.f(log_probs[j])
            self.i = (self.i + 1) % self.N
    
    # sample: return minibatch of size n
    def sample(self, n):
        minibatch = random.sample(range(self.filled), n)
        S, A, Ret, LP = [], [], [], []
        
        for mbi in minibatch:
            s, a, ret, lp = self.S[mbi], self.A[mbi], self.Ret[mbi], self.LP[mbi]
            S += [s]; A += [a]; Ret += [ret]; LP += [lp]

        return torch.stack(S), torch.stack(A), torch.stack(Ret), torch.stack(LP)