import numpy as np
import MDP


class RL2:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Output:
        rewards_earned -- total rewards earned in each of the iterations (array of |nIterations+1| entries). Entry 0
        corresponds to 0th iteration, 1 corresponds to 1st iteration, and n+1th entry corresponds to nth iteration
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded

        empirical_mean = np.zeros(self.mdp.nActions)
        state = 0
        n_a = np.zeros(self.mdp.nActions)  # Counts current number of times action a is selected.

        rewards_earned = np.zeros(nIterations + 1)

        for iteration in range(1, (nIterations + 1)):
            epsilon = (1 / iteration)
            p = np.random.random(1)
            if p < epsilon:
                action = np.random.randint(self.mdp.nActions)
            else:
                action = np.argmax(empirical_mean)

            reward, _ = self.sampleRewardAndNextState(state, action)
            empirical_mean[action] = (((n_a[action] * empirical_mean[action]) + reward) / (n_a[action] + 1))
            n_a[action] = n_a[action] + 1  # updating number of times the selected action was encountered.
            rewards_earned[iteration] = reward          #Keeping track of reward earned at iteration

        return rewards_earned


    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Output:
        rewards_earned -- total rewards earned in each of the iterations (array of |nIterations+1| entries). Entry 0
        corresponds to 0th iteration, 1 corresponds to 1st iteration, and n+1th entry corresponds to nth iteration
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded

        state = 0
        rewards_earned = np.zeros(nIterations + 1)

        for iteration in range(1, nIterations + 1):
            empirical_mean = np.zeros(self.mdp.nActions)
            for a in range(self.mdp.nActions):
                k_sample_rewards_for_action_a = np.random.beta(a=prior[a, 0],
                                                               # Alpha hyperparameter for beta distribution for arm a
                                                               b=prior[a, 1],
                                                               # Beta hyperparameter for beta distribution for arm a
                                                               size=k  # number of sampled average rewards
                                                               )  # Sample k rewards from beta distribution for action a.

                empirical_mean[a] = np.mean(k_sample_rewards_for_action_a)  # Estimate empirical average of K sample
                # rewards for action a

            action = np.argmax(empirical_mean)  #choose the best action
            reward, _ = self.sampleRewardAndNextState(state, action)  #Select the reward for the best action
            rewards_earned[iteration] = reward
            hyperparameter = (1-reward)  #If reward = 1, hyperparameter = (1-reward) = 0, alpha hyperparameter.
                                         #If reward = 0, hyperparameter = (1-reward) = 1, beta hyperparameter
            prior[action, hyperparameter] += 1     # Reward always between 0 and 1, we update the prior of the best
            # action based on the reward

        return rewards_earned


    def UCBbandit(self, nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Output:
        rewards_earned -- total rewards earned in each of the iterations (array of |nIterations+1| entries). Entry 0
        corresponds to 0th iteration, 1 corresponds to 1st iteration, and n+1th entry corresponds to nth iteration
        '''


        empirical_mean = np.zeros(self.mdp.nActions)
        state = 0
        n_a = np.zeros(self.mdp.nActions)  # Counts current number of times action a is selected.
        rewards_earned = np.zeros(nIterations + 1)
        epsilon = (10 ** -8)  # To avoid overflow issues.

        for iteration in range(1, nIterations + 1):
            n = iteration#(iteration-1)+epsilon
            action = np.argmax((empirical_mean + (np.sqrt(2 * np.log(n) / (n_a + epsilon)))))
            reward, _ = self.sampleRewardAndNextState(state, action)
            empirical_mean[action] = (((n_a[action] * empirical_mean[action]) + reward)/(n_a[action]+1))
            n_a[action] = n_a[action] + 1  # updating number of times the selected action was encountered.
            rewards_earned[iteration] = reward

        return rewards_earned

