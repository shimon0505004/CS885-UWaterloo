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

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        state = 0
        n_a = np.zeros(self.mdp.nActions)  # Counts current number of times action a is selected.

        rewards_earned = np.zeros(nIterations + 1)
        V = rewards_earned[0]   #Total Reward

        for iteration in range(1, (nIterations + 1)):
            epsilon = (1 / iteration)
            p = np.random.random(1)
            if p < epsilon:
                action = np.random.randint(self.mdp.nActions)
            else:
                action = np.argmax(empiricalMeans)

            [reward, nextState] = self.sampleRewardAndNextState(state, action)

            # Updating empirical mean for selected action with new reward
            previous_total_reward_for_action = (n_a[action] * empiricalMeans[action])
            updated_total_reward_for_action = (previous_total_reward_for_action + reward)
            n_a[action] = n_a[action] + 1  # updating number of times the selected action was encountered.
            new_empirical_mean_for_action = updated_total_reward_for_action / n_a[action]
            empiricalMeans[action] = new_empirical_mean_for_action
            # print("Updated Empirical Means: " + str(empiricalMeans))
            rewards_earned[iteration] = reward          #Keeping track of reward earned at iteration
            V += rewards_earned[iteration]

        return empiricalMeans, rewards_earned, V

    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        state = 0
        rewards_earned = np.zeros(nIterations + 1)
        n_a = np.zeros(self.mdp.nActions)  # Counts current number of times action a is selected.
        V = rewards_earned[0]  # Total Reward

        for iteration in range(1, nIterations + 1):
            R_hat = np.zeros(self.mdp.nActions)
            for a in range(self.mdp.nActions):
                k_sample_rewards_for_action_a = np.random.beta(a=prior[a, 0],
                                                               # Alpha hyperparameter for beta distribution for arm a
                                                               b=prior[a, 1],
                                                               # Beta hyperparameter for beta distribution for arm a
                                                               size=k  # number of sampled average rewards
                                                               )  # Sample k rewards from beta distribution for action a.

                R_hat[a] = np.mean(k_sample_rewards_for_action_a)  # Estimate empirical average of K sample
                # rewards for action a

            action = np.argmax(R_hat)  #choose the best action
            [reward, nextState] = self.sampleRewardAndNextState(state, action)  #Select the reward for the best action

            # Updating empirical mean for selected action with new reward
            previous_total_reward_for_action = (n_a[action] * empiricalMeans[action])
            updated_total_reward_for_action = (previous_total_reward_for_action + reward)
            n_a[action] = n_a[action] + 1  # updating number of times the selected action was encountered.
            new_empirical_mean_for_action = updated_total_reward_for_action / n_a[action]
            empiricalMeans[action] = new_empirical_mean_for_action

            rewards_earned[iteration] = reward
            V += rewards_earned[iteration]

            prior[action, reward] += 1     # Reward always between 0 and 1, we update the prior of the best
            # action based on the reward

        return empiricalMeans, rewards_earned, V

    def UCBbandit(self, nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        state = 0
        n_a = np.zeros(self.mdp.nActions)  # Counts current number of times action a is selected.

        rewards_earned = np.zeros(nIterations + 1)
        V = rewards_earned[0]  # Total Reward

        epsilon = (10 ** -8)  # To avoid overflow issues.

        for iteration in range(1, nIterations + 1):
            n = iteration#(iteration-1)+epsilon
            action = np.argmax((empiricalMeans + (np.sqrt(2 * np.log(n) / (n_a + epsilon)))))

            [reward, nextState] = self.sampleRewardAndNextState(state, action)

            # Updating empirical mean for selected action with new reward
            previous_total_reward_for_action = (n_a[action] * empiricalMeans[action])
            updated_total_reward_for_action = (previous_total_reward_for_action + reward)
            n_a[action] = n_a[action] + 1  # updating number of times the selected action was encountered.
            new_empirical_mean_for_action = updated_total_reward_for_action / n_a[action]
            empiricalMeans[action] = new_empirical_mean_for_action
            # print("Updated Empirical Means: " + str(empiricalMeans))
            rewards_earned[iteration] = reward
            V += rewards_earned[iteration]

        return empiricalMeans, rewards_earned, V
