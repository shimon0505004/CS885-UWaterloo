import numpy as np
import matplotlib.pyplot as plt

class RL:
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
        #self.counterStateActionPair = np.zeros((self.mdp.nActions, self.mdp.nStates), dtype=np.int)

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

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon=0, temperature=0.0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''


        TRIAL = 100
        avg_cumulative_rewards = np.zeros(nEpisodes)

        Q = initialQ
        policy = np.zeros(self.mdp.nStates, int)

        for trial in range(TRIAL):
            Q = initialQ
            policy = np.zeros(self.mdp.nStates, int)

            #counterStateActionPair = np.zeros((self.mdp.nActions, self.mdp.nStates), dtype=np.int)

            cumulative_rewards = np.zeros(nEpisodes)

            for episode in range(nEpisodes):
                s = s0  # Starting each episode at s0 according to specifications
                counterStateActionPair = np.zeros((self.mdp.nActions, self.mdp.nStates), dtype=np.int)

                for t in range(nSteps):

                    #print("Trial :" + str(trial+1) + " Episode : " + str(episode + 1) + " Timestamp t : " + str(t + 1))

                    # Where the loop starts

                    #   select action via q value (epsilon-greedy or Bolzman temperature or greedy)
                    #   Initializing action with a random value
                    a = np.argmax(Q, axis=0)[s]  # Choosing the best action a* = argmax_a Q(s,a)
                    if epsilon > 0:
                        # perform epsilon exploration (i.e., with probability epsilon, select action at random )
                        #print("Epsilon Greedy Policy")
                        currentProbValue = np.random.random_sample()
                        #print("Current val: " + str(currentProbValue) + " Epsilon: " + str(epsilon))

                        if currentProbValue <= epsilon:
                            #print("Executing Random Action")
                            a = np.random.randint(self.mdp.nActions)
                        else:
                            #print("Chosen Action with best Q-value")
                            a = np.argmax(Q, axis=0)[s]  # Choosing the best action a* = argmax_a Q(s,a)
                    else:
                        if temperature > 0:
                            # perform Boltzmann exploration with temperature parameter
                            #print("Boltzman Exploration Policy")
                            all_a = Q[:, s] / temperature
                            pr_all_a = (np.exp(all_a) / np.sum(np.exp(all_a)))

                            a = 0
                            currentProbValue = np.random.random_sample()
                            currentPrSum = 0.0
                            while a < (len(pr_all_a)-1):
                                currentPrSum += pr_all_a[a]
                                if currentProbValue <= currentPrSum:
                                    break

                                a += 1

                            #print(currentProbValue)
                            #print(pr_all_a)
                            #print(a)
                            #a = np.argmax(pr_all_a)

                        else:
                            # no exploration (i.e., selection action with best Q-value)
                            #print("Chosen Action with best Q-value")
                            a = np.argmax(Q, axis=0)[s]  # Choosing the best action a* = argmax_a Q(s,a)

                    #   End of Action selection

                    # Execute a
                    [reward, nextState] = self.sampleRewardAndNextState(s, a)

                    counterStateActionPair[a][s] += 1
                    learningRate = (1.0 / counterStateActionPair[a][s])

                    Q[a][s] = Q[a][s] + (learningRate * (reward
                                                         + (self.mdp.discount * np.amax(Q[:, nextState]))
                                                         - Q[a][s]
                                                         )
                                         )
                    cumulative_rewards[episode] += (np.power(self.mdp.discount, t) * reward)
                    policy[s] = a
                    s = nextState

            avg_cumulative_rewards += cumulative_rewards

        avg_cumulative_rewards = avg_cumulative_rewards / TRIAL

        xAxis = range(1, (len(avg_cumulative_rewards)+1))
        yAxis = avg_cumulative_rewards
        pltLabel = "Epsilon : " + str(epsilon) + "\nTemp : " + str(temperature)
        plt.plot(xAxis, yAxis, label= pltLabel)


        return [Q, policy]
