import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV
        iterId = 0
        epsilon = np.inf
        policy = np.zeros(len(initialV), dtype=int)

        print("===========================================================================")
        print("Executing Value Iteration")
        print("--------------------------------------------")
        print("Iteration : " + str(iterId) + " V : " + str(V) + " Policy: " + str(policy))
        while iterId < nIterations and epsilon > tolerance:
            iterId = iterId + 1

            Ta_V = np.matmul(self.T, V)
            gamma_Ta_V = self.discount * Ta_V
            all_possible_values = self.R + gamma_Ta_V
            policy = np.argmax(all_possible_values, axis=0)     # Choose the best actions for each state, policy means keep
            V_new = np.amax((all_possible_values), axis=0)          # Choose the best action values for each state
            # np.round/np.around does not work for 0.5 so not reducing to 2 decimal places
            V_diff = (V_new - V)
            V = V_new
            epsilon = np.linalg.norm(V_diff, np.inf)
            print("Iteration : " + str(iterId) + " V : " + str(V) + " Policy: " + str(policy))
        print("--------------------------------------------")
        print("Final State values after " + str(iterId) + " iterations , V: " + str(V) + " Policy: " + str(policy))
        print("===========================================================================")

        return [V,iterId,epsilon]

    def extractPolicy(self, V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        print("***************************************************************")
        print("Executing Policy Extraction")
        print("--------------------------------------------")

        Ta_V = np.matmul(self.T, V)
        gamma_Ta_V = self.discount * Ta_V
        all_possible_values = (self.R + gamma_Ta_V)        # Get values for all possible state transition in this state
        print("All Values for All possible state transition: ")
        print(all_possible_values)

        policy = np.argmax(all_possible_values, axis=0)     # Choose the best actions for each state, policy means keep
        print("Extracted Policy : " + str(policy))
        print("--------------------------------------------")

        #track of action chosen at timestamp t, instead of choosing only value
        max_values = [all_possible_values[policy[i]][i] for i in range(len(policy))]
        print("Values Corresponding to Selected Policies: " + str(max_values))
        print("***************************************************************")

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        #V = np.zeros(self.nStates)

        print("***************************************************************")
        print("Executing Policy Evaluation")
        print("--------------------------------------------")
        print("Policy : " + str(policy))
        print("Evaluating a policy by solving a system of linear equations")

        R_policy = np.array([self.R[policy[i]][i] for i in range(len(policy))])
        T_policy = np.array([self.T[policy[i]][i] for i in range(len(policy))])
        gamma_T_policy = self.discount * T_policy
        assert gamma_T_policy.shape[0] == gamma_T_policy.shape[1], "gamma_T_policy matrix should be square"
        V = np.matmul(np.linalg.inv(np.identity(len(policy)) - gamma_T_policy), R_policy)

        print("V : " + str(V))
        print("***************************************************************")

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy #np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0

        print("===========================================================================")
        print("Executing Policy Iteration")
        print("--------------------------------------------")
        print("Iteration : " + str(iterId) + " policy : " + str(policy))

        while iterId < nIterations:
            iterId = iterId + 1
            V = self.evaluatePolicy(policy)
            policy_new = self.extractPolicy(V)

            print("Iteration : " + str(iterId) + " policy : " + str(policy_new))
            if np.array_equal(policy_new, policy):
                break
            else:
                policy = policy_new


        print("--------------------------------------------")
        print("Final policy after " + str(iterId) + " iterations , policy: " + str(policy))
        print("===========================================================================")

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0

        return [policy,V,iterId,epsilon]
        
