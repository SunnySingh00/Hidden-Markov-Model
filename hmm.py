from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in self.state_dict:
            s = self.state_dict[i]
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            for i in self.state_dict:
                alpha[self.state_dict[i]][t] = self.B[ self.state_dict[i]][self.obs_dict[Osequence[t]]] * np.sum(
                    [self.A[self.state_dict[s_p]][self.state_dict[i]] * alpha[self.state_dict[s_p]][t - 1] for s_p in self.state_dict])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        T = L - 1

        for i in self.state_dict:
            s = self.state_dict[i]
            beta[s][T] = 1

        timeSteps = range(T)
        for t in timeSteps[::-1]:
            for i in self.state_dict:
                s = self.state_dict[i]
                ztp1 = self.obs_dict[Osequence[t + 1]]
                beta[s][t] = np.sum([self.A[s][self.state_dict[s_p]] * self.B[self.state_dict[s_p]][ztp1] *
                                     beta[self.state_dict[s_p]][t + 1] for s_p in self.state_dict])

        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        L = Osequence.shape[0]
        T = L - 1

        alpha = self.forward(Osequence)

        for i in self.state_dict:
            s = self.state_dict[i]
            prob += alpha[s][T]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq = self.sequence_prob(Osequence)
        prob = np.multiply(alpha, beta) / seq
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        for t in range(L - 1):
            for s in range(S):
                for s_ in range(S):
                    prob[s, s_, t] = alpha[s, t] * self.A[s, s_] * self.B[s_, self.obs_dict[Osequence[t + 1]]] * beta[
                        s_, t + 1]

        prob = np.divide(prob, self.sequence_prob(Osequence))
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        states = np.zeros([S, L])

        key_list = list(self.state_dict.keys())
        val_list = list(self.state_dict.values())

        for i in range(S):
            delta[i][0] = self.pi[i] * self.B[i][self.obs_dict[Osequence[0]]]
            states[i][0] = i
        for j in range(1, L):
            key = Osequence[j]
            index = self.obs_dict[key]
            for i in range(S):
                prev = np.array([self.A[k][i] * delta[k][j - 1] for k in range(S)])
                delta[i][j] = self.B[i][index] * np.max(prev)
                states[i][j] = np.argmax(prev)

        path = [0 for i in range(L)]
        path[L - 1] = np.argmax(np.array([delta[i][L - 1] for i in range(S)]))
        for i in range(L - 2, -1, -1):
            path[i] = states[int(path[i + 1])][i + 1]
        for i in range(L):
            path[i] = key_list[val_list.index(int(path[i]))]
        ###################################################
        return path
