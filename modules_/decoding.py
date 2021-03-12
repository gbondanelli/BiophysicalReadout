import numpy as np
from numpy.linalg import solve, norm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from numba import njit, prange

class Decoder():
    def __init__(self, responses_all, labels_all):
        if responses_all.shape[1] != len(labels_all):
            raise Exception('length of labels is not equal to length of responses.')
        if len(responses_all.shape) == 1:
            responses_all = responses_all[None,:]
        self.responses_all = responses_all
        labels          = np.unique(labels_all)
        self.responses1 = responses_all[:,labels_all == labels[0]]
        self.responses2 = responses_all[:,labels_all == labels[1]]
        self.labels_all = labels_all

class FisherDecoder(Decoder):

    def __init__(self, responses_all, labels_all):
        super().__init__(responses_all, labels_all)
        self._decoder_done = False
        self._predict_done = False

    def get_decoder(self):
        self.responses_all  = np.hstack((self.responses1, self.responses2))
        self.ntrials        = len(self.labels_all)
        mu1                 = np.mean(self.responses1, axis=1, keepdims=1)
        mu2                 = np.mean(self.responses2, axis=1, keepdims=1)
        resp1_meansubtr     = self.responses1 - mu1
        resp2_meansubtr     = self.responses2 - mu2
        S1                  = np.dot(resp1_meansubtr, resp1_meansubtr.T)
        S2                  = np.dot(resp2_meansubtr, resp2_meansubtr.T)
        S_tot               = S1 + S2
        mu1                 = mu1[:,0]
        mu2                 = mu2[:,0]
        w                   = solve(S_tot, mu2-mu1)
        w                   = w/norm(w)
        self.w_fisher       = w
        self.b_fisher       = .5 * np.dot( self.w_fisher, (mu1 + mu2) )
        self._decoder_done   = True
        return self.w_fisher, self.b_fisher

    def decision_function(self, x):
        return np.dot(self.w_fisher, x) - self.b_fisher

    def get_predicted_stim(self,x):
        if len(x.shape) ==1:
            x = x[:,None]
        responses_proj_S    = self.decision_function(x)
        self.predicted_labels_all   = np.sign(responses_proj_S)
        self._predict_done          = True
        return self.predicted_labels_all

    def get_accuracy(self):
        if not self._decoder_done:
            _, _ = self.get_decoder()
        if not self._predict_done:
            _ = self.get_predicted_stim(self.responses_all)
        self.accuracy = sum(self.labels_all == self.predicted_labels_all)/float(self.ntrials)
        return self.accuracy

    def decode(self):
        _,_ = self.get_decoder()
        _   = self.get_predicted_stim(self.responses_all)
        _   = self.get_accuracy()

class LinearSVM(Decoder):

    def __init__(self, responses_all, labels_all):
        super().__init__(responses_all, labels_all)
        self.C_values   = np.linspace(0.001,10,10)
        n_features      = responses_all.shape[0]
        n_samples       = len(labels_all)
        self.dual       = False if n_samples > n_features else True
        self.K          = 3

    def set_C_values(self, C_values):
        self.C_values = C_values

    def set_K(self,K):
        self.K = K

    def get_accuracy(self):
        X = self.responses_all.T
        y = self.labels_all
        n_C = len(self.C_values)
        cv_accuracies = np.empty((self.K, n_C))
        for i_C in range(n_C):
            clf = LinearSVC(dual = self.dual, C = self.C_values[i_C])
            accuracy = cross_val_score(clf, X, y, cv = self.K, n_jobs = 8)
            cv_accuracies[:,i_C] = accuracy
        self.cv_accuracies = cv_accuracies
        best_idx = np.argmax(np.mean(cv_accuracies,0))
        self.C_best = self.C_values[best_idx]
        return self.cv_accuracies[:,best_idx]

    def get_predicted_labels(self):
        clf = LinearSVC(dual = self.dual, C = self.C_best)
        clf.fit(self.responses_all.T, self.labels_all)
        self.predicted_labels = clf.predict(self.responses_all.T)
        return self.predicted_labels










##

