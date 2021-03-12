from numpy import random
import numpy as np
from numpy.linalg import qr, inv
from scipy.special import expit
import stattools
from decoding import *

class StimChoiceDec():
    def __init__(self, responses_all, stimuli, choices):
        self.responses_all  = responses_all
        self.stimuli        = stimuli
        self.choices        = choices
        N                   = responses_all.shape[0]
        self.ntrials        = responses_all.shape[1]
        self.observed       = range(N)
        self.not_observed   = []

    def _all_but(self, n, l):
        v = range(n)
        return [i for i in v if i not in l]

    def opt_lin_decoder(self, labels, idx_to_keep = None):

        if labels == 'stimulus':
            labels_all = self.stimuli
        elif labels == 'choice':
            labels_all = self.choices

        if idx_to_keep is None:
            r = self.responses_all
        else:
            r = self.responses_all[idx_to_keep,:]

        decoder = FisherDecoder(r, labels_all)
        decoder.decode()
        w                       = decoder.w_fisher
        b                       = decoder.b_fisher
        predicted_labels_all    = decoder.predicted_labels_all
        accuracy                = decoder.accuracy
        if labels == 'stimulus':
            self.predicted_labels_S_all = predicted_labels_all
            self.accuracy_S             = accuracy
            self.w_opt_S                = w
            self.b_opt_S                = b
        elif labels == 'choice':
            self.predicted_labels_C_all = predicted_labels_all
            self.accuracy_C             = accuracy
            self.w_opt_C                = w
            self.b_opt_C                = b

    def set_choice_regressors(self, type = 'r', idx_to_keep = None):
        """ Set default regressors as neural activity of all neurons;
        idx_to_keep: indeces of specific neurons to keep as regressors
        if type = 'decoded_stim' regressors are set as the identity of decoded stimulus."""
        if type == 'r' or type == 'r_s':
            if idx_to_keep is None:
                self.regressors = self.responses_all
            else:
                self.regressors = self.responses_all[idx_to_keep,:]
        elif type == 'sdec' or type == 'sdec_s':
            self.regressors = self.predicted_labels_S_all[None,:]
        if type == 'r_s' or type == 'sdec_s':
            self.regressors = np.vstack((self.regressors, self.stimuli))
        if type == 's':
            self.regressors = self.stimuli[None,:]

    def _shuffle(self, r, idx_to_shuffle = None):
        """ Shuffle trial labels across cells independently.
        r is (n_features x n_trials)."""
        n_features = r.shape[0]
        n_trials = r.shape[1]
        r_sh = r
        if idx_to_shuffle is None:
            idx_to_shuffle = range(n_features)
        for i in range(n_features):
            if i in idx_to_shuffle:
                r_sh[i,:] = r[i,random.permutation(range(n_trials))]
        return r_sh

    def _shuffle_regressors(self, idx_to_shuffle = None):
        if idx_to_shuffle is None:
            idx_to_shuffle = range(len(self.regressors)-1) # shuffle all regressors except last one, i.e. the stimulus
        self.regressors_shuffled = self._shuffle(self.regressors, idx_to_shuffle)
        return self.regressors_shuffled

    def regress_choicesCV(self, K=10): # default: 10-fold cv
        """ Regress generated choices against regressors.
        regressors: (nfeatures, nsamples) or (nsamples,) array."""

        betas, bias, predicted_labels_C_all, accuracy_C = stattools.logistic_regression_cv(self.choices, self.regressors, K)
        return betas, bias, predicted_labels_C_all, accuracy_C

    def compute_BP(self, betas, bias, type, idx_to_keep = None):
        """ Compute BP according to p(c=s) = sum_{x,s} p(c=s|s,x)*p(s,x),
        where here p(s,x) = 1/ntrials. """

        if idx_to_keep is not None:
            betas[self._all_but(len(betas),idx_to_keep)] = 0.

        if type == 'rates':
            y1 = self.responses_all[:,self.stimuli == -1]
            y2 = self.responses_all[:,self.stimuli == +1]
        if type == 'regressors':
            y1 = self.regressors[:,self.stimuli == -1]
            y2 = self.regressors[:,self.stimuli == +1]
        if type == 'shuffled':
            y_sh = self._shuffle_regressors()
            y1 = y_sh[:,self.stimuli == -1]
            y2 = y_sh[:,self.stimuli == +1]

        x1      = np.dot(betas, y1) + bias
        x2      = np.dot(betas, y2) + bias
        p_c_1   = 1 - expit(x1) # = p(c=-1|s=-1,x)
        p_c_2   = expit(x2)
        BP      = np.sum(p_c_1) + np.sum(p_c_2)
        BP      = BP/float(self.ntrials)
        return BP

    def compute_exp_BP(self):
        return np.sum(self.stimuli == self.choices)/float(self.ntrials)

    def compute_BV(self):
        ds2 = (max(self.stimuli) - min(self.stimuli))**2
        BP  = np.sum(self.stimuli == self.choices)/self.ntrials
        return ds2 * (1-BP)

    def compute_exp_BV(self):
        ds2 = (max(self.stimuli) - min(self.stimuli))**2
        return ds2*(1-self.compute_exp_BP())

    def compute_BP_observed(self, n_shuffles):

        BP_A = np.empty(n_shuffles)
        BP_B = np.empty(n_shuffles)
        BP_full = self.compute_exp_BP()
        self.set_choice_regressors('r_s', self.observed)
        betas, bias, _, _ = self.regress_choicesCV()
        for i in range(n_shuffles):
            BP_B[i] = self.compute_BP(betas, bias, 'shuffled')
            BP_A[i] = BP_full-BP_B[i]
        return BP_A, np.mean(BP_A)


class GaussianRateModel(StimChoiceDec):

    def __init__(self, N):
        self.N = N

    def set_mean_resp(self, r_mean):
        self.mean_resp_1 = r_mean
        self.mean_resp_2 = -r_mean

    def set_mean_resp_random(self, r_norm):
        r_mean = random.normal(0,r_norm/np.sqrt(self.N), self.N)
        self.mean_resp_1 = r_mean
        self.mean_resp_2 = -r_mean

    def set_covariances(self, cov1, cov2 = None):
        self.cov1 = cov1
        self.cov2 = cov1 if cov2 is None else cov2

    def generate_responses(self, ntrials):
        """ Set responses to stim 1 and 2 according to a multivariate Gaussian,
        and stimulus labels (+ - 1)."""
        random.seed()
        self.ntrials = ntrials
        m1           = self.mean_resp_1
        m2           = self.mean_resp_2
        cov1         = self.cov1
        cov2         = self.cov2
        responses1     = random.multivariate_normal(m1, cov1, int(ntrials/2)).T  # N x ntrials/2
        responses2     = random.multivariate_normal(m2, cov2, ntrials-int(ntrials/2)).T  # N x ntrials/2
        self.responses_all  = np.hstack((responses1, responses2))
        stimulus1      = -1*np.ones(responses1.shape[1])
        stimulus2      = +1*np.ones(responses2.shape[1])
        self.stimuli   = np.hstack((stimulus1, stimulus2))

    def set_observed_population(self, **kwargs):

        N = self.N
        if 'fraction' in kwargs.keys():
            f       = kwargs['fraction']
            idx_A   = random.choice(np.arange(self.N), int(f*N), replace = False)
        else:
            idx_A   = kwargs['observed']
        idx_B               = np.asarray(self._all_but(N,idx_A))
        self.observed       = idx_A
        self.not_observed   = idx_B

    def set_choice_readout(self, overlap, noise = 0, b_C = None):
        """ Set correlation between stimulus decoder and
        (true) choice linear readout w_opt_C
        to overlap. Also sets offset b_C; default b_C = b_opt_S.
        Add noise in the choice decoder. Resulting variance of the
        stim/choice decoder overlap equals noise^2/2. """

        self.b_C  =  self.b_opt_S if b_C is None else b_C
        w_S       =  self.w_opt_S
        temp_vec  =  random.normal(0,1,len(w_S))
        temp_vec  =  temp_vec/norm(temp_vec)
        q,_       =  qr(np.array([w_S, temp_vec]).T)
        w_S_orth  =  q[:,1]
        w_S_orth  =  w_S_orth/norm(w_S_orth)
        w_C       =  overlap * w_S + np.sqrt(1-overlap**2) * w_S_orth
        w_C       =  w_C + noise * random.normal(0,1/np.sqrt(len(w_C)), len(w_C))
        w_C       =  w_C/norm(w_C)
        self.w_C  =  w_C

    def generate_choices(self, beta = np.inf):
        """ Generate choices with a linear readout w_C
        and probability given by a logistic func with parameter beta."""
        self.w_C            = self.w_C/norm(self.w_C)
        w_C                 = self.w_C
        b_C                 = self.b_C

        # reverse sign of w_C if the projection of the mean responses
        # for stim +1 on w_C is negative (-> ensures that BP is always > 0.5)
        w_C                 = w_C * np.sign(np.dot(w_C,np.mean(self.responses_all[:,self.stimuli == +1],1))-b_C)
        responses_proj_C    = beta * (np.dot(w_C, self.responses_all) - b_C)
        prob                = expit(responses_proj_C)
        self.choices        = np.empty(self.ntrials)
        self.betas_true     =  beta * w_C
        self.bias_true      = -beta * b_C

        for i in range(self.ntrials):
            self.choices[i] = +1 if random.uniform() < prob[i] else -1

    def compute_BP_true(self, idx_to_keep = None):
        return self.compute_BP(self.betas_true, self.bias_true, 'rates', idx_to_keep)

class RateModel(StimChoiceDec):
    def __init__(self,N,t):
        self.N = N
        self.t = t
        self.dt = t[1]-t[0]
        self.nsteps = len(t)
        self.f = lambda x : x

    def set_connectivity(self,J):
        self.J = J

    def set_nonlinearity(self, type ='ReLu', power = 1):
        if type is None:
            self.f = lambda x : x
        if type == 'ReLu':
            self.f = lambda x : (x*(x>0))**power

    def set_external_input(self, mode, type = 'constant'):
        input = np.empty((self.N,self.nsteps))
        if type == 'constant':
            input = np.tile(mode[:,None],self.nsteps)
        self.input = input
        self.mode = mode

    def set_external_input_noise(self, sigma):
        noise = sigma*random.normal(0,1,(self.N, self.nsteps))
        noise = noise/np.sqrt(self.dt)
        self.noise = noise

    def set_initial_condition(self, r0):
        if r0 == '0':
            self.r0 = np.zeros(self.N)
        elif r0 == 'ss':
            self.r0 = np.dot(inv(np.identity(self.N) - self.J), self.mode)
        else:
            self.r0 = r0

    def simulate(self):
        r = np.empty((self.N, self.nsteps))
        J = self.J
        input = self.input + self.noise
        r[:,0] = self.r0
        for i in range(self.nsteps-1):
            r[:,i+1] = self.f( r[:,i] + self.dt * (-r[:,i] + ( np.dot(J,r[:,i]) + input[:,i] ) ) )
        self.r = r
        return r
















##

