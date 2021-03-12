import numpy as np
from numpy.linalg import qr, norm, eigh
from numpy import random
from sklearn.linear_model import LogisticRegressionCV
from decoding import *
from numba import njit,prange
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from scipy.special import *


def generate_covariance_matrix(principal_axes, principal_variances, orth_variance):
     """Parameters:
     principal_axes: N x M array containing the eigenvectors of the covariance matrix C
     principal_variances: M-dim array containing eigenvalues of C
     orth_variance: value of the remaining N-M eigenvalue of C."""

     N = principal_axes.shape[0]
     M = principal_axes.shape[1]
     variances = np.hstack((principal_variances, orth_variance*np.ones(N-M)))
     covariance = np.zeros((N,N))
     principal_axes = principal_axes/norm(principal_axes,axis=0, keepdims=1)
     A = random.normal(0,1,(N,N))
     A[:,:M] = principal_axes
     pc_axes, _ = qr(A)
     for var, pc in zip(list(variances), list(pc_axes.T)):
         covariance += var*np.outer(pc,pc)

     return covariance

def logistic_regression_cv(choices, regressors, K):  # uses K-fold CV

    if len(regressors.shape) == 1:
        regressors = regressors[:,None]
    else:
        regressors = regressors.T
    clf = LogisticRegressionCV(cv = K, penalty = 'l1', solver = 'saga', scoring = 'neg_log_loss', random_state = 0, n_jobs = 20)
    clf.fit(regressors, choices)
    betas = clf.coef_[0]
    bias = clf.intercept_
    pred_choices = clf.predict(regressors)
    pred_acc = clf.score(regressors, choices)
    return clf, betas, bias, pred_choices, pred_acc

def PCA(X):
    """ X is a matrix #cells x #timepoints
     returns the eigvals and eigvects of the correlation matrix in descending order."""

    if len(X.shape) == 3:
        S1 = X.shape[0]
        S2 = X.shape[1]
        S3 = X.shape[2]
        X = reshape(X,(S1,S2*S3),order = 'F')
    X = X-np.mean(X,axis = 1,keepdims = True)
    C = np.dot(X,X.T)
    d,V = eigh(C)
    V = V[:,np.argsort(d)[::-1]]
    d = d[np.argsort(d)[::-1]]
    return d,V

@njit(parallel=True)
def generate_poisson(R, dt, ntrials):
    """
    :param R: N x #timesteps matrix of firing rates
    :param dt: length of timesteps
    :param ntrials: # of trials
    :return: N x #timesteps x ntrials matrix of spike trains (0-1 valued)
    """
    N = R.shape[0]
    nsteps = R.shape[1]
    rmax = np.amax(R)
    rmin = np.amin(R)
    if rmax * dt > 1:
        print('Careful: r*dt = p > 1 at some point.')
    spike_trains = np.zeros((N,nsteps,ntrials))
    for i_trial in prange(ntrials):
        for i_n in prange(N):
            for i_t in prange(nsteps):
                p = R[i_n, i_t] * dt
                u = random.uniform(0,1)
                spike_trains[i_n, i_t, i_trial] = 1 if u < p else 0
    return spike_trains


def compute_consistency(R, stimuli, ntrials=100, nsplits = 1):
    N = R.shape[0]
    ntrials_tot = R.shape[1]
    consistency = np.empty(ntrials)
    trials = random.choice(range(ntrials_tot), ntrials, replace = False)
    for i_trial in range(len(trials)):
        v = trials[i_trial]
        dec_acc = np.empty(nsplits)
        for i_split in range(nsplits):
            pool1 = random.choice(range(N), int(N / 2), replace=False).tolist()
            pool2 = [n for n in range(N) if n not in pool1]

            decoder1 = FisherDecoder(R[pool1,:], stimuli)
            _,_ = decoder1.get_decoder()
            s1 = decoder1.get_predicted_stim(R[pool1,v])

            decoder2 = FisherDecoder(R[pool2, :], stimuli)
            _, _ = decoder2.get_decoder()
            s2 = decoder2.get_predicted_stim(R[pool2, v])

            dec_acc[i_split] = 1 if s1 == s2 else 0

        consistency[i_trial] = 1 if np.sum(dec_acc) > float(nsplits/2) else 0
    return consistency

def number_coincidences(s1, s2, t):
    ntrials = s1.shape[1]
    dt = t[1] - t[0]
    rate1 = np.mean(np.sum(s1,0)/t[-1])
    rate2 = np.mean(np.sum(s2,0) / t[-1])
    n_coincidences = np.zeros(ntrials)
    for i in range(ntrials):
        n_coincidences[i] = np.sum([1 for v in range(len(t)) if s1[v,i] == 1 and s2[v,i] == 1 ])
    n_coincidence_norm = n_coincidences / rate1/ rate2/ dt/ t[-1]
    return n_coincidences, n_coincidence_norm

def generate_correlated_rates(rS, muS, sigma, noisecorr, t):
    N = len(rS)
    r1 = np.empty((N,len(t)))
    r2 = np.empty((N, len(t)))

    avg_rate  = np.tile(muS[0]*rS[:,None],len(t))
    indep_noise = np.dot(np.diag(rS), random.normal(0,1,(N,len(t))))
    shared_noise = np.outer(rS,random.normal(0,1,len(t)))
    r1 = avg_rate + sigma * ( np.sqrt(1-noisecorr**2) * indep_noise + noisecorr * shared_noise )

    avg_rate = np.tile(muS[1]*rS[:,None],len(t))
    indep_noise = np.dot(np.diag(rS), random.normal(0, 1, (N, len(t))))
    shared_noise = np.outer(rS, random.normal(0, 1, len(t)))
    r2 = avg_rate + sigma * (np.sqrt(1 - noisecorr ** 2) * indep_noise + noisecorr * shared_noise)

    return r1, r2

def generate_correlated_rates_diffcorr(mu1, mu2, sigma, epsilon, ntrials):
    N = len(mu1)
    f_prime = mu1 - mu2
    f_prime = f_prime/norm(f_prime)
    A = random.normal(0,1,(N,N))
    A[:,0] = f_prime
    q,_ = qr(A)
    V = q[:,1:]
    sigma0 = sigma * V @ V.T
    cov = sigma0 + epsilon * np.outer(f_prime,f_prime)
    r1 = random.multivariate_normal(mu1, cov, ntrials).T
    r2 = random.multivariate_normal(mu2, cov, ntrials).T
    return r1, r2

@njit
def low_pass_filter(r,tau,t):
    T = t[-1]
    dt = t[1] - t[0]
    Tau_n = int(tau/dt)
    N = r.shape[0]
    nsteps = r.shape[1]
    meanr = np.empty((N,nsteps))
    noise = np.empty((N,nsteps))
    for i_n in range(N):
        a = np.mean(r[i_n,:])
        meanr[i_n,:] = np.array([a]*nsteps)
        noise[i_n,:] = r[i_n,:] - a
    r_lp = np.empty((N,nsteps))
    r_lp[:,0] = r[:,0]
    for i in range(nsteps-1):
        dr_lp = dt/tau * (-r_lp[:,i] + meanr[:,i] + np.sqrt(2*tau)*noise[:,i]/np.sqrt(dt))
        r_lp[:,i+1] = r_lp[:,i] + dr_lp
    return r_lp

@njit(parallel = True)
def compute_rate_from_spike_trains(spike_train, timescale, t, nsamples):
    dt = t[1] - t[0]
    T = t[-1]
    timescale = int(timescale/dt)
    N = spike_train.shape[0]
    ntrials = spike_train.shape[2]
    rates = np.empty((N,nsamples,ntrials))
    for i in prange(nsamples):
        while True:
            c = random.choice(len(t)-timescale)
            if c+timescale+1 < len(t):
                break
        for i_trial in prange(ntrials):
            for i_N in prange(N):
                b = spike_train[i_N, c:c+timescale+1, i_trial]
                rates[i_N,i,i_trial] = np.sum(b) / timescale / dt
    return rates

@njit(parallel = True)
def compute_rate_from_spike_trains2(spike_train, timescale, t):
    dt = t[1] - t[0]
    T = t[-1]
    timescale = int(timescale/dt)
    N = spike_train.shape[0]
    ntrials = spike_train.shape[2]
    nsamples = int(len(t)/timescale)
    rates = np.empty((N,nsamples,ntrials))
    for i in prange(nsamples):
        for i_trial in prange(ntrials):
            for i_N in prange(N):
                c = i * timescale
                b = spike_train[i_N, c:c+timescale, i_trial]
                rates[i_N,i,i_trial] = np.sum(b) / timescale / dt
    return rates
    
def decode_from_spikes(st1, st2, timescale, t, nsamples, decoder, time_lag = None):
    """
    :param st1: spike trains for stimulus 1
    :param st2: spike trains for stimulus 2 - spike trains needs to have shape (n_units x n_timepoints x n_trials)
    :param decoder: decoder type
    :return: decoding accuracy ()
    """
    if time_lag is None:
        r1,r2 = [compute_rate_from_spike_trains2(st, timescale, t) for st in [st1,st2]]
    else:
        dt = t[1]-t[0]
        time_lag = int(time_lag/dt)
        r1 = compute_rate_from_spike_trains2(st1, timescale, t)
        r2 = compute_rate_from_spike_trains2(st2[:,time_lag:,:], timescale, t)
        n  = min(r1.shape[1], r2.shape[1])
        r1 = r1[:,:n,:]
        r2 = r2[:,:n,:]

    ntrials = r1.shape[2]
    nsamples = r1.shape[1]
    accuracy = np.empty(ntrials)
    predicted_labels = np.empty((2*nsamples, ntrials))
    for i in range(ntrials):
        R1 = r1[:,:,i]
        R2 = r2[:,:,i]
        R = np.hstack((R1,R2))
        S = np.hstack((-np.ones(nsamples), +np.ones(nsamples)))
        if decoder == 'linsvm':
            clf = LinearSVM(R,S)
            acc = clf.get_accuracy()
            accuracy[i] = np.mean(acc)
            predicted_labels[:,i] = clf.get_predicted_labels()
        if decoder == 'fisher':
            clf = FisherDecoder(R,S)
            accuracy[i] = clf.get_accuracy()
    return accuracy, S, predicted_labels

def decode_from_rates(R, S, decoder):
    if R.shape[1] != len(S):
        print('R and S need to have consistent shape')
    ntrials = R.shape[2]
    nsamples = R.shape[1]
    accuracy = np.empty(ntrials)
    predicted_labels = np.empty((nsamples, ntrials))
    for i in range(ntrials):
        r = R[:,:,i]
        if decoder == 'linsvm':
            clf = LinearSVM(r,S)
            acc = clf.get_accuracy()
            accuracy[i] = np.mean(acc)
            predicted_labels[:,i] = clf.get_predicted_labels()
        if decoder == 'fisher':
            clf = FisherDecoder(r,S)
            accuracy[i] = clf.get_accuracy()
    return accuracy, S, predicted_labels

def compute_ISI(st,t):
    """
    :param st: spike train - shape (n_timepoints,n_trials)
    :return: vector of inter-spike intervals
    """
    n_timepoints = st.shape[0]
    n_trials = st.shape[1]
    st = np.reshape(st,(n_timepoints*n_trials))
    dt = t[1] - t[0]
    spike_times = np.where(st == 1)[0]
    ISI = dt*(np.diff(spike_times))
    return ISI

def equalize_trials(x, frac, n_subsamplings):
    """
    :param n_subsamplings: number of subsamplings
    :param x: vector of 0 and 1 (for error and correct trials)
    :param frac: subsampling fraction: n = frac * min(#error,#correct)
    :return: indices of equalized error and correct trials
    """
    idx_corr = np.where(x == 1)[0]
    idx_err = np.where(x == 0)[0]
    n_err = len(idx_err)
    n_corr = len(idx_corr)
    n = int(frac * min(n_err,n_corr))
    equalized_idx = np.empty((2, n, n_subsamplings))
    for i in range(n_subsamplings):
        equalized_idx[0, :, i] = np.random.choice(idx_corr, n, replace = False)
        equalized_idx[1, :, i] = np.random.choice(idx_err, n, replace=False)
    return equalized_idx.astype(int)

def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))
def scalar_product_lag(x, y, lag=0):
    y1 = y[lag:]
    x1 = x[:len(y1)]
    return np.dot(x1,y1)
def pearson_lag(x,y,lag = 0):
    y1 = y[lag:]
    x1 = x[:len(y1)]
    return pearsonr(x1,y1)[0]

def jitter_spike_train(st,jitter_window,dt):
    N = st.shape[0]
    nsteps = st.shape[1]
    st_jittered = st.copy()
    jitter_window_int_half = int(jitter_window/dt/2)
    for i_N in range(N):
        spikes = np.where(st[i_N,:] == 1)[0]
        for idx_spike in spikes:
            st_jittered[i_N,idx_spike] = 0
            while(1):
                J = random.randint(-jitter_window_int_half, jitter_window_int_half)
                if idx_spike+J<0 or idx_spike+J>=nsteps:
                    continue
                elif st_jittered[i_N,idx_spike+J] == 0:
                    st_jittered[i_N,idx_spike+J] = 1
                    break
    return st_jittered

def jitter_spike_train2(st,jitter_window,dt):
    N = st.shape[0]
    nsteps = st.shape[1]
    st_jittered = st.copy()
    jitter_window_int = int(jitter_window/dt)
    for i_N in range(N):
        spikes = np.where(st[i_N,:] == 1)[0]
        for idx_spike in spikes:
            idx_window = int(idx_spike / jitter_window_int)
            st_jittered[i_N,idx_spike] = 0
            while(1):
                J = random.randint(idx_window*jitter_window_int, (idx_window + 1)*jitter_window_int)
                if J<0 or J>=nsteps:
                    continue
                elif st_jittered[i_N,J] == 0:
                    st_jittered[i_N,J] = 1
                    break
    return st_jittered

def frac_dev_expl(y,X,model):
    D = 2*log_loss(y, model.predict_proba(X))
    b0 = logit(float(len(y[y==1])/len(y)))
    model_null = model
    model_null.coef_ = np.zeros(model.coef_.shape)
    model_null.intercept_[0] = b0
    D0 = 2*log_loss(y, model_null.predict_proba(X))
    return 1-D/D0

