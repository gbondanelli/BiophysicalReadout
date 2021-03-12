import sys
sys.path.insert(0,'..')
# sys.path.append('..')
import numpy as np
from math import log2
from numba import njit, prange
# import BROJA_2PID


# class ComputeII:
#     def __init__(self, p_src):
#         self.p_src = np.asarray(p_src)
#         self.n_S = self.p_src.shape[0]
#         self.n_C = self.p_src.shape[2]
#         self.n_R = int(self.p_src.size/(self.n_S*self.n_C))

#     def calculate(self):
#         p_cr_s = dict()
#         p_sr_c = dict()

#         for c in range(0, self.n_C):
#             for r in range(0, self.n_R):
#                 for s in range(0, self.n_S):
#                     # here the target is the third index: e.g. in p_rc_s --> S is the target
#                     p_cr_s[(c,r,s)] = float(self.p_src[s][r][c])
#                     p_sr_c[(s,r,c)] = float(self.p_src[s][r][c])

#         pid_s = BROJA_2PID.pid(p_cr_s)
#         pid_c = BROJA_2PID.pid(p_sr_c)
#         return min(pid_s['SI'], pid_c['SI'])


def contingency_matrix(R,S):
    """
    :param R: respose: shape (trials,)
    :param S: stimuli: shape (ntrials,)
    :return: contingency matrix P(R,S)
    """
    ntrials = R.shape[0]
    Rvals = np.unique(R)
    Svals = np.unique(S)
    P = np.empty((len(Rvals),len(Svals)))
    for i_r in range(len(Rvals)):
        for i_s in range(len(Svals)):
             P[i_r,i_s] = np.sum( (R == Rvals[i_r]).astype(float) * (S == Svals[i_s]).astype(float) ) / float(ntrials)
    return P


def contingency_matrix_2features(R,S):
    """
    :param R: respose: shape (trials,)
    :param S: stimuli: shape (ntrials,)
    :return: contingency matrix P(R,S)
    """
    ntrials = R.shape[1]
    R1 = R[0,:]
    R2 = R[1,:]
    R1vals = np.unique(R1)
    R2vals = np.unique(R2)
    Svals = np.unique(S)
    P = np.empty((len(R1vals),len(R2vals),len(Svals)))
    for i_r1 in range(len(R1vals)):
        for i_r2 in range(len(R2vals)):
            for i_s in range(len(Svals)):
                P[i_r1,i_r2,i_s] = np.sum( (R1 == R1vals[i_r1]).astype(float) * \
                                            (R2 == R2vals[i_r2]).astype(float) * \
                                            (S == Svals[i_s]).astype(float) ) / float(ntrials)
    return P

def compute_Shannon_info(P):
    P1 = P[P!=0]
    P1 = P1/np.sum(P1)
    return -np.sum(P1*np.log2(P1))

def _computeMI(P):
    P_R = np.sum(P,1)
    P_S = np.sum(P,0)
    P_SR = P
    H_R = compute_Shannon_info(P_R)
    H_S = compute_Shannon_info(P_S)
    H_RS = compute_Shannon_info(P_SR)
    return H_R + H_S - H_RS

def _computeMI_2features(P):
    P_R = np.sum(P,2)
    P_S = np.sum(P,(0,1))
    P_SR = P
    H_R = compute_Shannon_info(P_R)
    H_S = compute_Shannon_info(P_S)
    H_RS = compute_Shannon_info(P_SR)
    return H_R + H_S - H_RS

def computeMI(R,S):
    ntrials = R.shape[1]
    MI = np.empty(ntrials)
    for i in range(ntrials):
        P = contingency_matrix(R[:,i],S)
        MI[i] = _computeMI(P)
    return MI

def computeMI_2features(R,S):
    ntrials = R.shape[2]
    MI = np.empty(ntrials)
    for i in range(ntrials):
        P = contingency_matrix_2features(R[:,:,i],S)
        MI[i] = _computeMI_2features(P)
    return MI




##

