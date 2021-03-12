from numpy import random
import numpy as np
from numba import njit,jit,prange

# class LIF():
#     def __init__(self, Vr, Vth, tau, t):
#         self.Vr = Vr
#         self.V0 = Vr
#         self.Vth = Vth
#         self.tau = tau
#         self.t = t
#         self.dt = t[1] - t[0]
#
#     def set_FF_weights(self,w):
#         self.w = w
#
#     def set_input(self, input):
#         self.input = input
#
#     def simulate(self, mu0, sigma):
#         nsteps = len(self.t)
#         ntrials = self.input.shape[2]
#         dt = self.dt
#         tau = self.tau
#         Vr = self.Vr
#         Vth = self.Vth
#         spike_trains = np.zeros((nsteps,ntrials))
#         Vtot = np.empty((nsteps,ntrials))
#         for i_trial in range(ntrials):
#             V = np.empty(nsteps)
#             V[0] = self.V0
#             for i in range(nsteps-1):
#                 dV = dt/tau * (Vr-V[i]+ mu0 + sigma * random.normal(0,1)/np.sqrt(dt)) + np.dot(self.w, self.input[:,i,i_trial])
#                 V[i+1] = V[i] + dV
#                 if V[i+1] > Vth:
#                     spike_trains[i+1,i_trial] = 1
#                     V[i] = 20
#                     V[i+1] = Vr
#             Vtot[:,i_trial] = V
#         self.Vtot = Vtot
#         self.spike_trains = spike_trains
#         return self.spike_trains, self.Vtot

@njit(parallel = True)
def simulate_lif(Vr, Vth, tau, t, w, w_mf, input, input_mf, mu0, sigma):
    V0 = Vr
    dt = t[1] - t[0]
    nsteps = len(t)
    ntrials = input.shape[2]
    spike_trains = np.zeros((nsteps, ntrials))
    Vtot = np.empty((nsteps, ntrials))
    w = np.ascontiguousarray(w)
    w_mf = np.ascontiguousarray(w_mf)
    for i_trial in prange(ntrials):
        V = np.empty(nsteps)
        V[0] = V0
        for i in range(nsteps - 1):
            dV = dt / tau * ( Vr - V[i] + mu0 +
                              sigma * random.normal(0, 1) / np.sqrt(dt) +
                              tau/dt * w @ np.ascontiguousarray(input[:,i,i_trial]) )
            V[i + 1] = V[i] + dV
            if V[i + 1] > Vth:
                spike_trains[i + 1, i_trial] = 1
                V[i] = 20
                V[i + 1] = Vr
        Vtot[:, i_trial] = V
    return spike_trains, Vtot






##

