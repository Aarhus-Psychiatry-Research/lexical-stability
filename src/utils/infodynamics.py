"""
Class for estimation of information dynamics of time-dependent probabilistic document representations. 
Author:
    Kristoffer Laigaard Nielbo
"""
import numpy as np

import numpy as np
from scipy import stats


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))


def jsd(p, q, base=np.e):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions  
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. +  stats.entropy(q, m, base=base)/2.


class InfoDynamics:
    def __init__(self, data, time, window=3, weight=0, sort=False):
        """
        - data: list/array (of lists), bow representation of documents
        - time: list/array, time coordinate for each document (identical order as data)
        - window: int, window to compute novelty, transience, and resonance over
        - weight: int, parameter to set initial window for novelty and final window for transience
        - sort: bool, if time should be sorted in ascending order and data accordingly
        """
        self.window = window
        self.weight = weight
        if sort:
            self.data = np.array([text for _, text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time
        self.m = self.data.shape[0]

    def novelty(self, meas=kld):
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[
                (i - self.window) : i,
            ]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window]) + self.weight

            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)

        self.nsignal = N_hat
        self.nsigma = N_sd

    def transience(self, meas=kld):
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[
                i + 1 : (i + self.window + 1),
            ]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window])

            T_hat[i] = np.mean(tmp)
            T_hat[-self.window :] = np.zeros([self.window]) + self.weight
            T_sd[i] = np.std(tmp)

        self.tsignal = T_hat
        self.tsigma = T_sd

    def resonance(self, meas=kld):
        self.novelty(meas)
        self.transience(meas)
        self.rsignal = self.nsignal - self.tsignal
        self.rsignal[: self.window] = np.zeros([self.window]) + self.weight
        self.rsignal[-self.window :] = np.zeros([self.window]) + self.weight
        self.rsigma = (self.nsigma + self.tsigma) / 2
        self.rsigma[: self.window] = np.zeros([self.window]) + self.weight
        self.rsigma[-self.window :] = np.zeros([self.window]) + self.weight

