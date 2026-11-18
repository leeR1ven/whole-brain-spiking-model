"""The same network WITHOUT spikes: every cell is one number, its firing rate (2026-11-18, a comparison asked for by the
designer: 'would it be better not to use spiking neurons?').
Wiring, weights, inhibitory cells: taken unchanged from a levels.py network (so a trained spiking state can be loaded).
What replaces the spiking cell is its measured input-output curve (fitted on 131,072 cell x presentation samples of the
spiking network, untrained and trained; correlation of predicted and real rate 0.96):
    excitatory cell   rate = [0.505 X - 0.072 Y - 36.5]+ / (1 + 0.010 Y)
    inhibitory cell   rate = [0.219 X - 0.050 Y - 22.0]+
    X = sum over excitatory synapses of weight x phi(rate of the sending cell),  Y = same over inhibitory synapses (plain rate)
    phi(r) = r u x, the steady state of the paper's short-term plasticity (U 0.2, 0.15 s, 0.6 s)
Mine: the time constants with which a rate follows its input (50 ms excitatory, 10 ms inhibitory)."""
import numpy as np
from numba import njit

FE = (0.5052, 0.0722, 36.53, 0.010); FI = (0.2194, 0.0501, 21.98)


@njit(cache=True)
def phi(r):
    u = 0.2 * (1.0 + 0.6 * r) / (1.0 + 0.12 * r); return r * u / (1.0 + u * r * 0.15)


@njit(cache=True)
def run(nsteps, dt, tauE, tauI, rS, rE, rI, se_bptr, se_bpos, se_bpre, se_w, ee_bptr, ee_bpos, ee_bpre, ee_w, ie_bptr, ie_bpos, ie_bpre, ie_w,
        ei_bptr, ei_bpre, ii_bptr, ii_bpre, fe, fi, ee_on, accE, accI, trace_sets, trace_ptr, trace_out, every):
    """nsteps of length dt; rates relax to the value given by the input-output curve. accE / accI: sums of the rates
    (divide by nsteps for the mean). trace_*: mean rate of given cell sets every `every` steps (for time courses)."""
    ne = rE.shape[0]; ni = rI.shape[0]; XS = np.zeros(ne); pE = np.zeros(ne); nset = trace_ptr.shape[0] - 1
    for i in range(ne):
        s = 0.0
        for q in range(se_bptr[i], se_bptr[i + 1]): s += se_w[se_bpos[q]] * phi(rS[se_bpre[q]])
        XS[i] = s
    for it in range(nsteps):
        for i in range(ne): pE[i] = phi(rE[i])
        nE2 = rE.copy(); nI2 = rI.copy()
        for i in range(ne):
            x = XS[i]
            if ee_on:
                for q in range(ee_bptr[i], ee_bptr[i + 1]): x += ee_w[ee_bpos[q]] * pE[ee_bpre[q]]
            y = 0.0
            for q in range(ie_bptr[i], ie_bptr[i + 1]): y += ie_w[ie_bpos[q]] * rI[ie_bpre[q]]
            t = fe[0] * x - fe[1] * y - fe[2]; t = t / (1.0 + fe[3] * y) if t > 0.0 else 0.0
            nE2[i] = rE[i] + dt / tauE * (t - rE[i])
        for i in range(ni):
            x = 0.0
            for q in range(ei_bptr[i], ei_bptr[i + 1]): x += pE[ei_bpre[q]]
            y = 0.0
            for q in range(ii_bptr[i], ii_bptr[i + 1]): y += rI[ii_bpre[q]]
            t = fi[0] * x - fi[1] * y - fi[2]; t = t if t > 0.0 else 0.0
            nI2[i] = rI[i] + dt / tauI * (t - rI[i])
        rE[:] = nE2; rI[:] = nI2
        for i in range(ne): accE[i] += rE[i]
        for i in range(ni): accI[i] += rI[i]
        if nset > 0 and (it + 1) % every == 0:
            for a in range(nset):
                s = 0.0
                for q in range(trace_ptr[a], trace_ptr[a + 1]): s += rE[trace_sets[q]]
                trace_out[(it + 1) // every - 1, a] = s / max(1, trace_ptr[a + 1] - trace_ptr[a])
    return 0


class RateNet:
    """rate version of a levels.Net: same S dictionary (wiring and weights)"""
    def __init__(self, S, bg=1.0, scale=25.0, dt=0.002, tauE=0.05, tauI=0.01):
        self.S = S; self.ne = len(S['lev']); self.ni = S['ii'][0].shape[0] - 1; self.ns = S['se'][0].shape[0] - 1
        self.bg, self.scale, self.dt, self.tauE, self.tauI = bg, scale, dt, tauE, tauI; self.rE = np.zeros(self.ne); self.rI = np.zeros(self.ni); self.ee_on = True

    def reset(self):
        self.rE[:] = 0.0; self.rI[:] = 0.0

    def show(self, units, seconds, sets=None, every=50):
        """input units `units` fire bg + scale Hz, the others bg Hz, for `seconds`; returns the mean rate of every excitatory cell
        (and, if cell sets are given, their mean rates every `every` steps)"""
        S = self.S; rS = np.full(self.ns, float(self.bg))
        if units is not None and len(units): rS[np.asarray(units)] += self.scale
        n = int(round(seconds / self.dt)); accE = np.zeros(self.ne); accI = np.zeros(self.ni)
        if sets is None: tp = np.zeros(1, np.int64); ts = np.zeros(1, np.int64); out = np.zeros((1, 1))
        else: tp = np.concatenate([[0], np.cumsum([len(x) for x in sets])]).astype(np.int64); ts = np.concatenate(sets).astype(np.int64); out = np.zeros((n // every, len(sets)))
        se, ee, ie, ei, ii = S['se'], S['ee'], S['ie'], S['ei'], S['ii']
        run(n, self.dt, self.tauE, self.tauI, rS, self.rE, self.rI, se[2], se[3], se[4], S['se_ws'][:, 0].copy(), ee[2], ee[3], ee[4], S['ee_ws'][:, 0].copy(), ie[2], ie[3], ie[4], S['ie_w'],
            ei[2], ei[4], ii[2], ii[4], np.array(FE), np.array(FI), self.ee_on, accE, accI, ts, tp, out, every)
        self.meanI = accI / n
        return (accE / n, out) if sets is not None else accE / n
