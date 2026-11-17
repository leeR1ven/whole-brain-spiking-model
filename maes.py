"""A recurrent network that learns the ORDER of its inputs and then replays it by itself - a port of the public code of
Maes, Barahona & Clopath 2020 (PLoS Comput Biol 16:e1007606; ModelDB 257609, file 'Learning sequential dynamics/sim.jl',
saved in data/maes2020_code/; that code is itself an edited copy of Litwin-Kumar & Doiron 2014).

  2400 excitatory cells in 30 groups of 80 + 600 inhibitory cells; every pair connected with probability 0.2.
  excitatory cell   exponential integrate-and-fire, threshold that jumps 10 mV after a spike (decay 30 ms),
                    adaptation current (+1000 pA per spike, decay 100 ms), 5 ms refractory
  inhibitory cell   leaky integrate-and-fire
  synapses          conductances, difference of two exponentials (E: 1 / 6 ms, I: 0.5 / 2 ms); arrive one step later
  background        Poisson input to every cell (E 4.5 kHz, I 2.25 kHz)
  E -> E plasticity voltage-based (Clopath et al. 2010): depression at a sending spike if the receiver's slow voltage is
                    above -70 mV; potentiation while the receiver's voltage is above -49 mV, in proportion to the sending
                    cell's 3.5 ms spike trace; every 20 ms the summed input weight of each cell is brought back to its
                    start value; weights stay within 1.45 - 32.68
  I -> E plasticity Vogels et al. 2011 (target rate 3 Hz), weights within 48.7 - 243
  training          the 30 groups are stimulated one after the other, each for 9 ms (extra 18 kHz) every 15 ms, round and
                    round; while this goes on, the cells that are not being stimulated get a net inhibitory background.

Kept exactly: equations, numbers and the order of operations inside a step (cells are updated one after the other).
The original counts cells from 1 and tests 'cell < Ne', so its last excitatory cell is treated as an inhibitory one in
several places; this is copied too.
Mine: only existing synapses are stored (the original keeps a full matrix and skips the zeros - same arithmetic); the
state of the cells is carried on from one call to the next (the original restarts it every 2 minutes); my own random
numbers."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'pylib'))
from numba import njit
from math import exp

NE, NI, NPOP, P_CONN = 2400, 600, 30, 0.2
JEE0, JEI0, JIE, JII = 2.83, 62.87, 1.96, 20.91            # start weights: E->E, I->E, E->I (fixed), I->I (fixed)


def lists(m):
    """m[pre, post] boolean. Returns incoming lists (by receiving cell) and outgoing lists pointing into them."""
    pre, post = np.nonzero(m); o = np.argsort(post, kind='stable'); pre, post = pre[o], post[o]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=m.shape[1]))]).astype(np.int64)
    b = np.argsort(pre, kind='stable'); optr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=m.shape[0]))]).astype(np.int64)
    return ptr, pre.astype(np.int64), optr, b.astype(np.int64), post[b].astype(np.int64)


def build(seed=1):
    rng = np.random.default_rng(seed); N = NE + NI; S = {}
    m = rng.random((N, N)) < P_CONN; np.fill_diagonal(m, False)
    S['ee'] = lists(m[:NE, :NE]); S['ee_w'] = np.full(len(S['ee'][1]), JEE0)
    S['ie'] = lists(m[NE:, :NE]); S['ie_w'] = np.full(len(S['ie'][1]), JEI0)
    e_ = lists(m[:NE, NE:]); i_ = lists(m[NE:, NE:]); S['ei'] = (e_[2], e_[4]); S['ii'] = (i_[2], i_[4])     # outgoing lists only (fixed weights)
    S['sum0'] = np.array([S['ee_w'][S['ee'][0][c]:S['ee'][0][c + 1]].sum() for c in range(NE)])
    S['v'] = -60.0 + 8.0 * rng.random(N); S['vth'] = np.full(NE, -52.0); S['wad'] = np.zeros(NE); S['last'] = np.full(N, -100.0)
    S['tri'] = np.zeros(N); S['u'] = np.zeros(NE); S['vv'] = np.zeros(NE); S['x'] = np.zeros(NE)
    S['syn'] = np.zeros((4, N)); S['fprev'] = np.zeros((2, N)); S['rx'] = np.concatenate([np.full(NE, 4.5), np.full(NI, 2.25)]); S['rx'][NE - 1] = 2.25
    S['nextx'] = rng.exponential(1.0, N) / S['rx']; S['step'] = 0
    S['pop'] = np.repeat(np.arange(NPOP), NE // NPOP); S['cur'] = np.array([-1], np.int64)      # group of every E cell; group being stimulated now
    return S


@njit(cache=True)
def seed(x):
    np.random.seed(x)


@njit(cache=True)
def run(nsteps, step0, stim_on, stdp_from, v, vth, wad, last, tri, u, vv, x, syn, fprev, rx, nextx, sum0,
        ee_ptr, ee_pre, ee_optr, ee_opos, ee_opost, ee_w, ie_ptr, ie_pre, ie_optr, ie_opos, ie_opost, ie_w,
        ei_optr, ei_opost, ii_optr, ii_opost, pop, cur, cnt, rec_t, rec_i, rec_n):
    """stim_on: the groups are stimulated in turn (and the others get the net inhibitory background);
    stdp_from: plasticity is active from this time (ms) on."""
    ne = vth.shape[0]; n = v.shape[0]; npop = 30
    taue = 20.0; taui = 20.0; vleake = -70.0; vleaki = -62.0; deltathe = 2.0; C = 300.0; erev = 0.0; irev = -75.0
    vth0 = -52.0; ath = 10.0; tauth = 30.0; vre = -60.0; taurefrac = 5.0; bw = 1000.0; tauw = 100.0
    tauerise = 1.0; tauedecay = 6.0; tauirise = 0.5; tauidecay = 2.0; rex = 4.5
    jeemin = 1.45; jeemax = 32.68; jeimin = 48.7; jeimax = 243.0; jex = 1.6; jix = 1.52; jie = 1.96; jii = 20.91
    altd = 0.0014; altp = 0.0008; thetaltd = -70.0; thetaltp = -49.0; tauu = 10.0; tauv = 7.0; taux = 3.5
    tauy = 20.0; eta = 1.0; r0 = 0.003; dt = 0.1; vpeak = 20.0; inorm = 200; nrec = rec_t.shape[0]
    fE = np.zeros(n); fI = np.zeros(n); xer, xed, xir, xid = syn[0], syn[1], syn[2], syn[3]; fEp, fIp = fprev[0], fprev[1]
    for it in range(nsteps):
        tt = step0 + it + 1; t = dt * tt; tprev = dt * (tt - 1); learn = t > stdp_from
        for i in range(n): fE[i] = 0.0; fI[i] = 0.0
        g_now = -1                                                   # stimulus k (k = 1, 2, ...): group k % 30 from 1 + 15 k to 10 + 15 k ms
        if stim_on:
            k = (tt - 10) // 150
            if k >= 1 and (tt - 10) % 150 < 90: g_now = k % npop
        if g_now != cur[0]:
            for c in range(ne):
                if pop[c] == cur[0]: rx[c] -= 18.0
                if pop[c] == g_now: rx[c] += 18.0
            cur[0] = g_now
        if tt % inorm == 0:                                          # summed input weight of every E cell back to its start value
            for c in range(ne):
                a = ee_ptr[c]; b = ee_ptr[c + 1]; s_ = 0.0
                if c == ne - 1:                                      # the original's last E cell has no start sum: its inputs fall to the minimum
                    for k in range(a, b): ee_w[k] = jeemin
                    continue
                for k in range(a, b): s_ += ee_w[k]
                d = (s_ - sum0[c]) / (b - a)
                for k in range(a, b):
                    w = ee_w[k] - d
                    ee_w[k] = jeemin if w < jeemin else (jeemax if w > jeemax else w)
        for c in range(n):
            tri[c] -= dt * tri[c] / tauy
            while t > nextx[c]:                                      # background input
                nextx[c] += np.random.exponential(1.0) / rx[c]
                if c < ne - 1:
                    fEp[c] += jex
                    if stim_on and rx[c] == rex: fEp[c] -= 1.5 * jex
                else:
                    fEp[c] += jix
            xer[c] += -dt * xer[c] / tauerise + fEp[c]; xed[c] += -dt * xed[c] / tauedecay + fEp[c]
            xir[c] += -dt * xir[c] / tauirise + fIp[c]; xid[c] += -dt * xid[c] / tauidecay + fIp[c]
            exc = c < ne - 1                                         # the original tests 'cc < Ne' counting from 1: its last E cell is treated as inhibitory
            if exc:
                vth[c] += dt * (vth0 - vth[c]) / tauth; wad[c] += dt * (-wad[c]) / tauw
                u[c] += dt * (v[c] - u[c]) / tauu; vv[c] += dt * (v[c] - vv[c]) / tauv; x[c] -= dt * x[c] / taux
            ge = (xed[c] - xer[c]) / (tauedecay - tauerise); gi = (xid[c] - xir[c]) / (tauidecay - tauirise)
            spiked = False
            if t > last[c] + taurefrac:
                if exc:
                    v[c] += dt * ((vleake - v[c] + deltathe * exp((v[c] - vth[c]) / deltathe)) / taue + ge * (erev - v[c]) / C + gi * (irev - v[c]) / C - wad[c] / C)
                    if v[c] > vpeak:
                        spiked = True; wad[c] += bw
                else:
                    v[c] += dt * ((vleaki - v[c]) / taui + ge * (erev - v[c]) / C + gi * (irev - v[c]) / C)
                    if v[c] > vth0: spiked = True
                if spiked:
                    v[c] = vre; last[c] = t; cnt[c] += 1; tri[c] += 1.0
                    kk = rec_n[0]
                    if kk < nrec and c < ne:
                        rec_t[kk] = tt; rec_i[kk] = c; rec_n[0] = kk + 1
                    if exc:
                        x[c] += 1.0; vth[c] = vth0 + ath
                    if c < ne:                                       # the last E cell still sends through its E -> E / E -> I synapses
                        if exc:
                            for q in range(ee_optr[c], ee_optr[c + 1]): fE[ee_opost[q]] += ee_w[ee_opos[q]]
                            for q in range(ei_optr[c], ei_optr[c + 1]): fE[ne + ei_opost[q]] += jie
                        else:                                        # ... but, as in the original, into the inhibitory conductance
                            for q in range(ee_optr[c], ee_optr[c + 1]): fI[ee_opost[q]] += ee_w[ee_opos[q]]
                            for q in range(ei_optr[c], ei_optr[c + 1]): fI[ne + ei_opost[q]] += jie
                    else:
                        ci = c - ne
                        for q in range(ie_optr[ci], ie_optr[ci + 1]): fI[ie_opost[q]] += ie_w[ie_opos[q]]
                        for q in range(ii_optr[ci], ii_optr[ci + 1]): fI[ne + ii_opost[q]] += jii
            if spiked and learn:                                     # inhibitory plasticity
                if c < ne:
                    for k in range(ie_ptr[c], ie_ptr[c + 1]):
                        w = ie_w[k] + eta * tri[ne + ie_pre[k]]
                        ie_w[k] = jeimax if w > jeimax else w
                else:
                    ci = c - ne
                    for q in range(ie_optr[ci], ie_optr[ci + 1]):
                        k = ie_opos[q]; w = ie_w[k] + eta * (tri[ie_opost[q]] - 2.0 * r0 * tauy)
                        ie_w[k] = jeimax if w > jeimax else (jeimin if w < jeimin else w)
            if spiked and learn and exc:                             # depression: this cell sent a spike
                for q in range(ee_optr[c], ee_optr[c + 1]):
                    d = ee_opost[q]
                    if u[d] > thetaltd:
                        k = ee_opos[q]; w = ee_w[k] - dt * altd * (u[d] - thetaltd)
                        ee_w[k] = jeemin if w < jeemin else w
            if learn and exc and v[c] > thetaltp and vv[c] > thetaltd:   # potentiation of this cell's inputs
                f = dt * altp * (v[c] - thetaltp) * (vv[c] - thetaltd)
                for k in range(ee_ptr[c], ee_ptr[c + 1]):
                    d = ee_pre[k]
                    if d < ne - 1 and x[d] > 1e-12:
                        w = ee_w[k] + f * x[d]
                        ee_w[k] = jeemax if w > jeemax else w
        for i in range(n): fEp[i] = fE[i]; fIp[i] = fI[i]
    return 0


class Sim:
    def __init__(self, seed_=1):
        self.S = build(seed_); seed(seed_ + 500)

    def run(self, ms, stim_on, stdp_delay=0.0, record=False, max_spikes=4_000_000):
        """ms of network time. stdp_delay: plasticity starts this long (ms) after the start of this call (as in the
        original, where every call restarts its clock). Returns (E rate, I rate, spikes of E cells or None)."""
        S = self.S; n = int(round(ms / 0.1)); cnt = np.zeros(NE + NI, np.int64)
        rec_t = np.zeros(max_spikes if record else 1, np.int64); rec_i = np.zeros(max_spikes if record else 1, np.int32); rec_n = np.array([0 if record else 1], np.int64)
        run(n, S['step'], stim_on, S['step'] * 0.1 + stdp_delay if stdp_delay >= 0 else 1e30, S['v'], S['vth'], S['wad'], S['last'], S['tri'], S['u'], S['vv'], S['x'],
            S['syn'], S['fprev'], S['rx'], S['nextx'], S['sum0'], *S['ee'], S['ee_w'], *S['ie'], S['ie_w'], *S['ei'], *S['ii'], S['pop'], S['cur'], cnt, rec_t, rec_i, rec_n)
        S['step'] += n
        return cnt[:NE].mean() / ms * 1e3, cnt[NE:].mean() / ms * 1e3, ((rec_t[:rec_n[0]].copy(), rec_i[:rec_n[0]].copy()) if record else None)

    def structure(self):
        """mean E -> E weight inside a group, from a group to the NEXT one, to the PREVIOUS one, to all other groups"""
        S = self.S; ptr, pre = S['ee'][0], S['ee'][1]; post = np.repeat(np.arange(NE), np.diff(ptr)); a = S['pop'][pre]; b = S['pop'][post]; w = S['ee_w']
        d = (b - a) % NPOP
        return w[d == 0].mean(), w[d == 1].mean(), w[d == NPOP - 1].mean(), w[(d > 1) & (d < NPOP - 1)].mean()
