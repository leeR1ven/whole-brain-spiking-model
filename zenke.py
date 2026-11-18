"""A recurrent network that LEARNS cell assemblies and then holds them active - a port of the published simulation of
Zenke, Agnes & Gerstner 2015 (Nat Commun 6:6922). Everything is copied from the authors' code saved in
data/zenke2015_code/ (sim_rc_p10c.cpp = the version behind the paper's Fig. 3, P10Connection.cpp, GlobalPFConnection.cpp,
and the neuron / stimulus / trace / delay classes of their simulator Auryn in data/zenke2015_code/auryn/), with the
settings of their run scripts (globalvars.sh, 1run_init.sh, 2run_learn.sh, 3run_cued.sh). Their README says these settings
differ slightly from the paper's (shorter training).

  4096 excitatory + 1024 inhibitory integrate-and-fire cells, 4096 input units (a 64 x 64 field).
  neuron      conductance input; fast (5 ms) and slow (100 ms) excitation, inhibition (10 ms); after a spike the threshold
              jumps and relaxes (5 ms); excitatory cells also have spike-triggered adaptation (100 ms).
  synapses    input -> E: each E cell sees a disc of the field (file rf_discsR8.mtx), plastic.
              E -> E (5 % connected, plastic), E -> I, I -> I (fixed), I -> E (plastic, inhibitory).
              all excitatory synapses: short-term depression / facilitation. All delays 0.8 ms.
  plasticity of excitatory synapses (the paper's equations 11-14, 16):
              + A  * (slow trace of the receiving cell) * (trace of the sending cell)        at every spike of the receiver
              - A  * (trace of the receiving cell)                                           at every arriving spike
              - beta * (trace of the receiving cell)^3 * (w - reference)                     at every spike of the receiver
              + delta                                                                        at every arriving spike
              reference weight: slowly follows w in a double-well (two stable values), time constant 20 min
  plasticity of inhibitory synapses onto E cells (equations 19-20): stronger when the whole excitatory population fires
              above its target rate (4 Hz), weaker when below.
  stimulus    one of the patterns is switched on for a random time: its units fire `scale` Hz on top of the `bg` Hz that
              all units always fire.
  hom=1       the variant sim_rc_p11 (their scripts' default): the depression term is multiplied by
              (slow rate estimate of the receiving cell / 10 Hz)^2.

Order of operations inside a step = Auryn's (System.cpp): neurons, then arriving spikes and weight changes, then
short-term states, then traces (so the current spike is not in the traces when weights change).
Mine (not from the code): input spikes are drawn as a Poisson number of events per step; 64-bit numbers instead of 32-bit;
my own random connections (the authors' random numbers are not reproduced).
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'pylib'))
from numba import njit
from math import exp

NE, NI, NS, DT, DELAY = 4096, 1024, 4096, 1e-4, 8
CODE = os.path.join(HERE, 'data/zenke2015_code')


def load_patterns(fn):
    pats = []
    for line in open(fn):
        t = line.split()
        if line.startswith('#'): pats.append([])
        elif t and pats: pats[-1].append(int(t[0]))
    return [np.array(sorted(set(p)), dtype=np.int64) for p in pats if p]


def csr(pre, post, n_pre, n_post):
    """forward table by sending cell + backward table by receiving cell (positions into the forward arrays)"""
    o = np.argsort(pre, kind='stable'); pre, post = pre[o], post[o]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n_pre))]).astype(np.int64)
    b = np.argsort(post, kind='stable')
    bptr = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=n_post))]).astype(np.int64)
    return [ptr, post.astype(np.int64), bptr, b.astype(np.int64), pre[b].astype(np.int64), pre.astype(np.int64)]


def build(seed=1, wee=0.1, wie=0.2, xi=0.5, sparse=0.05, alpha=4.0, sparse_ee=None):
    rng = np.random.default_rng(seed); S = {}
    def rand(n_pre, n_post, no_self=False, p=None):
        m = rng.random((n_pre, n_post)) < (sparse if p is None else p)
        if no_self: np.fill_diagonal(m, False)
        return np.nonzero(m)
    # sparse_ee: another connection probability for E -> E only (not the paper's)
    i, j = rand(NE, NE, True, sparse_ee); S['ee'] = csr(i, j, NE, NE); S['ee_ws'] = np.zeros((len(i), 2)); S['ee_ws'][:, 0] = wee     # columns: weight, reference weight
    d = np.loadtxt(os.path.join(CODE, 'rf_discsR8.mtx'), skiprows=3)
    i, j = d[:, 0].astype(np.int64) - 1, d[:, 1].astype(np.int64) - 1
    S['se'] = csr(i, j, NS, NE); S['se_ws'] = np.zeros((len(i), 2)); S['se_ws'][:, 0] = xi
    i, j = rand(NI, NE); S['ie'] = csr(i, j, NI, NE); S['ie_w'] = np.full(len(i), wie)
    i, j = rand(NE, NI); S['ei'] = csr(i, j, NE, NI)
    i, j = rand(NI, NI, True); S['ii'] = csr(i, j, NI, NI)
    z = lambda n, v=0.0: np.full(n, v)
    S['E'] = [z(NE, -70e-3), z(NE), z(NE), z(NE), z(NE), z(NE), z(NE)]           # mem, thr, g_ampa, g_gaba, g_nmda, g_adapt, slow g_adapt
    S['I'] = [z(NI, -70e-3), z(NI), z(NI), z(NI), z(NI)]
    S['tr'] = [z(NE), z(NE), z(NE), z(NS), z(NI), z(NE, 1200.0)]                  # E as sender, E receiver 20 ms, 100 ms, input, I, slow rate estimate
    S['stp'] = [z(NE, 1.0), z(NE, 0.2), z(NE, 1.0), z(NE, 0.2), z(NS, 1.0), z(NS, 0.2),   # x, u of E->E, of E->I, of input->E (as of the unit's last spike)
                np.zeros(NE, np.int64), np.zeros(NS, np.int64)]                           # step of the last spike of each E cell / input unit
    S['dl'] = [np.zeros((DELAY, NE), np.int64), np.zeros((DELAY, NE)), np.zeros((DELAY, NE)), np.zeros(DELAY, np.int64),
               np.zeros((DELAY, NI), np.int64), np.zeros(DELAY, np.int64),
               np.zeros((DELAY, NS), np.int64), np.zeros((DELAY, NS)), np.zeros(DELAY, np.int64)]
    S['sched'] = np.array([0.0, 0.0, 1.0 / DT, 0.0]); S['pf'] = np.array([alpha * NE]); S['step'] = 0
    return S


def params(eta=1e-3, beta=0.05, delta=0.02, alpha=4.0, taud=0.15, tauf=0.6, U=0.2, bg=5.0, scale=25.0, on=1.0, off=2.0, wei=0.2, wii=0.2,
           wmax=5.0, wmaxi=5.0, wa=0.0, wc=0.5, pot=0.1, hom=0.0, tauh=600.0, kappa=10.0, plastic=1.0, seq=0.0, tau_pre=0.02, adapt2=0.0, ee_on=1.0, wmax2=5.0, ne=NE):
    wt = (wc - wa) / 4.0; norm = (wt - wa) * (wt - (wa + wc) / 2) * (wt - wc)
    return np.array([eta, beta, delta * (eta / 1e-3) * eta, wmax, 3 * wei, wii, wmaxi, (eta / 50) / (alpha * ne), alpha * ne, taud, tauf, U, bg, scale, on, off,
                     pot / norm, wa, (wa + wc) / 2, wc, hom, tauh, kappa, plastic, seq, tau_pre, adapt2, ee_on, wmax2])


@njit(cache=True)
def seed(x):
    np.random.seed(x)


@njit(cache=True)
def run(nsteps, step0, P, memE, thrE, gaE, ggE, gnE, gadE, gad2E, memI, thrI, gaI, ggI, gnI, trpre, trpost, trslow, trS, trI, hom,
        xE, uE, xEI, uEI, xS, uS, lastE, lastS,
        ee_ptr, ee_post, ee_bptr, ee_bpos, ee_bpre, ee_pre, ee_ws, ee_pl, se_ptr, se_post, se_bptr, se_bpos, se_bpre, se_pre, se_ws,
        ie_ptr, ie_post, ie_bptr, ie_bpos, ie_bpre, ie_pre, ie_w, ei_ptr, ei_post, ii_ptr, ii_post,
        dE, dEa, dEa2, dEn, dI, dIn, dS, dSa, dSn, sched, pf, pat_ptr, pat_idx, cntE, cntI, rec_t, rec_i, rec_n, ev):
    eta, beta, delta, wmax, w_ei, w_ii, wmaxi, lr, expected = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]
    taud, tauf, U, bg, scale, mon, moff, pot, wa, wb, wc = P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17], P[18], P[19]
    use_hom = P[20] > 0; tauh = P[21]; kappa = P[22]; plastic = P[23] > 0; seq = P[24] > 0     # seq: patterns in fixed order 0, 1, 2, ... (not used by the paper)
    dt = 1e-4; e_rest = -70e-3; e_rev = -80e-3; thr_rest = -50e-3; dthr = 100e-3
    s_ampa = exp(-dt / 5e-3); s_gaba = exp(-dt / 10e-3); s_thr = exp(-dt / 5e-3); s_ad = exp(-dt / 0.1); m_nmda = dt / 0.1; m_mem = dt / 20e-3
    AeA = 0.2 / 1.2; AeN = 1.0 / 1.2; AiA = 0.3 / 1.3; AiN = 1.0 / 1.3
    d20 = exp(-dt / 20e-3); d100 = exp(-dt / 100e-3); pfm = exp(-dt / 10.0); dhom = exp(-dt / tauh)
    # tau_pre (not the paper's unless 0.02): time window of the SENDING cell's trace in the potentiation of E -> E synapses;
    # the potentiation is scaled by 0.02 / tau_pre, so that steady joint firing changes a weight exactly as before
    dpre = exp(-dt / P[25]); eta_ee = eta * 0.02 / P[25]
    # adapt2: the second, long-lasting adaptation of the excitatory cells (paper: Methods, used in its Fig. 4c; the simulator's
    # neuron class AIF2Group: decay 20 s, default jump 0.002; the paper's run scripts leave it at 0)
    dg2 = P[26]; s_ad2 = exp(-dt / 20.0)
    wmax2 = P[28]
    ee_on = P[27] > 0                                                # 0: the E -> E synapses are switched off (no loop among the excitatory cells) - a control, not the paper
    ne = memE.shape[0]; ni = memI.shape[0]; ns = xS.shape[0]; npat = pat_ptr.shape[0] - 1; nd = dE.shape[0]
    spE = np.zeros(ne, np.int64); spI = np.zeros(ni, np.int64); nrec = rec_t.shape[0]; nev = ev.shape[0]
    for it in range(nsteps):
        step = step0 + it; slot = step % nd
        # ---- stimulus schedule (StimulusGroup: off time exponential, on time normal with sd = mean / 3)
        if step >= sched[2]:
            if sched[0] > 0:
                sched[0] = 0.0; sched[3] = step; sched[2] = step + int(np.random.exponential(moff) / dt)
            else:
                sched[0] = 1.0; sched[1] = float((int(sched[1]) + 1) % npat) if seq else float(np.random.randint(npat)); sched[2] = step + int(max(0.0, np.random.normal(mon, mon / 3.0)) / dt)
            k = int(ev[0, 0]) + 1
            if k < nev:
                ev[k, 0] = step; ev[k, 1] = sched[0]; ev[k, 2] = sched[1]; ev[0, 0] = k
        active = sched[0] > 0; cur = int(sched[1])
        # ---- excitatory cells (AIF2Group)
        for i in range(ne): gaE[i] *= s_ampa
        for i in range(ne): ggE[i] *= s_gaba
        for i in range(ne): gadE[i] *= s_ad
        for i in range(ne): gad2E[i] *= s_ad2
        for i in range(ne): thrE[i] *= s_thr
        for i in range(ne): gnE[i] = (gnE[i] + m_nmda * gaE[i]) * (1.0 - m_nmda)
        for i in range(ne):
            v = memE[i]
            v += m_mem * (-(AeA * gaE[i] + AeN * gnE[i]) * v - (v - e_rev) * (ggE[i] + gadE[i] + gad2E[i]) - (v - e_rest))
            memE[i] = min(max(v, e_rev), 0.0)
        nE = 0
        for i in range(ne):
            if memE[i] > thr_rest + thrE[i]:
                memE[i] = e_rest; thrE[i] = dthr; gadE[i] += 0.1; gad2E[i] += dg2; spE[nE] = i; nE += 1
        # ---- inhibitory cells (IFGroup: currents from the values before their decay)
        for i in range(ni): thrI[i] *= s_thr
        for i in range(ni):
            v = memI[i]
            v += m_mem * (-(AiA * gaI[i] + AiN * gnI[i]) * v - (v - e_rev) * ggI[i] - (v - e_rest))
            memI[i] = min(max(v, e_rev), 0.0)
        for i in range(ni): gnI[i] = (gnI[i] + m_nmda * gaI[i]) * (1.0 - m_nmda)
        for i in range(ni): gaI[i] *= s_ampa
        for i in range(ni): ggI[i] *= s_gaba
        nI = 0
        for i in range(ni):
            if memI[i] > thr_rest + thrI[i]:
                memI[i] = e_rest; thrI[i] = dthr; spI[nI] = i; nI += 1
        # ---- records
        for a in range(nE):
            cntE[spE[a]] += 1
            k = rec_n[0]
            if k < nrec:
                rec_t[k] = step; rec_i[k] = spE[a]; rec_n[0] = k + 1
        for a in range(nI): cntI[spI[a]] += 1
        # ---- arrivals: spikes sent DELAY steps ago (this slot of the buffers), with their short-term state
        gm = lr * (pf[0] - expected)
        nSd = dSn[slot]; nEd = dEn[slot]; nId = dIn[slot]
        for a in range(nSd):                                               # input -> E
            j = dS[slot, a]; at = dSa[slot, a]
            for k in range(se_ptr[j], se_ptr[j + 1]):
                i = se_post[k]; gaE[i] += se_ws[k, 0] * at
                if plastic:
                    if use_hom:
                        r = hom[i] / tauh / kappa; w = se_ws[k, 0] - eta * trpost[i] * r * r + delta
                    else:
                        w = se_ws[k, 0] - eta * trpost[i] + delta
                    se_ws[k, 0] = w if w > 0.0 else 0.0
        for a in range(nEd):                                               # E -> E (plastic), E -> I
            j = dE[slot, a]; at = dEa[slot, a]
            for k in range(ee_ptr[j], ee_ptr[j + 1] if ee_on else ee_ptr[j]):
                i = ee_post[k]; gaE[i] += ee_ws[k, 0] * at
                if plastic and ee_pl[k]:                       # ee_pl[k] = 0: this E -> E synapse is kept fixed (not in the paper)
                    if use_hom:
                        r = hom[i] / tauh / kappa; w = ee_ws[k, 0] - eta * trpost[i] * r * r + delta
                    else:
                        w = ee_ws[k, 0] - eta * trpost[i] + delta
                    if ee_pl[k] == 2 and w > wmax2: w = wmax2
                    ee_ws[k, 0] = w if w > 0.0 else 0.0
            at = dEa2[slot, a]
            for k in range(ei_ptr[j], ei_ptr[j + 1]):
                gaI[ei_post[k]] += w_ei * at
        for a in range(nId):                                               # I -> E (plastic), I -> I
            j = dI[slot, a]
            for k in range(ie_ptr[j], ie_ptr[j + 1]):
                i = ie_post[k]; ggE[i] += ie_w[k]
                if plastic:
                    w = ie_w[k] + gm * (trpost[i] + 1.0)
                    ie_w[k] = w if w > 0.0 else 0.0
            for k in range(ii_ptr[j], ii_ptr[j + 1]):
                ggI[ii_post[k]] += w_ii
        # ---- spikes of receiving E cells: potentiation, heterosynaptic term, inhibitory rule
        if plastic:
            for a in range(nE):
                i = spE[a]; zs = trslow[i]; z3 = beta * trpost[i] ** 3
                for q in range(ee_bptr[i], ee_bptr[i + 1] if ee_on else ee_bptr[i]):
                    k = ee_bpos[q]
                    if ee_pl[k] == 0: continue
                    w = ee_ws[k, 0] + eta_ee * zs * trpre[ee_bpre[q]] - z3 * (ee_ws[k, 0] - ee_ws[k, 1])
                    cap = wmax if ee_pl[k] == 1 else wmax2     # ee_pl = 2: synapses with their own, lower ceiling (returns from a higher level; not in the paper)
                    ee_ws[k, 0] = cap if w > cap else (0.0 if w < 0.0 else w)
                for q in range(se_bptr[i], se_bptr[i + 1]):
                    k = se_bpos[q]; w = se_ws[k, 0] + eta * zs * trS[se_bpre[q]] - z3 * (se_ws[k, 0] - se_ws[k, 1])
                    se_ws[k, 0] = wmax if w > wmax else (0.0 if w < 0.0 else w)
                for q in range(ie_bptr[i], ie_bptr[i + 1]):
                    k = ie_bpos[q]; w = ie_w[k] + gm * trI[ie_bpre[q]]
                    ie_w[k] = wmaxi if w > wmaxi else w
        # ---- traces take the spikes (sender traces: the ones that just arrived; receiver traces: the ones just fired)
        for a in range(nSd): trS[dS[slot, a]] += 1.0
        for a in range(nEd): trpre[dE[slot, a]] += 1.0
        for a in range(nId): trI[dI[slot, a]] += 1.0
        for a in range(nE):
            i = spE[a]; trpost[i] += 1.0; trslow[i] += 1.0; hom[i] += 1.0
        # ---- send: this step's spikes (with x*u) go into the slot just read; they arrive DELAY steps later
        for a in range(nE):
            i = spE[a]; dE[slot, a] = i; n_ = step - lastE[i]; lastE[i] = step
            fx = (1.0 - dt / taud) ** n_; fu = (1.0 - dt / tauf) ** n_; fu2 = (1.0 - dt / 0.6) ** n_
            x = 1.0 - (1.0 - xE[i]) * fx; u = U + (uE[i] - U) * fu
            dEa[slot, a] = x * u; uE[i] = u + U * (1.0 - u); xE[i] = x - u * x
            x = 1.0 - (1.0 - xEI[i]) * fx; u = 0.2 + (uEI[i] - 0.2) * fu2
            dEa2[slot, a] = x * u; uEI[i] = u + 0.2 * (1.0 - u); xEI[i] = x - u * x
        dEn[slot] = nE
        for a in range(nI): dI[slot, a] = spI[a]
        dIn[slot] = nI
        k_ = np.random.poisson(ns * bg * dt); m = 0
        for a in range(k_):
            if m < ns:
                dS[slot, m] = np.random.randint(ns); m += 1
        if active:
            p0 = pat_ptr[cur]; pn = pat_ptr[cur + 1] - p0
            k_ = np.random.poisson(pn * scale * dt)
            for a in range(k_):
                if m < ns:
                    dS[slot, m] = pat_idx[p0 + np.random.randint(pn)]; m += 1
        for a in range(m):
            i = dS[slot, a]; n_ = step - lastS[i]; lastS[i] = step
            x = 1.0 - (1.0 - xS[i]) * (1.0 - dt / taud) ** n_; u = U + (uS[i] - U) * (1.0 - dt / tauf) ** n_
            dSa[slot, a] = x * u; uS[i] = u + U * (1.0 - u); xS[i] = x - u * x
        dSn[slot] = m
        # ---- short-term states relax, global factor, reference weights, trace decay
        for i in range(ne): trpre[i] *= dpre
        for i in range(ne): trpost[i] *= d20
        for i in range(ne): trslow[i] *= d100
        if use_hom:
            for i in range(ne): hom[i] *= dhom
        for i in range(ns): trS[i] *= d20
        for i in range(ni): trI[i] *= d20
        pf[0] = pf[0] * pfm + nEd / 10.0
        if plastic and step % 12000 == 0:
            for k in range(ee_ws.shape[0]):
                if ee_pl[k] == 0: continue
                s_ = ee_ws[k, 1]; ee_ws[k, 1] = s_ + ((ee_ws[k, 0] - s_) + pot * (wa - s_) * (wb - s_) * (wc - s_)) * 1e-3
            for k in range(se_ws.shape[0]):
                s_ = se_ws[k, 1]; se_ws[k, 1] = s_ + ((se_ws[k, 0] - s_) + pot * (wa - s_) * (wb - s_) * (wc - s_)) * 1e-3
    return 0


class Sim:
    """the network + helpers to run it in pieces, record spikes and the stimulus log"""
    def __init__(self, seed_=1, **kw):
        self.S = build(seed=seed_, **kw); seed(seed_ + 1000)
        self.cntE = np.zeros(NE, np.int64); self.cntI = np.zeros(NI, np.int64)
        self.none_t = np.zeros(1, np.int64); self.none_i = np.zeros(1, np.int32)

    def run(self, seconds, P, pats, record=False, max_spikes=4_000_000):
        """returns (E rate, I rate, stimulus log [step, on/off, pattern], spikes (step, cell) or None)"""
        S = self.S; n = int(round(seconds / DT))
        if len(S['E']) == 6: S['E'].append(np.zeros(NE))                      # a state saved before the slow adaptation existed
        if 'ee_pl' not in S: S['ee_pl'] = np.ones(len(S['ee_ws']), np.int8)
        pat_ptr = np.concatenate([[0], np.cumsum([len(p) for p in pats])]).astype(np.int64); pat_idx = np.concatenate(pats)
        rec_t = np.zeros(max_spikes, np.int64) if record else self.none_t
        rec_i = np.zeros(max_spikes, np.int32) if record else self.none_i
        rec_n = np.array([0 if record else 1], np.int64); ev = np.zeros((max(16, int(seconds * 40)), 3))
        self.cntE[:] = 0; self.cntI[:] = 0
        run(n, S['step'], P, *S['E'], *S['I'], *S['tr'], *S['stp'], *S['ee'], S['ee_ws'], S['ee_pl'], *S['se'], S['se_ws'],
            *S['ie'], S['ie_w'], S['ei'][0], S['ei'][1], S['ii'][0], S['ii'][1], *S['dl'], S['sched'], S['pf'], pat_ptr, pat_idx,
            self.cntE, self.cntI, rec_t, rec_i, rec_n, ev)
        S['step'] += n; k = int(ev[0, 0])
        sp = (rec_t[:rec_n[0]].copy(), rec_i[:rec_n[0]].copy()) if record else None
        return self.cntE.mean() / seconds, self.cntI.mean() / seconds, ev[1:k + 1].copy(), sp

    def save(self, fn):
        S = self.S
        np.savez_compressed(fn, ee_ws=S['ee_ws'], se_ws=S['se_ws'], ie_w=S['ie_w'],
                            ee_pre=S['ee'][5], ee_post=S['ee'][1], se_pre=S['se'][5], se_post=S['se'][1], ie_pre=S['ie'][5], ie_post=S['ie'][1],
                            ei_pre=S['ei'][5], ei_post=S['ei'][1], ii_pre=S['ii'][5], ii_post=S['ii'][1], step=S['step'])
