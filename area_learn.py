"""One cortical area of the brain model (the layered circuit of Potjans & Diesmann 2014, scaled as in brain.py) whose
synapses onto the layer-2/3 excitatory cells CHANGE by the rules of Zenke, Agnes & Gerstner 2015 (reproduced in zenke.py):
    layer 4 e -> layer 2/3 e   (feed-forward)       excitatory rule
    layer 2/3 e -> layer 2/3 e (the loop)           excitatory rule, part of the charge through a slow (100 ms) current
    layer 2/3 i -> layer 2/3 e                      inhibitory rule (target rate = the cells' rate in the full-size circuit)
Input: thalamic fibres spread over the area's 1 mm^2; a 'pattern' = the fibres of one quarter of the sheet firing.

Putting the two papers together is MINE, and so are: which connections change; the share of slow current; fibres wired
to cells by lateral distance (width `sig_tc`); the patterns; the fibres' rate."""
import sys, json, time, numpy as np
from sim import Net
import cortex

RF = json.load(open(__file__.rsplit('/', 1)[0] + '/data/column_rates_full.json'))


def build(seed=1, scale=0.02, kscale=0.13, learn=True, a_fast=0.5, stp=True, nth=cortex.N_TH_FULL, sig_tc=0.1, exc=None, ff=True, stp_ff=None):
    net = Net(dt=0.1, seed=seed); rng = np.random.default_rng(seed + 7); pl = None
    if learn:
        e = dict(a_fast=a_fast, stp=stp); e.update(exc or {})
        pl = {('L23e', 'L23e'): e, ('L23i', 'L23e'): {}}
        if ff: pl[('L4e', 'L23e')] = dict(e, a_fast=1.0, stp=stp if stp_ff is None else stp_ff)      # stp: short-term plasticity of the loop synapses; stp_ff: of the feed-forward ones
    n, span = cortex.add_areas_sparse(net, ['A'], {'A': (40.0, 0.0, 0.0)}, scale, kscale, RF, plastic=pl, slow={'L23e': 100.0} if learn else None)
    tp = np.zeros((nth, 3), dtype=np.float32); tp[:, 0] = 40.0; tp[:, 1:] = rng.uniform(-0.5, 0.5, (nth, 2))
    net.add('th', nth, 'poisson', pos=tp, rate=0.0)
    for p, c in cortex.TH_CONN.items():                                              # fibres -> layers 4 and 6: published number and strength
        q = net.pops['cx.' + p]; K = cortex.k_total(c, cortex.N_TH_FULL, cortex.N_FULL[p]) / cortex.N_FULL[p]
        P0 = np.exp(-((q['pos'][:, None, 1:] - tp[None, :, 1:]) ** 2).sum(2) / (2 * sig_tc ** 2))
        cnt = rng.poisson(P0 / P0.sum(1, keepdims=True) * K); jj, ii = np.nonzero(cnt); r = cnt[jj, ii]
        net.connect_pairs('th', 'cx.' + p, np.repeat(ii, r), np.repeat(jj, r), cortex.W, cortex.W * cortex.W_SD_REL, cortex.D_E, cortex.D_E_SD)
    net.finalize()
    quad = lambda pos: (pos[:, 1] > 0).astype(int) + 2 * (pos[:, 2] > 0).astype(int)
    info = dict(n=n, quad_th=quad(tp), quad={p: quad(net.pops['cx.' + p]['pos']) for p in cortex.POPS})
    return net, info


def show(net, info, pat, rate, T_ms, record=None):
    """pattern `pat` (0-3, or None for a pause) for T_ms; returns rates (and spikes)"""
    net.set_rate('th', np.where(info['quad_th'] == pat, rate, 0.0) if pat is not None else 0.0)
    return net.run(T_ms, record=record)


def by_quad(net, info, p):
    """mean count of the cells of population p in each quarter, since the last run() call"""
    c = net.pops['cx.' + p]['count']; q = info['quad'][p]
    return np.array([c[q == k].mean() for k in range(4)])


def syn(net, a, b):
    """all changing synapses from population a to population b: (sending cell, receiving cell, weight)"""
    qa, qb = net.pops['cx.' + a], net.pops['cx.' + b]; out = []
    for T in net.ptab:
        if T['Gpre'] is qa['grp'] and T['Gpost'] is qb['grp']:
            m = (T['pre'] >= qa['off']) & (T['pre'] < qa['off'] + qa['n']) & (T['post'] >= qb['off']) & (T['post'] < qb['off'] + qb['n'])
            if m.any(): out.append((T['pre'][m] - qa['off'], T['post'][m] - qb['off'], T['ws'][m, 0]))
    return [np.concatenate(x) for x in zip(*out)]


def weights(net, info, name):
    """mean weight of a plastic connection, by quarter of the sending (row) and receiving (column) cell"""
    a, b = name.split('->'); i, j, w = syn(net, a, b); qa = info['quad'][a][i]; qb = info['quad'][b][j]
    return np.array([[w[(qa == x) & (qb == y)].mean() for y in range(4)] for x in range(4)]), w
