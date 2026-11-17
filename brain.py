"""A small whole brain (left hemisphere), to scale, in real space, made only of neurons and synapses - version 1.

Every neuron has its own place (mm, coordinates of the human AAL3 atlas, see space.py). Wiring and delays follow rules:
inside a piece of cortex the chance of a synapse falls with distance; between parts the delay is distance / 3.5 m/s
(Schmidt et al. 2018, Table 3).
Proportions (Herculano-Houzel 2009, human): cerebral cortex 19 % of all neurons, cerebellum 80 %, the rest about 1 %.

  cortex        13 areas, each a patch of the layered circuit of Potjans & Diesmann 2014 (check_column.py reproduces that
                paper at full size). Areas and links: Tomasello et al. 2018 (hearing, speech movement, seeing, hand movement,
                three levels each, the four top areas interlinked) + entorhinal cortex (gateway to the hippocampus).
  cerebellum    Casali et al. 2019 - runs, but Purkinje / molecular-layer cells fire faster than in that paper.
  basal ganglia Lindahl & Hellgren Kotaleski 2016 - projection neurons in the paper's range, interneurons / output not yet.
  thalamus, hippocampus: cell-number ratios, places and the wiring DIAGRAM are from papers; all numbers in PROVISIONAL are
                mine. The hippocampal fields use the layer-2/3 circuit of the cortex as a stand-in.
NOT yet in the model (their papers are not read yet): inferior olive, pontine nuclei, dopamine cells, amygdala,
hypothalamus, brainstem, spinal cord, right hemisphere; synapses do not change with experience yet.
"""
import json, os, numpy as np
from sim import Net
import cortex, cerebellum, bg, space

HERE = os.path.dirname(os.path.abspath(__file__))
AREAS = ['A1', 'AB', 'PB', 'M1i', 'PMi', 'PFi', 'V1', 'TO', 'AT', 'M1L', 'PML', 'PFL', 'EC']
LEVEL = dict(A1=1, AB=2, PB=3, M1i=1, PMi=2, PFi=3, V1=1, TO=2, AT=3, M1L=1, PML=2, PFL=3, EC=4)
LINKS = [('A1', 'AB'), ('AB', 'PB'), ('A1', 'PB'), ('M1i', 'PMi'), ('PMi', 'PFi'), ('M1i', 'PFi'),
         ('V1', 'TO'), ('TO', 'AT'), ('V1', 'AT'), ('M1L', 'PML'), ('PML', 'PFL'), ('M1L', 'PFL'),
         ('PB', 'PFi'), ('PB', 'AT'), ('PB', 'PFL'), ('PFi', 'AT'), ('PFi', 'PFL'), ('AT', 'PFL'),
         ('PB', 'EC'), ('AT', 'EC'), ('PFi', 'EC'), ('PFL', 'EC')]
COL_SCALE, K_SCALE = 0.02, 0.13  # each area: 2 % of the cells of 1 mm^2 of cortex, spread over that 1 mm^2; 13 % of the synapses per cell
ETA = 0.68                       # excitation grows along the hierarchy by 1 + ETA * h (Chaudhuri et al. 2015), h = (level - 1) / 3
SPEED = (3.5, 0.5)               # mm per ms; relative spread of the travel time (Schmidt et al. 2018)
TH_TARGET = dict(MGB=['A1'], LGN=['V1'], VL=['M1i', 'M1L', 'PMi', 'PML'], MD=['PFi', 'PFL'])
PROVISIONAL = dict(              # numbers that were NOT read in a paper
    K_between=40,                # inputs per cell for every link between two areas (each direction)
    n_tc=40, n_rt=88,            # relay cells per thalamic nucleus; reticular cells (ratio 4909 : 8952 is Iavarone et al. 2023)
    K_tc_rt=20, K_rt_tc=20, K_l6_th=30, K_l5_md=50, K_sn_tc=10, K_dcn_tc=5, w_driver=2.0, w_modul=0.5,
    K_organ=3, w_organ=50.0,      # a sense fibre is a strong 'driver': 3 fibres per relay cell, each about 7.5 mV (strength not read)
    hip_share=0.05,              # hippocampal cells / cortical cells
    K_hip=50,                    # inputs per cell along entorhinal -> dentate / CA3 / CA1, dentate -> CA3, CA3 -> CA1, CA1 -> entorhinal
    K_ctx_str=20)                # frontal layer-5 inputs per striatal / subthalamic cell


LEARN = {('L4e', 'L23e'): dict(stp=False, a_fast=1.0, hom=True), ('L23e', 'L23e'): dict(stp=False, a_fast=0.5, hom=True), ('L23i', 'L23e'): {}}


def build(seed=1, learn=False):
    P = PROVISIONAL; rf = json.load(open(os.path.join(HERE, 'data/column_rates_full.json')))
    net = Net(dt=0.1, seed=seed); rng0 = net.rng; W, G = cortex.W, cortex.G; C = space.CENTER; info = {}
    # ---------------------------------------------------------------- cortex
    f = 1 / np.sqrt(K_SCALE); ts = cortex.NEURON['tau_s']
    gain = {a: 1 + ETA * (LEVEL[a] - 1) / 3.0 for a in AREAS}
    pl = LEARN if learn else None                                                     # synapses onto layer-2/3 excitatory cells that change (see area_learn.py)
    n, span = cortex.add_areas_sparse(net, AREAS, {a: C[a] for a in AREAS}, COL_SCALE, K_SCALE, rf, exc_gain=gain, plastic=pl, slow={'L23e': 100.0} if learn else None)
    info.update(cx_n=n, span=span, gain=gain)

    def comp(dp, r0, r1, K, w, rate):    # a constant current that cancels the MEAN of an added input at rest
        net.pops['cx.' + dp]['p']['I_dc'][r0:r1] -= 1e-3 * ts * K * w * rate

    def link(src, sp, dst, dps, K):
        for dp in dps:
            w = W * f * gain[dst]; r0, r1 = span(dst, dp)
            net.connect('cx.' + sp, 'cx.' + dp, K, w, w * 0.1, cortex.D_E, cortex.D_E_SD, mode='indeg', pre_rng=span(src, sp), post_rng=(r0, r1), speed=SPEED)
            comp(dp, r0, r1, K, w, rf[sp])
    for lo, hi in LINKS:
        if LEVEL[lo] > LEVEL[hi]: lo, hi = hi, lo
        if LEVEL[lo] < LEVEL[hi]:
            link(lo, 'L23e', hi, ['L4e', 'L4i'], P['K_between'])                              # forward: upper layers -> layer 4
            link(hi, 'L5e', lo, ['L23e', 'L23i', 'L5e', 'L5i'], P['K_between'])               # backward: deep layers -> not layer 4
        else:
            for x, y in ((lo, hi), (hi, lo)):
                link(x, 'L23e', y, ['L4e', 'L4i'], P['K_between'] / 2); link(x, 'L5e', y, ['L23e', 'L23i', 'L5e', 'L5i'], P['K_between'] / 2)
    # ---------------------------------------------------------------- thalamus: 4 relay nuclei + the reticular sheet
    NUC = list(TH_TARGET); nt = P['n_tc']; tspan = lambda k: (NUC.index(k) * nt, (NUC.index(k) + 1) * nt)
    net.add('th.TC', nt * 4, 'lif_cur', pos=np.concatenate([space.blob(C[k], nt, rng0) for k in NUC]), **cortex.NEURON)
    net.add('th.RT', P['n_rt'], 'lif_cur', pos=space.blob(C['RT'], P['n_rt'], rng0), **cortex.NEURON)
    net.connect('th.TC', 'th.RT', P['K_tc_rt'], W, 0, 1.0, 0, mode='indeg', speed=SPEED)
    net.connect('th.RT', 'th.TC', P['K_rt_tc'], G * W, 0, 1.0, 0, mode='indeg', speed=SPEED)
    for k, areas in TH_TARGET.items():
        for a in areas:
            for dp, c in cortex.TH_CONN.items():                                              # relay cells -> layers 4 and 6 (Potjans & Diesmann)
                K = cortex.k_total(c, cortex.N_TH_FULL, cortex.N_FULL[dp]) / cortex.N_FULL[dp]            # an outside input: full number, published strength
                net.connect('th.TC', 'cx.' + dp, K, W, W * 0.1, cortex.D_E, cortex.D_E_SD, mode='indeg', pre_rng=tspan(k), post_rng=span(a, dp), speed=SPEED)
            net.connect('cx.L6e', 'th.TC', P['K_l6_th'] / len(areas), W * P['w_modul'], 0, 1.5, 0, mode='indeg', pre_rng=span(a, 'L6e'), post_rng=tspan(k), speed=SPEED)
            net.connect('cx.L6e', 'th.RT', P['K_l6_th'] / 8.0, W * P['w_modul'], 0, 1.5, 0, mode='indeg', pre_rng=span(a, 'L6e'), speed=SPEED)
    for a in TH_TARGET['MD']:                                                                 # layer 5 drives the higher-order nucleus
        net.connect('cx.L5e', 'th.TC', P['K_l5_md'] / 2.0, W * P['w_driver'], 0, 1.5, 0, mode='indeg', pre_rng=span(a, 'L5e'), post_rng=tspan('MD'), speed=SPEED)
    info.update(tspan=tspan, NUC=NUC)
    # ---------------------------------------------------------------- hippocampus (stand-in circuit, see header)
    ncx = sum(n.values()) * len(AREAS); nh = P['hip_share'] * ncx / (1 + P['hip_share'])
    ratio = dict(DG=1000000.0, CA3=204700.0 + 25300.0, CA1=311500.0 + 38500.0); tot = sum(ratio.values())        # rat: Rolls 2013; Bezaire & Soltesz 2013
    Kin = cortex.indegree_full(); hn = {}; hbg = {}
    for q_, k in enumerate(('DG', 'CA3', 'CA1')):
        m = int(round(nh * ratio[k] / tot)); ni = max(8, int(round(m * 0.11))); ne = m - ni; hn[k + 'e'], hn[k + 'i'] = ne, ni
        cen = C['hippocampus'] + np.array([0.0, 2.0 * q_, 0.0])
        for t, m_ in (('e', ne), ('i', ni)):
            net.add('hp.%s%s' % (k, t), m_, 'lif_cur', pos=space.blob(cen, m_, rng0), **cortex.NEURON)
        for (pre, post, i_, j_) in ((k + 'e', k + 'e', 0, 0), (k + 'e', k + 'i', 1, 0), (k + 'i', k + 'e', 0, 1), (k + 'i', k + 'i', 1, 1)):
            if pre == 'DGe' and post == 'DGe': continue
            a, b = net.pops['hp.' + pre], net.pops['hp.' + post]
            d2 = ((b['pos'][:, None, :] - a['pos'][None, :, :]) ** 2).sum(2)
            jj, ii = np.nonzero(rng0.random(d2.shape) < np.minimum(1.0, cortex.C0[i_, j_] * np.exp(-d2 / (2 * cortex.SIGMA ** 2))))
            w = W if j_ == 0 else G * W
            net.connect_pairs('hp.' + pre, 'hp.' + post, ii, jj, w, abs(w) * 0.1, cortex.D_E if j_ == 0 else cortex.D_I, 0.4)
        for i_, t in ((0, 'e'), (1, 'i')):                                                    # the rest of its usual input, as for a layer-2/3 cell
            p_ = 'L23' + t
            hbg[k + t] = net.poisson_in('hp.%s%s' % (k, t), cortex.K_EXT[p_] * cortex.BG_RATE + sum(Kin[i_, j] * rf[cortex.POPS[j]] for j in range(0, 8, 2) if j != 2), W)
            if t == 'e': net.poisson_in('hp.%se' % k, Kin[0, 2] * rf['L4e'], W * cortex.W_L4E_L23E)
            else: hbg[k + t] += Kin[1, 2] * rf['L4e']
            net.poisson_in('hp.%s%s' % (k, t), sum(Kin[i_, j] * rf[cortex.POPS[j]] for j in range(1, 8, 2)), G * W)

    def hlink(pre, post, rate, pre_rng=None, post_rng=None, bgarr=None):
        net.connect(pre, post, P['K_hip'], W, W * 0.1, 1.5, 0.5, mode='indeg', pre_rng=pre_rng, post_rng=post_rng, speed=SPEED)
        if bgarr is not None: bgarr -= P['K_hip'] * rate
    ec = span('EC', 'L23e')
    for k in ('DG', 'CA3', 'CA1'):
        hlink('cx.L23e', 'hp.%se' % k, rf['L23e'], pre_rng=ec, bgarr=hbg[k + 'e'])
    hlink('hp.DGe', 'hp.CA3e', rf['L23e'], bgarr=hbg['CA3e']); hlink('hp.CA3e', 'hp.CA1e', rf['L23e'], bgarr=hbg['CA1e'])
    for dp in ('L5e', 'L6e'):
        r0, r1 = span('EC', dp); hlink('hp.CA1e', 'cx.' + dp, rf['L23e'], post_rng=(r0, r1)); comp(dp, r0, r1, P['K_hip'], W, rf['L23e'])
    info['hp_n'] = hn
    # ---------------------------------------------------------------- basal ganglia
    bn = bg.add_bg(net, 'bg', scale=300.0 / bg.N_FULL['M1'], floor=30)
    where = dict(M1='striatum', M2='striatum', FS='striatum', GA='pallidum', GI='pallidum', ST='STN', SN='SNr')
    for k, v in where.items():
        net.pops['bg.' + k]['pos'] = space.blob(C[v], bn[k], rng0)
    fr = ('M1i', 'PMi', 'PFi', 'M1L', 'PML', 'PFL')
    for k in ('M1', 'M2', 'FS', 'ST'):                                                        # frontal layer 5 -> striatum, subthalamic nucleus
        for a in fr:
            net.connect('cx.L5e', 'bg.' + k, P['K_ctx_str'] / float(len(fr)), 0.5 if k != 'ST' else 0.25, 0, 2.5, 0, ch=0, mode='indeg', pre_rng=span(a, 'L5e'), speed=SPEED)
        net.pops['bg.' + k]['ext'][0][0][:] -= max(1, round(P['K_ctx_str'] / float(len(fr)))) * len(fr) * rf['L5e']
    for k in ('VL', 'MD'):                                                                    # tonic inhibition of the thalamus
        net.connect('bg.SN', 'th.TC', P['K_sn_tc'], G * W, 0, 1.0, 0, mode='indeg', post_rng=tspan(k), speed=SPEED)
    info['bg_n'] = bn
    # ---------------------------------------------------------------- cerebellum
    n_cortex = ncx + sum(hn.values())
    cb_scale = (n_cortex * 80.0 / 19.0) / (sum(cerebellum.N_FULL.values()) - cerebellum.N_FULL['glom'])
    cn = cerebellum.add_cerebellum_3d(net, C['cerebellum'], C['DCN'], 'cb', scale=cb_scale)
    net.connect('cb.dcn', 'th.TC', P['K_dcn_tc'], W, 0, 1.0, 0, mode='indeg', post_rng=tspan('VL'), speed=SPEED)
    info['cb_n'] = cn
    # ---------------------------------------------------------------- organs
    for o, k in (('ear', 'MGB'), ('eye', 'LGN')):
        net.add(o, 100, 'poisson', rate=0.0, pos=space.blob(C[o], 100, rng0))
        net.connect(o, 'th.TC', P['K_organ'], W * P['w_organ'], 0, 1.0, 0, mode='indeg', post_rng=tspan(k), speed=SPEED)
    assert all(q['pos'] is not None and len(q['pos']) == q['n'] for q in net.pops.values()), 'a population without places'
    net.finalize()
    info['counts'] = dict(cortex=n_cortex, cerebellum=sum(v for k, v in cn.items() if k != 'glom'), rest=nt * 4 + P['n_rt'] + sum(bn.values()))
    return net, info


def area_rates(net, info, T_ms):
    return {a: {p: float(net.pops['cx.' + p]['count'][info['span'](a, p)[0]:info['span'](a, p)[1]].mean()) / (T_ms * 1e-3) for p in cortex.POPS} for a in AREAS}
