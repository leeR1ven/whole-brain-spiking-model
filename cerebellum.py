"""Cerebellar cortex + deep nuclei, copied from the scaffold model of Casali et al. 2019 (Front Neuroinform):
  numbers of cells (its Table 4), leaky integrate-and-fire neurons with conductance synapses (Table 2), synaptic weights and
  delays (Table 3). Inputs per cell ("convergence") are those reported for the same scaffold by Geminiani et al. 2019
  (Front Comput Neurosci, Table 2). Tables saved in data/.

NOT from these papers (flagged):
  - synaptic reversal potentials (0 mV excitatory, -85 mV inhibitory): not given in the text; the usual defaults of the
    simulator they used, from my own knowledge of it.
  - the scaffold places cells in space and wires by geometry; here cells are wired at random with the reported numbers of
    inputs per cell.
  - inhibition onto a Purkinje cell: the source gives 20 inputs from stellate + basket cells together; split 10 + 10 here.
  - the inferior olive / climbing fibres are NOT included yet (no neuron parameters for them in the papers read).
"""
N_FULL = dict(glom=7070, grc=88158, goc=219, sc=603, bc=603, pc=69, dcn=12)
#            C_m   tau_m  E_L   t_ref  I_e    V_reset V_th  tau_e  tau_i
CELL = dict(goc=(76.0, 21.0, -65.0, 2.0, 36.8, -75.0, -55.0, 0.5, 10.0),
            grc=(3.0, 2.0, -74.0, 1.5, 0.0, -84.0, -42.0, 0.5, 10.0),
            pc=(620.0, 88.0, -62.0, 0.8, 600.0, -72.0, -47.0, 0.5, 1.6),
            bc=(14.6, 14.6, -68.0, 1.6, 15.6, -78.0, -53.0, 0.64, 2.0),
            sc=(14.6, 14.6, -68.0, 1.6, 15.6, -78.0, -53.0, 0.64, 2.0),
            dcn=(89.0, 57.0, -59.0, 3.7, 55.8, -69.0, -48.0, 7.1, 13.6))
E_EXC, E_INH = 0.0, -85.0
#        pre     post   inputs/cell  weight nS  delay ms  kind
CONN = [('glom', 'grc', 4,      9.0,    4.0, 'e'),
        ('glom', 'goc', 65,     2.0,    4.0, 'e'),
        ('goc',  'grc', 2,      5.0,    2.0, 'i'),
        ('goc',  'goc', 34,     8.0,    1.0, 'i'),
        ('grc',  'goc', 360,    20.0,   2.0, 'e'),     # ascending axon
        ('grc',  'goc', 1600,   0.4,    5.0, 'e'),     # parallel fibre
        ('sc',   'sc',  4,      2.0,    1.0, 'i'),
        ('bc',   'bc',  4,      2.5,    1.0, 'i'),
        ('grc',  'sc',  1021,   0.2,    5.0, 'e'),
        ('grc',  'bc',  1004,   0.2,    5.0, 'e'),
        ('sc',   'pc',  10,     8.5,    5.0, 'i'),
        ('bc',   'pc',  10,     9.0,    4.0, 'i'),
        ('grc',  'pc',  249,    75.0,   2.0, 'e'),     # ascending axon
        ('grc',  'pc',  28401,  0.02,   5.0, 'e'),     # parallel fibre
        ('pc',   'dcn', 26,     0.0075, 4.0, 'i'),
        ('glom', 'dcn', 147,    0.006,  4.0, 'e')]
PUBLISHED_RATES = dict(grc=2.0, goc=22.7, sc=33.9, bc=30.1, pc=58.5, dcn=16.1)   # Hz, Casali Table 4, with 1 Hz on the glomeruli


def add_cerebellum(net, name='cb', scale=1.0, glom_rate=1.0, glom_model=True):
    """scale: fraction of the full number of cells. Inputs per cell are kept as published (never more than the number of
    sending cells). With glom_model=False the glomeruli are left out (to be driven by mossy fibres from elsewhere)."""
    n = {k: max(4, int(round(v * scale))) for k, v in N_FULL.items()}
    if glom_model:
        net.add(name + '.glom', n['glom'], 'poisson', rate=glom_rate)
    for k, (C, tm, EL, tr, Ie, Vr, Vt, te, ti) in CELL.items():
        net.add(name + '.' + k, n[k], 'lif_cond', C_m=C, tau_m=tm, E_L=EL, t_ref=tr, I_e=Ie, V_reset=Vr, V_th=Vt, tau_e=te, tau_i=ti, E_e=E_EXC, E_i=E_INH)
    for pre, post, conv, w, d, kind in CONN:
        if pre == 'glom' and not glom_model: continue
        k = min(conv, n[pre])                               # with fewer sending cells than published inputs: all of them, each stronger
        net.connect(name + '.' + pre, name + '.' + post, k, w * conv / k, 0.0, d, 0.0, ch=0 if kind == 'e' else 1, mode='indeg')
    return n


def add_cerebellum_3d(net, center, dcn_center, name='cb', scale=1.0, glom_rate=1.0, lazy=True):
    """The same network with every cell placed in space as in the scaffold (Casali et al. 2019): a slab with a
    400 x 400 micrometre base (at full size) made of the granular layer (150 um), the Purkinje layer (30 um) and the
    molecular layer (150 um); deep nuclei in a 200 x 200 x 600 um block. Axes: x across the folium, y up through the
    layers, z along the parallel fibres.
    Wiring by place (from the paper's text): a granule cell takes its 4 glomeruli within 40 um (the length of its
    dendrites); a Purkinje cell's dendritic tree is a flat fan, 150 um radius x 30 um thick, standing across the parallel
    fibres: it takes ascending-axon synapses from granule cells under the fan and parallel-fibre synapses from granule
    cells whose fibre crosses it. All other connections: at random with the published number of inputs (their geometry
    is in the paper's supplement, not read). Basket cells in the lower half of the molecular layer and stellate cells in
    the upper half: not from the paper."""
    import numpy as np
    rng = net.rng; n = {k: max(4, int(round(v * scale))) for k, v in N_FULL.items()}; s = 0.4 * np.sqrt(scale)
    def box(m, y0, y1, c=center, sx=s, sz=s):
        return np.c_[rng.uniform(-sx / 2, sx / 2, m), rng.uniform(y0, y1, m), rng.uniform(-sz / 2, sz / 2, m)]
    loc = dict(glom=box(n['glom'], 0, 0.15), grc=box(n['grc'], 0, 0.15), goc=box(n['goc'], 0, 0.15), pc=box(n['pc'], 0.15, 0.18),
               bc=box(n['bc'], 0.18, 0.255), sc=box(n['sc'], 0.255, 0.33), dcn=box(n['dcn'], 0, 0.6, sx=s / 2, sz=s / 2))
    pos = {k: (v + (np.asarray(dcn_center) if k == 'dcn' else np.asarray(center))).astype(np.float32) for k, v in loc.items()}
    net.add(name + '.glom', n['glom'], 'poisson', rate=glom_rate, pos=pos['glom'])
    for k, (C, tm, EL, tr, Ie, Vr, Vt, te, ti) in CELL.items():
        net.add(name + '.' + k, n[k], 'lif_cond', pos=pos[k], C_m=C, tau_m=tm, E_L=EL, t_ref=tr, I_e=Ie, V_reset=Vr, V_th=Vt, tau_e=te, tau_i=ti, E_e=E_EXC, E_i=E_INH,
                lazy=(lazy and k == 'grc'))        # granule cells: no drive of their own, silent unless a glomerulus fires
    for pre, post, conv, w, d, kind in CONN:
        ch = 0 if kind == 'e' else 1
        if (pre, post) == ('glom', 'grc'):
            ii = np.zeros((n['grc'], 4), dtype=np.int32); G_ = loc['glom']
            for a in range(0, n['grc'], 2000):
                d2 = ((loc['grc'][a:a + 2000, None, :] - G_[None, :, :]) ** 2).sum(2)
                near = np.argsort(d2 + (d2 > 0.04 ** 2) * 1e3 + rng.random(d2.shape) * 1e-9, axis=1)[:, :12]     # the ones within reach first
                pick = np.argsort(rng.random(near.shape) + (np.take_along_axis(d2, near, 1) > 0.04 ** 2) * 10, axis=1)[:, :4]
                ii[a:a + 2000] = np.take_along_axis(near, pick, 1)
            net.connect_pairs(name + '.glom', name + '.grc', ii.ravel(), np.repeat(np.arange(n['grc']), 4), w, 0, d, 0, ch=ch)
        elif (pre, post) == ('grc', 'pc'):
            I, J = [], []
            for k in range(n['pc']):
                dx = np.abs(loc['grc'][:, 0] - loc['pc'][k, 0]) < 0.15
                cand = np.nonzero(dx & (np.abs(loc['grc'][:, 2] - loc['pc'][k, 2]) < 0.015))[0] if conv < 1000 else np.nonzero(dx)[0]
                m = min(conv, len(cand)); take = rng.choice(cand, m, replace=False) if m else cand
                I.append(take); J.append(np.full(len(take), k))
            I = np.concatenate(I); J = np.concatenate(J)
            net.connect_pairs(name + '.grc', name + '.pc', I, J, w * conv * n['pc'] / max(1, len(I)), 0, d, 0, ch=ch)
        else:
            k = min(conv, n[pre])
            net.connect(name + '.' + pre, name + '.' + post, k, w * conv / k, 0.0, d, 0.0, ch=ch, mode='indeg')
    return n
