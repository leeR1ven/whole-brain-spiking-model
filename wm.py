"""A working-memory circuit: the attractor network of Brunel & Wang 2001 in the version of Loh, Rolls & Deco 2007
(PLoS Comput Biol; parameters from its supplement, saved in data/loh2007_supplement.txt).
400 excitatory + 100 inhibitory integrate-and-fire cells, every cell connected to every cell. Excitatory synapses have a
fast part (AMPA, 2 ms) and a slow part (NMDA, 100 ms, blocked by magnesium at low voltage, saturating); inhibitory synapses
GABA (10 ms). Two "selective pools" of excitatory cells whose mutual synapses are stronger (w+ = 2.1); all other excitatory
synapses onto pool cells are weaker (w-) so that the mean input stays the same.

NOT from the paper (flagged): the size of a selective pool (given only in its figure; 10 % of the excitatory cells is
used here); the synaptic delay (0.5 ms); the NMDA rise time (2 ms) is folded into the size of the jump per spike;
integration is forward Euler at 0.1 ms (the paper: Runge-Kutta at 0.02 ms)."""
import numpy as np
NE, NI, F, WP = 400, 100, 0.1, 2.1
E_CELL = dict(C_m=500.0, tau_m=20.0, E_L=-70.0, V_th=-50.0, V_reset=-55.0, t_ref=2.0, I_e=0.0)
I_CELL = dict(C_m=200.0, tau_m=10.0, E_L=-70.0, V_th=-50.0, V_reset=-55.0, t_ref=1.0, I_e=0.0)
SLOTS = [(2.0, 0.0), (100.0, 0.0, 'nmda'), (10.0, -70.0)]             # AMPA, NMDA, GABA
G = dict(E=dict(ext=2.08, ampa=0.208, nmda=0.654, gaba=2.50), I=dict(ext=1.62, ampa=0.162, nmda=0.516, gaba=1.946))   # nS
EXT_RATE = 2400.0                                                     # Hz: 800 outside cells at 3 Hz
NMDA_JUMP = 1 - np.exp(-0.5 * 2.0)                                    # alpha * rise time, with saturation
PUBLISHED = 'spontaneous: excitatory 3 Hz, inhibitory 9 Hz; after a 500 ms cue the cued pool keeps firing at about 30 Hz'


def add_wm(net, name='wm', ne=NE, ni=NI, f=F, wp=WP, pos_e=None, pos_i=None, d=0.5, g_scale=1.0):
    """g_scale: all recurrent conductances are multiplied by NE / ne * g_scale (the paper's values are for 400 cells)."""
    net.add(name + '.E', ne, 'lif_cond', slots=SLOTS, pos=pos_e, **E_CELL); net.add(name + '.I', ni, 'lif_cond', slots=SLOTS, pos=pos_i, **I_CELL)
    net.poisson_in(name + '.E', EXT_RATE, G['E']['ext'], 0); net.poisson_in(name + '.I', EXT_RATE, G['I']['ext'], 0)
    npool = int(round(f * ne)); pool = np.full(ne, -1); pool[:npool] = 0; pool[npool:2 * npool] = 1
    wm = (1 - f * wp) / (1 - f)                                       # keeps the mean input of a pool cell constant
    w = np.ones((ne, ne))                                             # w[pre, post]
    for k in (0, 1):
        w[:, pool == k] = wm; w[np.ix_(pool == k, pool == k)] = wp
    sc = float(NE) / ne * g_scale
    ii, jj = np.nonzero(np.ones((ne, ne)) - np.eye(ne))
    for cells, gE in ((w[ii, jj], None),):
        pass
    # excitatory -> excitatory: one projection per weight value (AMPA and NMDA)
    for val in np.unique(w):
        m = (w[ii, jj] == val)
        net.connect_pairs(name + '.E', name + '.E', ii[m], jj[m], G['E']['ampa'] * val * sc, 0, d, 0, ch=0)
        net.connect_pairs(name + '.E', name + '.E', ii[m], jj[m], G['E']['nmda'] * val * sc, 0, d, 0, ch=1, stp=('nmda', 100.0, NMDA_JUMP))
    a, b = np.nonzero(np.ones((ne, ni)))
    net.connect_pairs(name + '.E', name + '.I', a, b, G['I']['ampa'] * sc, 0, d, 0, ch=0)
    net.connect_pairs(name + '.E', name + '.I', a, b, G['I']['nmda'] * sc, 0, d, 0, ch=1, stp=('nmda', 100.0, NMDA_JUMP))
    a, b = np.nonzero(np.ones((ni, ne))); net.connect_pairs(name + '.I', name + '.E', a, b, G['E']['gaba'] * sc * NI / ni * ne / NE, 0, d, 0, ch=2)
    a, b = np.nonzero(np.ones((ni, ni)) - np.eye(ni)); net.connect_pairs(name + '.I', name + '.I', a, b, G['I']['gaba'] * sc * NI / ni * ne / NE, 0, d, 0, ch=2)
    return pool
