"""One cortical area = the layered local circuit of Potjans & Diesmann 2014 (Cereb Cortex), numbers from its Table 5
(saved in data/potjans2014_tables.txt). 8 populations: excitatory and inhibitory cells of layers 2/3, 4, 5, 6.

The L4e -> L2/3e synapses are twice as strong ("4E to 2/3E increased by factor 2", Schmidt et al. 2018, Table 3,
citing this paper). Not from the papers: the initial membrane potentials (the start-up transient is discarded).
The number of synapses K of a projection follows from the connection probability C by the paper's own procedure (every
synapse draws its pre- and postsynaptic cell at random):  C = 1 - (1 - 1/(Npre*Npost))^K.
"""
import numpy as np

POPS = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
N_FULL = dict(L23e=20683, L23i=5834, L4e=21915, L4i=5479, L5e=4850, L5i=1065, L6e=14395, L6i=2948)
K_EXT = dict(L23e=1600, L23i=1500, L4e=2100, L4i=1900, L5e=2000, L5i=1900, L6e=2900, L6i=2100)
BG_RATE = 8.0                                                    # Hz per external input
#            from: L23e   L23i   L4e    L4i    L5e    L5i    L6e    L6i        (rows = to)
CONN = np.array([[0.101, 0.169, 0.044, 0.082, 0.032, 0.0,    0.008, 0.0],
                 [0.135, 0.137, 0.032, 0.052, 0.075, 0.0,    0.004, 0.0],
                 [0.008, 0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.0],
                 [0.069, 0.003, 0.079, 0.160, 0.003, 0.0,    0.106, 0.0],
                 [0.100, 0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.0],
                 [0.055, 0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.0],
                 [0.016, 0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225],
                 [0.036, 0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144]])
TH_CONN = dict(L4e=0.0983, L4i=0.0619, L6e=0.0512, L6i=0.0196)   # thalamic relay cells -> cortex (N_th = 902 at full scale)
N_TH_FULL = 902
W, W_SD_REL, G = 87.8, 0.1, -4.0                                 # pA; inhibitory synapses are G times as strong
D_E, D_E_SD, D_I, D_I_SD = 1.5, 0.75, 0.8, 0.4                   # ms
NEURON = dict(tau_m=10.0, C_m=250.0, tau_s=0.5, E_L=-65.0, V_th=-50.0, V_reset=-65.0, t_ref=2.0)
W_L4E_L23E = 2.0                                                 # Schmidt et al. 2018, Table 3
PUBLISHED_RATES = dict(L23e=0.86, L4e=4.45, L5e=7.59, L6e=1.09)  # Hz, the paper's Table 6 (reference model)


def k_total(c, n_pre, n_post):
    return 0 if c <= 0 else int(round(np.log(1 - c) / np.log(1 - 1.0 / (n_pre * n_post))))


def indegree_full():
    """synapses per neuron at full scale, [to][from]"""
    return np.array([[k_total(CONN[i, j], N_FULL[POPS[j]], N_FULL[POPS[i]]) / N_FULL[POPS[i]] for j in range(8)] for i in range(8)])


def add_area(net, name, scale=1.0, rates_full=None, k_scale=None):
    """adds the 8 populations and their internal synapses. scale: fraction of the full number of neurons; k_scale: fraction
    of the full number of synapses per neuron (default = scale). When synapses per neuron are reduced by k_scale, each
    weight is divided by sqrt(k_scale) (keeps the size of the input fluctuations) and the part of the MEAN input that is
    lost, (1 - sqrt(k_scale)) x full mean input, is given as a constant current - computed from the firing rates of the
    full-scale circuit (rates_full, Hz per population; measured in our own full-scale run)."""
    k_scale = scale if k_scale is None else k_scale
    f = 1.0 / np.sqrt(k_scale)
    n = {p: max(1, int(round(N_FULL[p] * scale))) for p in POPS}
    Kin = indegree_full()
    for i, p in enumerate(POPS):
        q = dict(NEURON)
        if k_scale < 1.0:
            assert rates_full is not None
            mean_in = sum(Kin[i, j] * (W if j % 2 == 0 else G * W) * (W_L4E_L23E if (p == 'L23e' and POPS[j] == 'L4e') else 1.0) * rates_full[POPS[j]] for j in range(8))
            mean_in += K_EXT[p] * W * BG_RATE
            q['I_dc'] = 1e-3 * NEURON['tau_s'] * (1 - np.sqrt(k_scale)) * mean_in       # pA
        net.add(name + '.' + p, n[p], 'lif_cur', **q)
        net.poisson_in(name + '.' + p, K_EXT[p] * k_scale * BG_RATE, W * f)
    for i, post in enumerate(POPS):
        for j, pre in enumerate(POPS):
            K = Kin[i, j] * k_scale * n[post]
            if K < 1: continue
            exc = (j % 2 == 0); w = (W if exc else G * W) * f * (W_L4E_L23E if (post == 'L23e' and pre == 'L4e') else 1.0)
            net.connect(name + '.' + pre, name + '.' + post, K, w, abs(w) * W_SD_REL, D_E if exc else D_I, D_E_SD if exc else D_I_SD)
    return n


def add_areas(net, names, scale, rates_full, k_scale, prefix='cx'):
    """Several areas at once: one population per cell type holding all areas side by side (area a = cells
    [a*n, (a+1)*n) of each population); the wiring inside each area is exactly that of add_area().
    Returns n (cells per area and type) and rng(area, pop) -> (first, end)."""
    f = 1.0 / np.sqrt(k_scale); A = len(names)
    n = {p: max(1, int(round(N_FULL[p] * scale))) for p in POPS}
    Kin = indegree_full()
    for i, p in enumerate(POPS):
        q = dict(NEURON)
        mean_in = sum(Kin[i, j] * (W if j % 2 == 0 else G * W) * (W_L4E_L23E if (p == 'L23e' and POPS[j] == 'L4e') else 1.0) * rates_full[POPS[j]] for j in range(8))
        mean_in += K_EXT[p] * W * BG_RATE
        q['I_dc'] = np.full(n[p] * A, 1e-3 * NEURON['tau_s'] * (1 - np.sqrt(k_scale)) * mean_in, dtype=np.float32)
        net.add(prefix + '.' + p, n[p] * A, 'lif_cur', **q)
        net.poisson_in(prefix + '.' + p, K_EXT[p] * k_scale * BG_RATE, W * f)
    rng = lambda a, p: (names.index(a) * n[p], (names.index(a) + 1) * n[p])
    for a in names:
        for i, post in enumerate(POPS):
            for j, pre in enumerate(POPS):
                K = Kin[i, j] * k_scale * n[post]
                if K < 1: continue
                exc = (j % 2 == 0); w = (W if exc else G * W) * f * (W_L4E_L23E if (post == 'L23e' and pre == 'L4e') else 1.0)
                net.connect(prefix + '.' + pre, prefix + '.' + post, K, w, abs(w) * W_SD_REL, D_E if exc else D_I, D_E_SD if exc else D_I_SD, pre_rng=rng(a, pre), post_rng=rng(a, post))
    return n, rng


# ---------------------------------------------------------------------------------------------------- with places in space
SIGMA = 0.30        # mm, lateral spread of local connections. Derived from two statements of the paper ("a network of
                    # 10 000 neurons represents only around 20 % of the local connectivity", "80 000 ... > 85 %") with its
                    # Gaussian profile C(r) = C0 exp(-r^2 / 2 sigma^2); its literature range is 150-310 micrometres.
THICK = 2.0         # mm, thickness of the cortical sheet - NOT read in a paper; each layer gets the share of its cells


def _mean_factor(R, sg):
    return 2 * sg ** 2 / R ** 2 * (1 - np.exp(-R ** 2 / (2 * sg ** 2)))


C0 = CONN / _mean_factor(np.sqrt(1 / np.pi), SIGMA)     # peak probability such that the mean over the 1 mm^2 circuit is the table value


def add_areas_3d(net, names, centers, scale, rates_full, prefix='cx'):
    """Every area is a small patch of cortex (scale x 1 mm^2 of surface, full depth, real cell density) at its place in the
    brain. Inside a patch two cells are connected with probability C0 exp(-r^2 / 2 sigma^2), r = their lateral distance.
    The patch is far smaller than the reach of local connections, so most local synapses of a cell come from cortex
    AROUND the patch, which is not simulated: that part is given as random (Poisson) input with the firing rates of the
    full-size circuit. Weights are the published ones (no rescaling)."""
    A = len(names); rng = net.rng
    n = {p: max(2, int(round(N_FULL[p] * scale))) for p in POPS}; side = np.sqrt(scale)
    ntot = float(sum(N_FULL.values())); depth = {}; z = 0.0
    for lay in ('L23', 'L4', 'L5', 'L6'):
        th = THICK * (N_FULL[lay + 'e'] + N_FULL[lay + 'i']) / ntot; depth[lay] = (z, z + th); z += th
    lat = {}                                                                         # lateral coordinates inside the patch
    for p in POPS:
        pos = np.zeros((n[p] * A, 3), dtype=np.float32); uv = np.zeros((n[p] * A, 2), dtype=np.float32)
        for a, nm in enumerate(names):
            c = np.asarray(centers[nm], dtype=float); nrm = c / (np.linalg.norm(c) + 1e-9)       # outward
            t1 = np.cross(nrm, [0.0, 0.0, 1.0]); t1 = t1 / (np.linalg.norm(t1) + 1e-9); t2 = np.cross(nrm, t1)
            u = rng.uniform(-side / 2, side / 2, (n[p], 2)); d0, d1 = depth[p[:-1]]; dz = rng.uniform(d0, d1, n[p]) - THICK / 2
            pos[a * n[p]:(a + 1) * n[p]] = c + u[:, :1] * t1 + u[:, 1:] * t2 - dz[:, None] * nrm; uv[a * n[p]:(a + 1) * n[p]] = u
        net.add(prefix + '.' + p, n[p] * A, 'lif_cur', pos=pos, **NEURON); lat[p] = uv
    Kin = indegree_full(); Kpatch = np.zeros((8, 8))
    for a in range(A):
        for i, post in enumerate(POPS):
            for j, pre in enumerate(POPS):
                if C0[i, j] <= 0: continue
                up = lat[post][a * n[post]:(a + 1) * n[post]]; uq = lat[pre][a * n[pre]:(a + 1) * n[pre]]
                r2 = ((up[:, None, :] - uq[None, :, :]) ** 2).sum(2)
                jj, ii = np.nonzero(rng.random(r2.shape) < np.minimum(1.0, C0[i, j] * np.exp(-r2 / (2 * SIGMA ** 2))))
                exc = (j % 2 == 0); w = (W if exc else G * W) * (W_L4E_L23E if (post == 'L23e' and pre == 'L4e') else 1.0)
                net.connect_pairs(prefix + '.' + pre, prefix + '.' + post, ii + a * n[pre], jj + a * n[post], w, abs(w) * W_SD_REL, D_E if exc else D_I, D_E_SD if exc else D_I_SD)
                Kpatch[i, j] += len(ii) / float(n[post]) / A
    bgr = {}
    for i, p in enumerate(POPS):                                                     # background + the cortex around the patch
        miss = np.maximum(Kin[i] - Kpatch[i], 0.0)
        e = sum(miss[j] * rates_full[POPS[j]] for j in range(0, 8, 2) if not (p == 'L23e' and POPS[j] == 'L4e'))
        bgr[p] = net.poisson_in(prefix + '.' + p, K_EXT[p] * BG_RATE + e, W)
        if p == 'L23e':
            net.poisson_in(prefix + '.' + p, miss[2] * rates_full['L4e'], W * W_L4E_L23E)
        net.poisson_in(prefix + '.' + p, sum(miss[j] * rates_full[POPS[j]] for j in range(1, 8, 2)), G * W)
    span = lambda a, p: (names.index(a) * n[p], (names.index(a) + 1) * n[p])
    return n, span, bgr, Kpatch / np.maximum(Kin, 1e-9)


def stp_mean(rate_hz, U, taud, tauf, n=20000):
    """mean strength factor x u / U of a synapse with short-term depression / facilitation when the sending cell fires
    as a Poisson process at rate_hz (same update as fast.pl_pre)"""
    rng = np.random.default_rng(0); el = rng.exponential(1000.0 / max(rate_hz, 1e-6), n); x, u, tot = 1.0, U, 0.0
    for e in el:
        x = 1.0 - (1.0 - x) * np.exp(-e / taud); u = U + (u - U) * np.exp(-e / tauf); tot += x * u / U; x, u = x - u * x, u + U * (1.0 - u)
    return tot / n


def add_areas_sparse(net, names, centers, scale, k_scale, rates_full, prefix='cx', exc_gain=None, plastic=None, slow=None):
    """Scaled-down areas that keep their loops. Each area is the full 1 mm^2 circuit of which only a fraction `scale` of
    the cells is simulated (each simulated cell stands for 1/scale real ones), spread over the real 1 mm x 1 mm footprint
    and the real depth. Two cells are connected according to their lateral distance, C0 exp(-r^2 / 2 sigma^2), with the
    number of synapses per cell reduced to k_scale of the real number, each synapse stronger by 1/sqrt(k_scale) (keeps
    the size of the input fluctuations), and the missing part of the MEAN input given as a constant current computed
    from the firing rates of the full-size circuit. (scale 0.02 with k_scale 0.13 keeps the full-size firing rates:
    check_column.py 0.02 2000 0.13.)
    exc_gain {area: factor}: all excitatory synapses inside that area are stronger by this factor - the gradient of
    excitation along the cortical hierarchy of Chaudhuri et al. 2015 (factor 1 + 0.68 h, h = 0 lowest .. 1 highest area).
    plastic {(pre, post): settings for sim.PLASTIC}: these connections change with activity (their start weight counts as
    w0, default 0.1 excitatory / 0.2 inhibitory; an inhibitory one takes the full-size rate of its target as target rate).
    slow {population: decay time in ms}: these cells have a second, slow synaptic current."""
    from sim import PLASTIC
    exc_gain = exc_gain or {}; plastic = plastic or {}; slow = slow or {}
    A = len(names); rng = net.rng; f = 1.0 / np.sqrt(k_scale)
    n = {p: max(2, int(round(N_FULL[p] * scale))) for p in POPS}
    ntot = float(sum(N_FULL.values())); depth = {}; z = 0.0
    for lay in ('L23', 'L4', 'L5', 'L6'):
        th = THICK * (N_FULL[lay + 'e'] + N_FULL[lay + 'i']) / ntot; depth[lay] = (z, z + th); z += th
    Kin = indegree_full(); lat = {}
    for i, p in enumerate(POPS):
        pos = np.zeros((n[p] * A, 3), dtype=np.float32); uv = np.zeros((n[p] * A, 2), dtype=np.float32)
        for a, nm in enumerate(names):
            c = np.asarray(centers[nm], dtype=float); nrm = c / (np.linalg.norm(c) + 1e-9)
            t1 = np.cross(nrm, [0.0, 0.0, 1.0]); t1 = t1 / (np.linalg.norm(t1) + 1e-9); t2 = np.cross(nrm, t1)
            u = rng.uniform(-0.5, 0.5, (n[p], 2)); d0, d1 = depth[p[:-1]]; dz = rng.uniform(d0, d1, n[p]) - THICK / 2
            pos[a * n[p]:(a + 1) * n[p]] = c + u[:, :1] * t1 + u[:, 1:] * t2 - dz[:, None] * nrm; uv[a * n[p]:(a + 1) * n[p]] = u
        q = dict(NEURON)
        m_exc = sum(Kin[i, j] * W * (W_L4E_L23E if (p == 'L23e' and POPS[j] == 'L4e') else 1.0) * rates_full[POPS[j]] for j in range(0, 8, 2))
        m_inh = sum(Kin[i, j] * G * W * rates_full[POPS[j]] for j in range(1, 8, 2))
        gain = np.repeat([exc_gain.get(nm, 1.0) for nm in names], n[p])
        q['I_dc'] = (1e-3 * NEURON['tau_s'] * (1 - np.sqrt(k_scale)) * (m_exc * gain + m_inh + K_EXT[p] * W * BG_RATE)).astype(np.float32)
        if p in slow: q['tau_s2'] = slow[p]
        net.add(prefix + '.' + p, n[p] * A, 'lif_cur', pos=pos, **q); lat[p] = uv
        net.poisson_in(prefix + '.' + p, K_EXT[p] * k_scale * BG_RATE, W * f)
    for a in range(A):
        for i, post in enumerate(POPS):
            for j, pre in enumerate(POPS):
                if CONN[i, j] <= 0: continue
                up = lat[post][a * n[post]:(a + 1) * n[post]]; uq = lat[pre][a * n[pre]:(a + 1) * n[pre]]
                P0 = np.exp(-((up[:, None, :] - uq[None, :, :]) ** 2).sum(2) / (2 * SIGMA ** 2))
                cnt = rng.poisson(P0 * (Kin[i, j] * k_scale * n[post] / P0.sum()))             # synapses per pair, by distance
                jj, ii = np.nonzero(cnt); rep = cnt[jj, ii]; jj = np.repeat(jj, rep); ii = np.repeat(ii, rep)
                exc = (j % 2 == 0); w = (W if exc else G * W) * f * (W_L4E_L23E if (post == 'L23e' and pre == 'L4e') else 1.0)
                if exc: w *= exc_gain.get(names[a], 1.0)
                pl = None
                if (pre, post) in plastic:
                    kw = dict(plastic[(pre, post)]); w0 = kw.pop('w0', 0.1 if exc else 0.2)
                    if not exc: kw.setdefault('target', rates_full[post])
                    m = stp_mean(rates_full[pre], kw.get('U', 0.2), kw.get('taud', 150.0), kw.get('tauf', 600.0)) if (exc and kw.get('stp', True)) else 1.0
                    pl = PLASTIC(unit=abs(w) / w0 / m, rule='exc' if exc else 'inh', **kw)   # at the sending cells' usual rate the synapse is as strong as the fixed one
                net.connect_pairs(prefix + '.' + pre, prefix + '.' + post, ii + a * n[pre], jj + a * n[post], w, abs(w) * W_SD_REL, D_E if exc else D_I, D_E_SD if exc else D_I_SD,
                                  plastic=pl, label=a)
    span = lambda a, p: (names.index(a) * n[p], (names.index(a) + 1) * n[p])
    return n, span
