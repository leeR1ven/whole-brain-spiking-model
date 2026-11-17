"""Spiking-network simulator (numpy). Everything is neurons and synapses.

Neuron models - each is the model used in the paper the part is copied from:
  lif_cur   leaky integrate-and-fire, exponential synaptic CURRENTS              cortex      (Potjans & Diesmann 2014)
  lif_cond  leaky integrate-and-fire, exponential synaptic CONDUCTANCES + I_e    cerebellum  (Casali et al. 2019),
                                                                                 working-memory circuit (Brunel & Wang 2001)
  adex      adaptive exponential integrate-and-fire, conductances                GPe, STN, SNr (Lindahl & Kotaleski 2016)
  qif       quadratic integrate-and-fire with a slow recovery current            striatal projection neurons (same paper)
  fsn       the same with the fast-spiking interneuron's recovery rule           striatal interneurons (same paper)
  poisson   a spike source with a given rate (an input fibre)
A conductance population has a list of input slots (receptor kinds), each (decay time, reversal potential[, 'nmda']).
A synapse = (target slot, weight, delay). A projection may have short-term facilitation / depression,
stp = (U, tau_rec, tau_fac), or be an NMDA synapse whose open-channel fraction saturates per sending cell,
stp = ('nmda', decay time, jump). Spikes travel through a ring buffer as long as the longest delay.
Time step dt in ms; units mV, ms, pA, nS, pF.

Synapses that change (connect(..., plastic=PLASTIC(...))): the rules of Zenke, Agnes & Gerstner 2015, as in zenke.py
(which reproduces that paper's network). An excitatory one may put a share of its charge into the slow current of a
lif_cur cell that has tau_s2. Not from the paper: the weight change of a sending spike is made when the spike is
fired, not when it arrives; one 20 ms trace per cell serves as both sending and receiving trace.

Speed - the network is sparsely active:
  * all populations of one neuron model are updated together as one block of arrays;
  * background input and input fibres: only the events are generated (same statistics as drawing for every cell);
  * a population added with lazy=True (no background input, no drive of its own) is updated only for the cells an
    excitatory input has just reached (see _lazy; an approximation, checked against the full update);
  * the loops over cells and synapses are compiled (fast.py, numba); Net(fast_=False) runs the same in plain numpy.

Not taken from a paper: the adex / qif / fsn equations are written in the standard form of the models the paper names
(its equations were lost when the text was fetched) and integrated by forward Euler; leaky integrate-and-fire cells are
integrated exactly within a step; an NMDA slot uses the magnesium-block factor 1 / (1 + [Mg]/3.57 exp(-0.062 V)),
[Mg] = 1 mM (Jahr & Stevens, as in Brunel & Wang 2001).
"""
import numpy as np, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylib'))
try:
    import fast
except Exception as _e:                                                             # no compiler: the numpy path is used
    fast = None

ORDER = ['poisson', 'lif_cur', 'lif_cond', 'lazy', 'adex', 'qif', 'fsn']
PAR = dict(lif_cur=['tau_m', 'C_m', 'tau_s', 'E_L', 'V_th', 'V_reset'],
           lif_cond=['C_m', 'tau_m', 'E_L', 'V_th', 'V_reset', 'I_e'], lazy=['C_m', 'tau_m', 'E_L', 'V_th', 'V_reset', 'I_e'],
           adex=['C_m', 'g_L', 'E_L', 'V_T', 'D_T', 'a', 'b', 'tau_w', 'I_e', 'V_peak', 'V_reset'],
           qif=['C_m', 'k', 'v_r', 'v_t', 'a', 'b', 'c', 'd', 'v_peak'], fsn=['C_m', 'k', 'v_r', 'v_t', 'a', 'b', 'c', 'd', 'v_peak', 'v_b'])


def PLASTIC(unit, rule='exc', A=1e-3, beta=0.05, delta=2e-5, wmax=5.0, stp=True, U=0.2, taud=150.0, tauf=600.0, a_fast=1.0, tau_fast=0.5, tau_slow=100.0,
            pot=None, wa=0.0, wc=0.5, eta=2e-5, target=4.0, hom=False, tauh=600e3, kappa=10.0, hom0=2.0):
    """settings of a projection whose synapses change. A weight is a plain number (0.1 = the paper's start value);
    `unit` (pA) turns it into current. rule 'exc': triplet potentiation + depression + heterosynaptic + transmitter-
    induced terms + consolidation, with short-term plasticity; a_fast = share of the charge in the fast current, the rest
    goes into the slow current (same charge). rule 'inh': inhibitory synapses onto excitatory cells, strengthened when the
    receiving population (cells with the same label) fires above `target` Hz, weakened below. hom: depression scaled by
    (long-run rate of the receiving cell / kappa)^2, long-run = over tauh ms, starting at hom0 Hz. Default numbers = those
    of the authors' run scripts (zenke.py)."""
    if rule == 'inh': return dict(rule='inh', unit=float(unit), eta=eta, target=float(target), wmax=wmax)
    wt = (wc - wa) / 4.0; norm = (wt - wa) * (wt - (wa + wc) / 2) * (wt - wc)
    return dict(rule='exc', unit=float(unit), A=A, beta=beta, delta=delta, wmax=wmax, stp=bool(stp), U=U, taud=taud, tauf=tauf, a_fast=a_fast,
                slow_gain=(1.0 - a_fast) * tau_fast / tau_slow, pot=(0.1 if pot is None else pot) / norm, wa=wa, wb=(wa + wc) / 2, wc=wc,
                hom=bool(hom), tauh=float(tauh), kappa=float(kappa), hom0=float(hom0))


class Net:
    def __init__(s, dt=0.1, seed=0, fast_=True):
        s.dt, s.rng = dt, np.random.default_rng(seed); s.fast = fast if fast_ else None
        if s.fast: s.fast.seed(seed)
        s.pops = {}; s.proj = {}; s.N = 0; s.S = 0; s.nsyn = 0; s.maxd = 1; s.pproj = {}; s.ptab = []; s.learn = True

    # ---------------------------------------------------------------- building
    def add(s, name, n, model, slots=None, pos=None, **p):
        """pos: (n, 3) array, the place of every cell in mm (brain coordinates)"""
        if model == 'lif_cur': C = 2 if 'tau_s2' in p else 1                         # tau_s2: a second, slow synaptic current (slot 1)
        elif model == 'poisson': C = 0
        else:
            slots = slots or [(p['tau_e'], p['E_e']), (p['tau_i'], p['E_i'])]; C = len(slots)
        s.pops[name] = dict(name=name, n=int(n), model=model, p=p, slot0=s.S, C=C, slots=slots, ext=[], pos=None if pos is None else np.asarray(pos, dtype=np.float32))
        s.N += int(n); s.S += int(n) * C
        return s.pops[name]

    def poisson_in(s, name, rate_hz, w, ch=0):
        """independent Poisson spikes (total rate per neuron, Hz; one number or one per neuron), each of weight w, into
        slot ch. Returns the per-neuron rate array (it may be changed until finalize())."""
        r = np.full(s.pops[name]['n'], rate_hz, dtype=np.float64) if np.isscalar(rate_hz) else np.asarray(rate_hz, dtype=np.float64).copy()
        s.pops[name]['ext'].append((r, float(w), ch)); return r

    def connect(s, pre, post, K, w, w_sd, d, d_sd, ch=0, mode='total', stp=None, pre_rng=None, post_rng=None, speed=None, plastic=None, label=0):
        """mode 'total': K synapses, pre and post cell drawn independently for each (as Potjans & Diesmann);
        'indeg': K inputs for every post cell; 'outdeg': K targets for every pre cell.
        pre_rng / post_rng = (first, end): only that part of the population (e.g. one cortical area).
        speed = (mm per ms, relative spread): adds the travel time over the distance between the two cells to the delay."""
        a, b = s.pops[pre], s.pops[post]; rng = s.rng
        a0, a1 = pre_rng or (0, a['n']); b0, b1 = post_rng or (0, b['n'])
        if mode == 'total':
            K = int(K); i = rng.integers(a0, a1, K, dtype=np.int32); j = rng.integers(b0, b1, K, dtype=np.int32)
        elif mode == 'indeg':
            k = max(1, int(round(K))); j = np.repeat(np.arange(b0, b1, dtype=np.int32), k); i = rng.integers(a0, a1, len(j), dtype=np.int32)
        else:
            k = max(1, int(round(K))); i = np.repeat(np.arange(a0, a1, dtype=np.int32), k); j = rng.integers(b0, b1, len(i), dtype=np.int32)
        return s._store(pre, post, i, j, w, w_sd, d, d_sd, ch, stp, speed, plastic, label)

    def connect_pairs(s, pre, post, i, j, w, w_sd, d, d_sd, ch=0, stp=None, speed=None, plastic=None, label=0):
        """explicit synapses: cell i[k] of pre -> cell j[k] of post (used when wiring follows the cells' places)"""
        return s._store(pre, post, np.asarray(i, dtype=np.int32), np.asarray(j, dtype=np.int32), w, w_sd, d, d_sd, ch, stp, speed, plastic, label)

    def _store(s, pre, post, i, j, w, w_sd, d, d_sd, ch, stp, speed, plastic=None, label=0):
        a, b = s.pops[pre], s.pops[post]; rng = s.rng
        if len(i) == 0:
            return 0
        ww = (w + w_sd * rng.standard_normal(len(i))).astype(np.float32)
        ww = np.maximum(ww, 0) if w > 0 else np.minimum(ww, 0)                      # a synapse keeps its sign
        dl = d + d_sd * rng.standard_normal(len(i))
        if speed is not None:                                                       # travel time = distance between the two cells / speed
            tt = np.sqrt(((a['pos'][i] - b['pos'][j]) ** 2).sum(1)) / speed[0]
            dl = dl + tt * (1 + speed[1] * rng.standard_normal(len(i)))
        dd = np.maximum(1, np.rint(dl / s.dt)).astype(np.uint16)
        s.maxd = max(s.maxd, int(dd.max()) + 1)
        slot = (b['slot0'] + j.astype(np.int64) * b['C'] + ch).astype(np.int32)
        if plastic is not None:                                                     # a synapse that changes: see PLASTIC below
            key = (pre, post, tuple(sorted((k, v) for k, v in plastic.items() if k != 'unit')))     # 'unit' is kept per synapse
            s.pproj.setdefault(key, []).append((i, j, slot, np.abs(ww).astype(np.float64) / plastic['unit'], dd, np.full(len(i), label, dtype=np.int64),
                                                np.full(len(i), plastic['unit'], dtype=np.float32))); s.nsyn += len(i)
            return len(i)
        s.proj.setdefault(pre, {}).setdefault(stp, []).append((i, slot, ww, dd)); s.nsyn += len(i)
        return len(i)

    def finalize(s):
        dt, rng = s.dt, s.rng; s.L = s.maxd + 1
        key = lambda q: 'lazy' if (q['model'] == 'lif_cond' and q['p'].get('lazy')) else q['model']
        s.groups = []; slotmap = np.zeros(max(s.S, 1), dtype=np.int32); S = 0
        for m in ORDER:
            ps = [q for q in s.pops.values() if key(q) == m]
            if not ps: continue
            C = max(q['C'] for q in ps); n = sum(q['n'] for q in ps); G = dict(model=m, pops=ps, n=n, C=C, slot0=S); off = 0
            for q in ps:
                q['grp'], q['off'] = G, off
                if q['C']:
                    o = np.arange(q['n'] * q['C']); slotmap[q['slot0'] + o] = S + (off + o // q['C']) * C + o % q['C']
                off += q['n']
            S += n * C
            rep = lambda nm, d=None: np.concatenate([np.broadcast_to(np.asarray(q['p'].get(nm, d), dtype=np.float32), (q['n'],)) for q in ps])
            G['count'] = np.zeros(n, dtype=np.int64)
            if m == 'poisson':
                G['rate'] = rep('rate', 0.0).astype(np.float64); s._prate(G)
            else:
                G['ref'] = np.zeros(n, dtype=np.int32); G['tref'] = np.rint(rep('t_ref', 0.0) / dt).astype(np.int32)
                for nm in PAR[m]: G[nm] = rep(nm)
            if m == 'lif_cur':
                G['V'] = (rep('V0', -58.0) + rep('V0_sd', 10.0) * rng.standard_normal(n)).astype(np.float32); G['I'] = np.zeros(n, dtype=np.float32)
                G['P22'] = np.exp(-dt / G['tau_m']); G['P11'] = np.exp(-dt / G['tau_s'])
                G['P21'] = G['tau_s'] * G['tau_m'] / (G['C_m'] * (G['tau_m'] - G['tau_s'])) * (G['P22'] - G['P11'])
                G['dc'] = rep('I_dc', 0.0) * (1 - G['P22']) * G['tau_m'] / G['C_m']
                G['vpk'], G['vrs'] = G['V_th'], G['V_reset']
                if C == 2:
                    ts2 = rep('tau_s2', 100.0); G['I2'] = np.zeros(n, dtype=np.float32); G['P11b'] = np.exp(-dt / ts2)
                    G['P21b'] = ts2 * G['tau_m'] / (G['C_m'] * (G['tau_m'] - ts2)) * (G['P22'] - G['P11b'])
            elif m != 'poisson':
                v0 = G['E_L'] if 'E_L' in G else G['v_r']
                G['V'] = (v0 + 2.0 * rng.standard_normal(n)).astype(np.float32); G['g'] = np.zeros((n, C), dtype=np.float32); G['w'] = np.zeros(n, dtype=np.float32)
                dec = np.ones((n, C), dtype=np.float32); er = np.zeros((n, C), dtype=np.float32); nm_ = np.zeros((n, C), dtype=bool)
                for q in ps:
                    for k, sl in enumerate(q['slots']):
                        r0, r1 = q['off'], q['off'] + q['n']
                        dec[r0:r1, k] = np.exp(-dt / sl[0]); er[r0:r1, k] = sl[1]; nm_[r0:r1, k] = (len(sl) > 2 and sl[2] == 'nmda')
                G['dec'], G['Erev'], G['nmda'] = dec, er, (nm_ if nm_.any() else None)
                if m in ('lif_cond', 'lazy'):
                    G['gL'] = G['C_m'] / G['tau_m']; G['vpk'], G['vrs'] = G['V_th'], G['V_reset']
                elif m == 'adex':
                    G['a_below'] = rep('a_below', np.inf); G['vpk'], G['vrs'] = G['V_peak'], G['V_reset']
                else:
                    G['I_e'] = rep('I_e', 0.0); G['vpk'], G['vrs'] = G['v_peak'], G['c']
                if m == 'lazy':
                    q0 = ps[0]; assert all(q['slots'] == q0['slots'] and q['C'] == 2 for q in ps)
                    G['awake'] = np.zeros(n, dtype=np.int64); G['tup'] = np.zeros(n, dtype=np.int64)
                    G['wsteps'] = int(round(q0['p'].get('awake_ms', 10.0) / dt)); G['Ee'], G['Ei'], G['ti'] = q0['slots'][0][1], q0['slots'][1][1], q0['slots'][1][0]
            for q in ps:
                q['count'] = G['count'][q['off']:q['off'] + q['n']]
                if 'V' in G: q['V'] = G['V'][q['off']:q['off'] + q['n']]
            ent = {}                                                               # background input: (weight, slot) -> rate per cell of the block
            for q in ps:
                for r, w, ch in q['ext']:
                    ent.setdefault((float(w), ch), np.zeros(n))[q['off']:q['off'] + q['n']] += np.maximum(r, 0.0) * dt * 1e-3
            G['bg'] = [(float(lam.sum()), np.cumsum(lam) / max(float(lam.sum()), 1e-30), np.float32(w), ch) for (w, ch), lam in ent.items() if lam.sum() > 0]
            pcs = [(s._pieces(lam), w, ch) for (w, ch), lam in ent.items() if lam.sum() > 0]                 # the same, for the compiled loops
            cat = lambda xs, t: np.concatenate(xs).astype(t) if xs else np.zeros(0, dtype=t)
            G['blo'] = cat([x[0][0] for x in pcs], np.int64); G['bhi'] = cat([x[0][1] for x in pcs], np.int64); G['btot'] = cat([x[0][2] for x in pcs], np.float64)
            G['bw'] = cat([np.full(len(x[0][0]), x[1]) for x in pcs], np.float32); G['bch'] = cat([np.full(len(x[0][0]), x[2]) for x in pcs], np.int64)
            G['sp'] = np.zeros(n, dtype=np.int64)
            if m in ('lif_cond', 'adex', 'qif', 'fsn'):
                cols = dict(lif_cond=['gL', 'C_m', 'E_L', 'I_e'], adex=['C_m', 'g_L', 'E_L', 'V_T', 'D_T', 'a', 'a_below', 'tau_w', 'I_e'],
                            qif=['C_m', 'k', 'v_r', 'v_t', 'a', 'b', 'I_e', 'v_r'], fsn=['C_m', 'k', 'v_r', 'v_t', 'a', 'b', 'I_e', 'v_b'])[m]
                G['P'] = np.ascontiguousarray(np.stack([G[c] for c in cols], 1).astype(np.float32)); G['code'] = ['lif_cond', 'adex', 'qif', 'fsn'].index(m)
                G['jump'] = (G['b'] if m == 'adex' else G['d'] if m in ('qif', 'fsn') else np.zeros(n)).astype(np.float32)
                G['nmask'] = G['nmda'] if G['nmda'] is not None else np.zeros((n, C), dtype=np.bool_)
            if m == 'lazy':
                G['act'] = np.zeros(n, dtype=np.int64); G['nact'] = 0
                G['sc'] = tuple(float(G[c][0]) for c in ('gL', 'C_m', 'E_L', 'I_e')) + (float(G['Ee']), float(G['Ei']), float(G['dec'][0, 0]), float(G['dec'][0, 1]), float(G['ti']), float(G['V_th'][0]), float(G['V_reset'][0]), int(G['tref'][0]))
                assert all(np.all(G[c] == G[c][0]) for c in ('gL', 'C_m', 'E_L', 'I_e', 'V_th', 'V_reset', 'tref'))
            s.groups.append(G)
        for G in s.groups:                                                         # outgoing synapses of every block (after ALL slots are placed)
            ps, n = G['pops'], G['n']; G['csr'] = []
            keys = sorted({k for q in ps for k in s.proj.get(q['name'], {})}, key=str)
            for k in keys:
                parts = [(x[0].astype(np.int64) + q['off'], x[1], x[2], x[3]) for q in ps for x in s.proj.get(q['name'], {}).get(k, [])]
                i = np.concatenate([x[0] for x in parts]); o = np.argsort(i, kind='stable')
                tab = (np.concatenate([[0], np.cumsum(np.bincount(i, minlength=n))]).astype(np.int64), slotmap[np.concatenate([x[1] for x in parts])[o]],
                       np.concatenate([x[2] for x in parts])[o], np.concatenate([x[3] for x in parts])[o])
                if k is None: st = None
                elif k[0] == 'nmda': st = dict(kind='nmda', tau=k[1], a=k[2], s=np.zeros(n), last=np.full(n, -1e9))
                else: st = dict(kind='tm', U=k[0], trec=k[1], tfac=k[2], u=np.full(n, k[0]), x=np.ones(n), last=np.full(n, -1e9))
                G['csr'].append((tab, st))
        for G in s.groups: G['pl_out'] = []; G['pl_in'] = []
        merged = {}
        for (pre, post, pk), parts in s.pproj.items():
            a, b = s.pops[pre], s.pops[post]
            merged.setdefault((id(a['grp']), id(b['grp']), pk), dict(Gpre=a['grp'], Gpost=b['grp'], par=dict(pk), parts=[], names=[]))
            T = merged[(id(a['grp']), id(b['grp']), pk)]; T['names'].append((pre, post))
            T['parts'] += [(x[0].astype(np.int64) + a['off'], x[1].astype(np.int64) + b['off'], slotmap[x[2]], x[3], x[4], x[5], x[6]) for x in parts]
        for T in merged.values():
            cat = [np.concatenate([x[c] for x in T['parts']]) for c in range(7)]; o = np.argsort(cat[0], kind='stable'); del T['parts']
            Ga, Gb = T['Gpre'], T['Gpost']; pre = cat[0][o]; post = cat[1][o]
            T['ip'] = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=Ga['n']))]).astype(np.int64); T['pre'] = pre; T['post'] = post
            T['slot'] = cat[2][o].astype(np.int64); T['d'] = cat[4][o].astype(np.int64); T['ws'] = np.zeros((len(pre), 2)); T['ws'][:, 0] = cat[3][o]; T['unit'] = cat[6][o]
            bo = np.argsort(post, kind='stable'); T['bpos'] = bo.astype(np.int64); T['bpre'] = pre[bo]
            T['bptr'] = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=Gb['n']))]).astype(np.int64)
            T['x'] = np.ones(Ga['n']); T['u'] = np.full(Ga['n'], T['par'].get('U', 0.2)); T['last'] = np.zeros(Ga['n'], dtype=np.int64)
            if T['par']['rule'] == 'inh':                                           # label of every receiving cell (e.g. its area): one activity signal per label
                lab = np.zeros(Gb['n'], dtype=np.int64); lab[post] = cat[5][o]; nl = int(lab.max()) + 1; T['lab'] = lab
                T['nlab'] = np.maximum(np.bincount(lab[np.unique(post)], minlength=nl), 1).astype(np.float64)
                T['H'] = T['par']['target'] * T['nlab']; T['cnt'] = np.zeros(nl); T['gm'] = np.zeros(nl)
            for G in (Ga, Gb):
                if 'z20' not in G: G['z20'] = np.zeros(G['n']); G['z100'] = np.zeros(G['n']); G['zh'] = np.zeros(0); G['tauh'] = 600e3
            if T['par'].get('hom'):
                if len(Gb['zh']) == 0: Gb['zh'] = np.zeros(Gb['n'])
                Gb['tauh'] = T['par']['tauh']; Gb['zh'][np.unique(post)] = T['par']['hom0'] * T['par']['tauh'] * 1e-3
            Ga['pl_out'].append(T); Gb['pl_in'].append(T); s.ptab.append(T)
        s.pproj = None
        s.proj = None; s.S = S; s.buf = np.zeros(s.L * max(S, 1), dtype=np.float32); s.t = 0

    @staticmethod
    def _pieces(lam):
        """runs of cells with one rate: (first, end, expected events per step)"""
        cut = np.concatenate([[0], np.nonzero(np.diff(lam))[0] + 1, [len(lam)]])
        lo = cut[:-1]; hi = cut[1:]; ok = lam[lo] > 0
        return lo[ok].astype(np.int64), hi[ok].astype(np.int64), (lam[lo] * (hi - lo))[ok].astype(np.float64)

    def _prate(s, G):
        lam = np.maximum(G['rate'], 0.0) * s.dt * 1e-3; G['ptot'] = float(lam.sum()); G['pcum'] = np.cumsum(lam) / max(G['ptot'], 1e-30)
        G['plo'], G['phi'], G['ptotv'] = s._pieces(lam)

    def set_rate(s, name, rate):
        """change the rate (Hz) of a 'poisson' population while running"""
        q = s.pops[name]; q['grp']['rate'][q['off']:q['off'] + q['n']] = rate; s._prate(q['grp'])

    # ---------------------------------------------------------------- lazy cells
    def _lazy(s, G, inp):
        """A conductance cell without background input can only fire while an excitatory input is acting on it. Such a cell
        is updated step by step only for `awake_ms` after an excitatory input or a spike. While it sleeps its inhibitory
        conductance is kept exactly (decay + arriving inputs); when it wakes, its excitatory conductance has decayed away
        and its membrane potential is put at the value that the leak and the inhibition present hold it at (the membrane,
        time constant tau_m, follows the much slower inhibition)."""
        dt, t = s.dt, s.t; V, g, aw, tup = G['V'], G['g'], G['awake'], G['tup']; Ee, Ei, ti = G['Ee'], G['Ei'], G['ti']
        ex = np.nonzero(inp[:, 0])[0]
        if len(ex):
            new = ex[aw[ex] <= t]
            if len(new):                                                           # wake up
                gi = g[new, 1] * np.exp(-(t - tup[new]) * dt / ti); gL = G['gL'][new]
                g[new, 1] = gi; g[new, 0] = 0.0; V[new] = (gL * G['E_L'][new] + gi * Ei + G['I_e'][new]) / (gL + gi); tup[new] = t
            aw[ex] = t + G['wsteps']
        ih = np.nonzero(inp[:, 1])[0]
        if len(ih):
            sl = ih[aw[ih] <= t]
            if len(sl):                                                            # asleep: only the inhibitory conductance is kept
                g[sl, 1] = g[sl, 1] * np.exp(-(t - tup[sl]) * dt / ti) + inp[sl, 1]; tup[sl] = t
        a = np.nonzero(aw > t)[0]
        if len(a) == 0:
            return a
        ga = g[a] * G['dec'][a] + inp[a]; Va = V[a]; gL = G['gL'][a]
        gt = gL + ga[:, 0] + ga[:, 1]; vinf = (gL * G['E_L'][a] + G['I_e'][a] + ga[:, 0] * Ee + ga[:, 1] * Ei) / gt
        Va = vinf + (Va - vinf) * np.exp(-dt * gt / G['C_m'][a])
        r = G['ref'][a]; rf = r > 0
        Va[rf] = G['V_reset'][a][rf]; r[rf] -= 1
        f = Va >= G['V_th'][a]
        Va[f] = G['V_reset'][a][f]; r[f] = G['tref'][a][f]
        V[a] = Va; g[a] = ga; G['ref'][a] = r; tup[a] = t + 1
        sp = a[f]
        if len(sp): aw[sp] = t + G['wsteps']
        return sp

    # ---------------------------------------------------------------- running (compiled loops)
    def _run_fast(s, steps, T_ms, drive, quiet, rec, record):
        F = s.fast; S, L, buf, dt = s.S, s.L, s.buf, s.dt; t0 = time.time()
        s.d20, s.d100, s.dH, s.cons_every = np.exp(-dt / 20.0), np.exp(-dt / 100.0), np.exp(-dt / 10000.0), int(round(1200.0 / dt))
        for it in range(steps):
            t = s.t; row = (t % L) * S; arr = buf[row:row + S]
            forced = drive(t) if drive else None
            for G in s.groups:
                m, n, C, sp = G['model'], G['n'], G['C'], G['sp']
                if m == 'poisson':
                    ns = F.source(G['plo'], G['phi'], G['ptotv'], sp) if G['ptot'] > 0 else 0
                else:
                    inp = arr[G['slot0']:G['slot0'] + n * C]
                    if len(G['blo']): F.bg(inp, C, G['blo'], G['bhi'], G['btot'], G['bw'], G['bch'])
                    if m == 'lif_cur' and C == 2:
                        ns = F.lif_cur2(G['V'], G['I'], G['I2'], G['E_L'], G['P22'], G['P21'], G['P11'], G['P21b'], G['P11b'], G['dc'], G['vpk'], G['vrs'], G['ref'], G['tref'], inp.reshape(n, 2), sp)
                    elif m == 'lif_cur':
                        ns = F.lif_cur(G['V'], G['I'], G['E_L'], G['P22'], G['P21'], G['P11'], G['dc'], G['vpk'], G['vrs'], G['ref'], G['tref'], inp, sp)
                    elif m == 'lazy':
                        ns, G['nact'] = F.lazy(G['V'], G['g'], G['awake'], G['tup'], G['act'], G['nact'], G['ref'], inp.reshape(n, 2), sp, t, dt, G['wsteps'], *G['sc'])
                    else:
                        ns = F.cond(G['code'], G['V'], G['g'], G['w'], G['dec'], G['Erev'], G['nmask'], G['nmda'] is not None, G['P'], G['jump'], G['vpk'], G['vrs'], G['ref'], G['tref'], inp.reshape(n, C), sp, dt)
                if forced:
                    for q in G['pops']:
                        if q['name'] in forced:
                            x = np.union1d(sp[:ns], np.asarray(forced[q['name']], dtype=np.int64) + q['off']); ns = len(x); sp[:ns] = x
                if ns == 0:
                    if 'z20' in G: F.traces(sp, 0, G['z20'], G['z100'], G['zh'], s.d20, s.d100, np.exp(-dt / G['tauh']))
                    continue
                F.count(G['count'], sp, ns)
                if rec:
                    x = sp[:ns]
                    for q in G['pops']:
                        if q['name'] in rec:
                            y = x[(x >= q['off']) & (x < q['off'] + q['n'])]
                            if len(y): rec[q['name']].append((t, y - q['off']))
                for (ip, slot, w, d), st in G['csr']:
                    if st is None: F.deliver(sp, ns, ip, slot, w, d, buf, t, L, S)
                    elif st['kind'] == 'nmda': F.deliver_nmda(sp, ns, ip, slot, w, d, buf, t, L, S, t * dt, st['tau'], st['a'], st['s'], st['last'])
                    else: F.deliver_tm(sp, ns, ip, slot, w, d, buf, t, L, S, t * dt, st['U'], st['trec'], st['tfac'], st['u'], st['x'], st['last'])
                for T in G['pl_out']:                                               # synapses that change: sending side
                    p = T['par']
                    if p['rule'] == 'inh': F.inh_pre(sp, ns, T['ip'], T['post'], T['slot'], T['ws'], T['d'], buf, t, L, S, T['Gpost']['z20'], T['lab'], T['gm'], T['unit'], s.learn)
                    else: F.pl_pre(sp, ns, T['ip'], T['post'], T['slot'], T['ws'], T['d'], buf, t, L, S, T['Gpost']['z20'], T['unit'], p['a_fast'], p['slow_gain'], p['A'], p['delta'],
                                   p['stp'], p['U'], p['taud'], p['tauf'], dt, T['x'], T['u'], T['last'], s.learn, p['hom'], T['Gpost']['zh'], p['tauh'] * 1e-3, p['kappa'])
                for T in G['pl_in']:                                                # receiving side
                    p = T['par']
                    if p['rule'] == 'inh': F.inh_post(sp, ns, T['bptr'], T['bpos'], T['bpre'], T['ws'], T['Gpre']['z20'], T['lab'], T['gm'] if s.learn else T['gm'] * 0, p['wmax'], T['cnt'])
                    elif s.learn: F.pl_post(sp, ns, T['bptr'], T['bpos'], T['bpre'], T['ws'], T['Gpre']['z20'], G['z20'], G['z100'], p['A'], p['beta'], p['wmax'])
                if 'z20' in G: F.traces(sp, ns, G['z20'], G['z100'], G['zh'], s.d20, s.d100, np.exp(-dt / G['tauh']))
            for T in s.ptab:                                                        # slow things of the synapses that change
                p = T['par']
                if p['rule'] == 'inh':
                    T['H'] *= s.dH; T['H'] += T['cnt'] / 10.0; T['cnt'][:] = 0; T['gm'][:] = p['eta'] * (T['H'] / (p['target'] * T['nlab']) - 1.0)
                elif s.learn and t % s.cons_every == 0:
                    F.consolidate(T['ws'], p['pot'], p['wa'], p['wb'], p['wc'], 1e-3)
            s.t += 1
            if not quiet and it % 2000 == 1999:
                print('  %.0f ms simulated, %.0f s elapsed' % ((it + 1) * dt, time.time() - t0), flush=True)
        rates = {k: float(q['count'].sum()) / q['n'] / (T_ms * 1e-3) for k, q in s.pops.items()}
        return (rates, rec) if record else rates

    # ---------------------------------------------------------------- running (numpy, the reference)
    def run(s, T_ms, drive=None, quiet=True, record=None):
        """drive: function(step) -> {population: local indices forced to spike now} (sense organs).
        record: list of population names whose spikes are returned as (step, local index) lists."""
        steps = int(round(T_ms / s.dt)); S, L, rng, buf, dt = s.S, s.L, s.rng, s.buf, s.dt; t0 = time.time()
        rec = {k: [] for k in (record or [])}
        for G in s.groups: G['count'][:] = 0
        if s.fast:
            return s._run_fast(steps, T_ms, drive, quiet, rec, record)
        assert not s.ptab, 'synapses that change are only in the compiled path'
        for it in range(steps):
            row = (s.t % L) * S; arr = buf[row:row + S]
            forced = drive(s.t) if drive else None
            for G in s.groups:
                m, n = G['model'], G['n']
                if m == 'poisson':
                    k = rng.poisson(G['ptot']) if G['ptot'] > 0 else 0
                    sp = np.unique(np.searchsorted(G['pcum'], rng.random(k))) if k else np.zeros(0, dtype=np.int64)
                else:
                    inp = arr[G['slot0']:G['slot0'] + n * G['C']].reshape(n, G['C']); V = G['V']
                    if m == 'lazy':
                        sp = s._lazy(G, inp)
                    else:
                        if m == 'lif_cur':
                            I = G['I']
                            V[:] = G['E_L'] + (V - G['E_L']) * G['P22'] + I * G['P21'] + G['dc']
                            I *= G['P11']; I += inp[:, 0]
                            if 'I2' in G:
                                V += G['I2'] * G['P21b']; G['I2'] *= G['P11b']; G['I2'] += inp[:, 1]
                            for tot, cum, w, ch in G['bg']:
                                k = rng.poisson(tot)
                                if k: np.add.at(I, np.searchsorted(cum, rng.random(k)), w)
                        else:
                            g = G['g']; g *= G['dec']; g += inp
                            for tot, cum, w, ch in G['bg']:
                                k = rng.poisson(tot)
                                if k: np.add.at(g[:, ch], np.searchsorted(cum, rng.random(k)), w)
                            ge = g if G['nmda'] is None else np.where(G['nmda'], g * (1.0 / (1.0 + np.exp(-0.062 * V) / 3.57))[:, None], g)
                            gs = ge.sum(1); gE = (ge * G['Erev']).sum(1)
                            if m == 'lif_cond':
                                gt = G['gL'] + gs; vinf = (G['gL'] * G['E_L'] + G['I_e'] + gE) / gt
                                V[:] = vinf + (V - vinf) * np.exp(-dt * gt / G['C_m'])
                            elif m == 'adex':
                                w_ = G['w']; ex = np.exp(np.minimum((V - G['V_T']) / G['D_T'], 20.0)); a = np.where(V < G['a_below'], G['a'], 0.0)
                                dV = (dt / G['C_m']) * (-G['g_L'] * (V - G['E_L']) + G['g_L'] * G['D_T'] * ex - w_ + G['I_e'] + gE - gs * V)
                                w_ += (dt / G['tau_w']) * (a * (V - G['E_L']) - w_); V += dV
                            else:
                                u = G['w']
                                dV = (dt / G['C_m']) * (G['k'] * (V - G['v_r']) * (V - G['v_t']) - u + G['I_e'] + gE - gs * V)
                                if m == 'qif': u += dt * G['a'] * (G['b'] * (V - G['v_r']) - u)
                                else: u += dt * G['a'] * (np.where(V >= G['v_b'], G['b'] * (V - G['v_b']) ** 3, 0.0) - u)
                                V += dV
                        r = G['ref']; rf = np.nonzero(r)[0]
                        if len(rf):
                            V[rf] = G['vrs'][rf]; r[rf] -= 1
                        sp = np.nonzero(V >= G['vpk'])[0]
                        if len(sp):
                            V[sp] = G['vrs'][sp]; r[sp] = G['tref'][sp]
                            if m == 'adex': G['w'][sp] += G['b'][sp]
                            elif m in ('qif', 'fsn'): G['w'][sp] += G['d'][sp]
                    inp[:] = 0
                if forced:
                    for q in G['pops']:
                        if q['name'] in forced: sp = np.union1d(sp, np.asarray(forced[q['name']], dtype=np.int64) + q['off'])
                if len(sp) == 0: continue
                G['count'][sp] += 1
                if rec:
                    for q in G['pops']:
                        if q['name'] in rec:
                            x = sp[(sp >= q['off']) & (sp < q['off'] + q['n'])]
                            if len(x): rec[q['name']].append((s.t, x - q['off']))
                for (ip, slot, w, d), st in G['csr']:
                    c0, c1 = ip[sp], ip[sp + 1]; cnt = c1 - c0
                    if not cnt.any(): continue
                    if len(sp) == 1: ix = slice(int(c0[0]), int(c1[0]))
                    else: ix = np.concatenate([np.arange(a_, b_) for a_, b_ in zip(c0, c1)])
                    ss, ww, dd = slot[ix], w[ix], d[ix]
                    if st is not None:
                        now = s.t * dt; el = now - st['last'][sp]
                        if st['kind'] == 'nmda':                                    # fraction of open channels: decays, jumps by a (1 - s)
                            sj = st['s'][sp] * np.exp(-el / st['tau']); mult = (st['a'] * (1 - sj)).astype(np.float32); st['s'][sp] = sj + mult
                        else:                                                       # short-term facilitation / depression, per sending cell
                            x = 1 - (1 - st['x'][sp]) * np.exp(-el / st['trec'])
                            u = st['U'] + (st['u'][sp] - st['U']) * (np.exp(-el / st['tfac']) if st['tfac'] > 0 else 0.0)
                            mult = (u * x / st['U']).astype(np.float32); st['x'][sp] = x - u * x; st['u'][sp] = u + st['U'] * (1 - u)
                        st['last'][sp] = now; ww = ww * np.repeat(mult, cnt)
                    np.add.at(buf, ((s.t + dd.astype(np.int64)) % L) * S + ss, ww)
            s.t += 1
            if not quiet and it % 2000 == 1999:
                print('  %.0f ms simulated, %.0f s elapsed' % ((it + 1) * dt, time.time() - t0), flush=True)
        rates = {k: float(q['count'].sum()) / q['n'] / (T_ms * 1e-3) for k, q in s.pops.items()}
        return (rates, rec) if record else rates
