"""no training: the SAME weights run once with spiking cells and once with rate units (rate_units.py). Do the same cells
light up, is activity held after a cue, is an old group displaced by a new cue? Needs the saved state of the one-level
concept run (check_concept2.py with seq=True, beta 0.01; data/zenke_run/c2w_seq_b01_state.pkl, not in the repository)."""
import os, sys, pickle, time, numpy as np
import levels as L, rate_units as RU
d = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/zenke_run/c2w_seq_b01_state.pkl'), 'rb')); C, ctx, mem = d['C'], d['ctx'], d['mem']; n = C['nsit']
codes = [(s, nm, c) for s in range(n) for nm, c in (('c', ctx[s]), ('a', mem[s][0]), ('b', mem[s][1]))]
for trained in (0, 1):
    net = L.Net(C['seed'], [C['n']], [C['n'] // 4], {0: [(C['K'], C['w']), (C['kweak'], C['wweak'])]}, {(0, 0): (C['ee'], C['wee'])}, alpha=C['alpha'])
    if trained: net.S = d['S']
    P = net.par(bg=C['bg'], plastic=0.0, alpha=C['alpha']); rn = RU.RateNet(net.S, bg=C['bg'], scale=C['scale'])
    A = {}; B = {}; HA = {}; HB = {}; t0 = time.time()
    for s, nm, c in codes:
        net.reset(); net.show(None, 0.3, P); A[(s, nm)] = net.show(c, 1.0, P).copy(); HA[(s, nm)] = net.show(None, 1.0, P).copy()
    t1 = time.time()
    for s, nm, c in codes:
        rn.reset(); rn.show(None, 0.3); B[(s, nm)] = rn.show(c, 1.0); HB[(s, nm)] = rn.show(None, 1.0)
    t2 = time.time()
    def stats(R, H):
        lit = {k: v > 10 for k, v in R.items()}; sh = lambda x, y: (lit[x] & lit[y]).sum() / max(1, min(lit[x].sum(), lit[y].sum()))
        same = np.mean([sh((s, 'a'), (s, 'b')) for s in range(n)]); diff = np.mean([sh((s, x), (t, y)) for s in range(n) for t in range(n) if s != t for x in 'ab' for y in 'ab'])
        cue = np.mean([sh((s, x), (s, 'c')) for s in range(n) for x in 'ab'])
        return np.mean([l.sum() for l in lit.values()]), np.mean([R[k][lit[k]].mean() if lit[k].any() else 0 for k in R]), same, diff, cue, np.mean([H[k][lit[k]].mean() if lit[k].any() else 0 for k in R]), lit
    a = stats(A, HA); b = stats(B, HB)
    ov = np.mean([(a[6][k] & b[6][k]).sum() / max(1, min(a[6][k].sum(), b[6][k].sum())) for k in A]); cr = np.mean([np.corrcoef(A[k], B[k])[0, 1] for k in A])
    print('%s weights:' % ('TRAINED' if trained else 'untrained'))
    for lab, x in (('spiking', a), ('rate   ', b)): print('  %s  cells lit by a code %4.0f at %4.1f Hz | members of one situation share %.2f, of different situations %.2f, member and own cue %.2f | held in the next second %4.1f Hz' % (lab, x[0], x[1], x[2], x[3], x[4], x[5]))
    print('  same cells? of the cells lit in one version, %.2f are lit in the other; correlation of the cells\' rates %.2f | computing time for 27.6 simulated seconds: spiking %.0f s, rate %.0f s' % (ov, cr, t1 - t0, t2 - t1), flush=True)
    if trained:
        G = []
        for s in range(n): G.append(np.any([a[6][(s, x)] for x in 'cab'], 0))
        only = np.sum(G, 0) == 1; G = [np.nonzero(g & only)[0] for g in G]; tr = np.zeros((10, 2)); k = 0
        for s in range(n):
            for t in range(n):
                if s == t: continue
                rn.reset(); rn.show(None, 0.3); rn.show(ctx[s], 1.0); rn.show(mem[s][0], 1.0); rn.show(None, 2.0); _, o = rn.show(ctx[t], 1.0, sets=[G[s], G[t]], every=50); tr += o; k += 1
        tr /= k
        print('  rate version, after the NEXT cue comes on (0.1 s steps): previous group ' + ' '.join('%4.1f' % x for x in tr[:, 0])); print('                                                          new group      ' + ' '.join('%4.1f' % x for x in tr[:, 1]))
        print('  (spiking, measured before:                              previous group 18.8 13.6  9.9  7.5  5.7  4.0  2.6  1.6  0.8  0.4 ; new group 1.3 3.6 5.3 7.0 8.8 10.9 13.0 15.0 17.2 18.5)')
