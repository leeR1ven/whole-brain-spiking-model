"""Steps 4-6 of the designer's account of how concept cells form (IDEAS.md), in spiking cells with the reproduced rules:
'cells that are lit from above while a signal arrives grow forward links from that signal, and afterwards the signal
lights them by itself'. Here the light from above is GIVEN (a 'context' code that is on whenever a member of its
situation is shown - it stands for the returns of a prefrontal loop that holds the situation; the loop itself is the
next step, check_concept3.py).

One level of 4096 excitatory + 1024 inhibitory cells (levels.py); every cell gets K random inputs from the 4096 input
units (no receptive fields). N situations, each with a context code and two member codes (all codes are disjoint random
sets of input units; the two members are never shown together). Training: context + one member for 1 s, pause 2 s.
Test, learning off: every member ALONE (no context). Do the two members of one situation light the same cells,
members of different situations different cells?
usage: python3 check_concept2.py [seconds = 1800] ["dict(...)" settings]"""
import sys, time, pickle, os, numpy as np
import levels as L, zenke as Z

T = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0
C = dict(nsit=4, mc=256, mm=256, K=60, w=1.5, bg=1.0, ctx=True, ee=0.05, hom=0.0, scale=25.0, alpha=4.0, tag='c2'); C.update(eval(sys.argv[2]) if len(sys.argv) > 2 else {})
rng = np.random.default_rng(3); perm = rng.permutation(L.NS); k = 0; ctx, mem = [], []
for s in range(C['nsit']):
    ctx.append(np.sort(perm[k:k + C['mc']])); k += C['mc']; mem.append([np.sort(perm[k:k + C['mm']]), np.sort(perm[k + C['mm']:k + 2 * C['mm']])]); k += 2 * C['mm']
net = L.Net(1, [4096], [1024], {0: (C['K'], C['w'])}, {(0, 0): (C['ee'], 0.1)} if C['ee'] > 0 else {(0, 0): (0.0005, 0.1)}, alpha=C['alpha'])
P = net.par(bg=C['bg'], hom=C['hom'], scale=C['scale'], alpha=C['alpha']); Poff = net.par(bg=C['bg'], plastic=0.0, scale=C['scale'], alpha=C['alpha'])   # alpha: target mean rate (Hz) of the inhibitory plasticity
print('settings', C, flush=True)


def test(label):
    """every code alone, 3 x 1 s, learning off. Cells counted as lit: rate > 10 Hz."""
    R = {}
    for s in range(C['nsit']):
        for name, code in (('c', ctx[s]), ('a', mem[s][0]), ('b', mem[s][1])):
            r = np.zeros(net.ne)
            for _ in range(3):
                net.show(None, 1.0, Poff); r += net.show(code, 1.0, Poff) / 3
            R[(s, name)] = r
    lit = {k_: set(np.nonzero(v > 10)[0]) for k_, v in R.items()}; sh = lambda x, y: len(lit[x] & lit[y]) / max(1, min(len(lit[x]), len(lit[y])))
    n = C['nsit']; same = np.mean([sh((s, 'a'), (s, 'b')) for s in range(n)]); diff = np.mean([sh((s, x), (t, y)) for s in range(n) for t in range(n) if s != t for x in 'ab' for y in 'ab'])
    cue = np.mean([sh((s, x), (s, 'c')) for s in range(n) for x in 'ab']); cued = np.mean([sh((s, x), (t, 'c')) for s in range(n) for t in range(n) if s != t for x in 'ab'])
    cc = lambda x, y: float(np.corrcoef(R[x], R[y])[0, 1]) if R[x].std() > 0 and R[y].std() > 0 else 0.0
    print('%s | cells lit by a member alone: %.0f (rate %.1f Hz), by a context alone: %.0f | shared by the two members of ONE situation %.2f, by members of DIFFERENT situations %.2f | member and its own context %.2f, another context %.2f | correlation of the cells\' rates: same situation %.2f, different %.2f' % (
        label, np.mean([len(lit[(s, x)]) for s in range(n) for x in 'ab']), np.mean([R[(s, x)][list(lit[(s, x)])].mean() if lit[(s, x)] else 0 for s in range(n) for x in 'ab']),
        np.mean([len(lit[(s, 'c')]) for s in range(n)]), same, diff, cue, cued, np.mean([cc((s, 'a'), (s, 'b')) for s in range(n)]),
        np.mean([cc((s, 'a'), (t, 'b')) for s in range(n) for t in range(n) if s != t])), flush=True)
    return R


test('before training      ')
t = 0.0; t0 = time.time(); nxt = 300.0; acc = []
while t < T - 1e-9:
    s = int(rng.integers(C['nsit'])); j = int(rng.integers(2)); units = np.concatenate([ctx[s], mem[s][j]]) if C['ctx'] else mem[s][j]
    r = net.show(units, 1.0, P); acc.append(((r > 10).mean(), r.mean())); net.show(None, 2.0, P); t += 3.0
    if t >= nxt:
        a = np.array(acc); acc = []; nxt += 300.0; S = net.S
        print('t=%5.0f s | while a pair is shown: %.1f %% of the cells above 10 Hz, mean rate %.1f Hz | weights: input %.3f (max %.2f), E-E %.3f, I-E %.3f | %.0f s' % (
            t, 100 * a[:, 0].mean(), a[:, 1].mean(), S['se_ws'][:, 0].mean(), S['se_ws'][:, 0].max(), S['ee_ws'][:, 0].mean(), S['ie_w'].mean(), time.time() - t0), flush=True)
        test('  test at %5.0f s     ' % t)
R = test('after training       ')
pickle.dump(dict(S=net.S, C=C, ctx=ctx, mem=mem), open(os.path.join(Z.HERE, 'data/zenke_run/%s_state.pkl' % C['tag']), 'wb'), protocol=4)
