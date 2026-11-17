"""The designer's account of how concept cells form (IDEAS.md, six steps), with a real loop above the concept level:

    input units  --random, plastic-->  level 0 (concept level)  --random, plastic-->  level 1 (prefrontal loop)
                                             ^-------------- returns (random, plastic) --------------'
Cells, synapses and learning rules are those of the reproduced network (zenke.py); the wiring is in levels.py.
N situations: a cue code, then at once one of its two member codes (the members are never shown together), then a pause.
Test, learning off, every code alone after a reset: do the two members of one situation light the same level-0 cells?
(also with the returns cut, to see what the FORWARD links alone do).
usage: python3 check_concept3.py [seconds = 1800] ["dict(...)" settings]"""
import sys, time, pickle, os, numpy as np
import levels as L, zenke as Z

T = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0
C = dict(nsit=4, m=512, K=60, w=1.5, bg=1.0, n0=4096, n1=2048, k01=150, w01=3.0, p11=0.25, w11=0.02, k10=150, w10=0.3, p00=0.0, tc=1.0, tm=1.0, gap=2.0, wmax=5.0, hom=0.0, scale=25.0, tag='c3')
C.update(eval(sys.argv[2]) if len(sys.argv) > 2 else {})
rng = np.random.default_rng(3); perm = rng.permutation(L.NS); k = 0; cue, mem = [], []; m = C['m']
for s in range(C['nsit']):
    cue.append(np.sort(perm[k:k + m])); k += m; mem.append([np.sort(perm[k:k + m]), np.sort(perm[k + m:k + 2 * m])]); k += 2 * m
ee = {(0, 1): (C['k01'] / C['n0'], C['w01']), (1, 1): (C['p11'], C['w11']), (1, 0): (C['k10'] / C['n1'], C['w10'])}
if C['p00'] > 0: ee[(0, 0)] = (C['p00'], 0.1)
net = L.Net(1, [C['n0'], C['n1']], [C['n0'] // 4, C['n1'] // 4], {0: (C['K'], C['w'])}, ee)
P = net.par(bg=C['bg'], wmax=C['wmax'], hom=C['hom'], scale=C['scale']); Poff = net.par(bg=C['bg'], plastic=0.0, wmax=C['wmax'], scale=C['scale']); l0, l1 = net.level(0), net.level(1); n = C['nsit']
kb = {b: net.block(*b)[0] for b in ee}
print('settings', C, '| synapses: input %d, ' % len(net.S['se_ws']) + ', '.join('%d->%d %d' % (a, b, len(kb[(a, b)])) for a, b in ee), flush=True)


def test(label, cut=False):
    """every code alone for 1 s after a reset, learning off; lit = rate > 10 Hz. cut: the returns 1 -> 0 are switched off."""
    ws = net.S['ee_ws']; keep = ws[kb[(1, 0)], 0].copy()
    if cut: ws[kb[(1, 0)], 0] = 0.0
    R = {}
    for s in range(n):
        for name, code in (('c', cue[s]), ('a', mem[s][0]), ('b', mem[s][1])):
            net.reset(); net.show(None, 0.3, Poff); R[(s, name)] = net.show(code, 1.0, Poff).copy()
    ws[kb[(1, 0)], 0] = keep; net.reset(); out = []
    for lv, sl in ((0, l0), (1, l1)):
        lit = {k_: set(np.nonzero(v[sl] > 10)[0]) for k_, v in R.items()}; sh = lambda x, y: len(lit[x] & lit[y]) / max(1, min(len(lit[x]), len(lit[y])))
        out.append('level %d: lit by a member %.0f, by a cue %.0f; shared by the members of ONE situation %.2f, of DIFFERENT situations %.2f; member and its own cue %.2f, another cue %.2f' % (
            lv, np.mean([len(lit[(s, x)]) for s in range(n) for x in 'ab']), np.mean([len(lit[(s, 'c')]) for s in range(n)]), np.mean([sh((s, 'a'), (s, 'b')) for s in range(n)]),
            np.mean([sh((s, x), (t, y)) for s in range(n) for t in range(n) if s != t for x in 'ab' for y in 'ab']), np.mean([sh((s, x), (s, 'c')) for s in range(n) for x in 'ab']),
            np.mean([sh((s, x), (t, 'c')) for s in range(n) for t in range(n) if s != t for x in 'ab'])))
    print(label + (' (returns cut)' if cut else '') + ' | ' + ' | '.join(out), flush=True)


test('before training   '); t = 0.0; t0 = time.time(); nxt = 300.0; acc = []
while t < T - 1e-9:
    s = int(rng.integers(n)); j = int(rng.integers(2))
    rc = net.show(cue[s], C['tc'], P).copy(); rm = net.show(mem[s][j], C['tm'], P).copy(); rp = net.show(None, C['gap'], P).copy(); t += C['tc'] + C['tm'] + C['gap']
    c1 = rc[l1] > 10                                                   # level-1 cells lit by the cue: do they stay on while the member is shown, and in the pause?
    acc.append(((rc[l0] > 10).mean(), (rc[l1] > 10).mean(), rm[l1][c1].mean() if c1.any() else 0.0, rp[l1][c1].mean() if c1.any() else 0.0, (rm[l0] > 10).mean(), rp[l0].mean(), rp[l1].mean()))
    if t >= nxt:
        a = np.array(acc).mean(0); acc = []; nxt += 300.0; ws = net.S['ee_ws'][:, 0]
        print('t=%5.0f s | cue: %.1f %% of level 0 and %.1f %% of level 1 lit; the level-1 cells lit by the cue fire %.1f Hz while the member is shown, %.1f Hz in the pause; member: %.1f %% of level 0 lit; pause: level 0 %.2f, level 1 %.2f Hz | weights: input %.2f, ' % (
            t, 100 * a[0], 100 * a[1], a[2], a[3], 100 * a[4], a[5], a[6], net.S['se_ws'][:, 0].mean()) + ', '.join('%d->%d %.3f (max %.1f)' % (x, y, ws[kb[(x, y)]].mean(), ws[kb[(x, y)]].max()) for x, y in ee) +
              ', I-E %.2f | %.0f s' % (net.S['ie_w'].mean(), time.time() - t0), flush=True)
        test('  test at %5.0f s  ' % t); test('  test at %5.0f s  ' % t, cut=True)
test('after training    '); test('after training    ', cut=True)
pickle.dump(dict(S=net.S, C=C, cue=cue, mem=mem), open(os.path.join(Z.HERE, 'data/zenke_run/%s_state.pkl' % C['tag']), 'wb'), protocol=4)
