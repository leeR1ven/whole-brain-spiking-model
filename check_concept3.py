"""Two levels, the designer's structure in spiking cells (IDEAS.md; cells and rules of the reproduced network, wiring in
levels.py):
    input units --> level 0 (sense level: no loop of its own) --> level 1 (concept / prefrontal level: has a loop)
                          ^------------------ returns (weak at first, plastic) ------------------'
Every forward step is random and convergent: each cell gets a few strong and many weak plastic links from the level below.
N situations: a cue code, then one of its two member codes ALONE (the two members never come together), then a pause.
Test, learning off, every code alone after a reset: cells lit (rate > 10 Hz) in each level; do the two members of one
situation light the same cells - in level 1 (expected: yes), in level 0 (expected: no, the sense level keeps them apart)?
Settings that differ from the paper's network (each found in the one-level experiments, MODEL.md): sliding-threshold
variant of the rule, heterosynaptic term beta = 0.01, inhibitory target rate 0.5 Hz, background 1 Hz.
Two options added on 2026-11-18 after the first staged run locked up (MODEL.md):
  wret  ceiling for every return synapse (0 = none). Reason: without it the returns onto a level-0 cell grew to a sum of
        124 (16 light a cell), the two levels kept each other active and a new cue could not get in. With the ceiling
        the returns of a held group stay below what lights a cell by itself (Hawkins & Ahmad 2016: feedback only
        depolarises, the feed-forward input fires the cell).
  stay  probability that the next trial belongs to the SAME situation (0 = the situation is drawn anew every trial):
        input that changes slowly, as the designer pointed out, instead of unrelated items following each other.
usage: python3 check_concept3.py [seconds = 3600] ["dict(...)" settings]"""
import sys, time, pickle, os, numpy as np
import levels as L, zenke as Z

T = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
C = dict(nsit=4, m=256, n0=4096, n1=4096, K=60, w=1.5, kweak=350, wweak=0.05, k01=150, w01=1.0, k01weak=350, w01weak=0.05, p11=0.05, w11=0.1, k10weak=350, w10weak=0.05,
         bg=1.0, alpha=0.5, beta=0.01, hom=1.0, seed=1, first=0.0, wret=0.0, stay=0.0, tag='c3')
C.update(eval(sys.argv[2]) if len(sys.argv) > 2 else {})
rng = np.random.default_rng(3 + 10 * (C['seed'] - 1)); perm = rng.permutation(L.NS); k = 0; cue, mem = [], []; m = C['m']; n = C['nsit']
for s in range(n):
    cue.append(np.sort(perm[k:k + m])); k += m; mem.append([np.sort(perm[k:k + m]), np.sort(perm[k + m:k + 2 * m])]); k += 2 * m
ee = {(0, 1): [(C['k01'] / C['n0'], C['w01']), (C['k01weak'] / C['n0'], C['w01weak'])], (1, 1): (C['p11'], C['w11']), (1, 0): (C['k10weak'] / C['n1'], C['w10weak'])}
net = L.Net(C['seed'], [C['n0'], C['n1']], [C['n0'] // 4, C['n1'] // 4], {0: [(C['K'], C['w']), (C['kweak'], C['wweak'])]}, ee, alpha=C['alpha'])
kw = dict(bg=C['bg'], alpha=C['alpha'], beta=C['beta'], wmax2=C['wret'] if C['wret'] > 0 else 5.0); P = net.par(hom=C['hom'], **kw); Poff = net.par(plastic=0.0, **kw); l0, l1 = net.level(0), net.level(1)
kret = net.block(1, 0)[0]; PL = 2 if C['wret'] > 0 else 1                  # 2: synapses with the lower ceiling
print('settings', C, flush=True)


def test(label, cut=False):
    """cut: the returns from level 1 to level 0 are switched off for the test"""
    ws = net.S['ee_ws']; keep = ws[kret, 0].copy()
    if cut: ws[kret, 0] = 0.0
    R = {}; H = {}
    for s in range(n):
        for name, code in (('c', cue[s]), ('a', mem[s][0]), ('b', mem[s][1])):
            r = np.zeros(net.ne); h = np.zeros(net.ne)
            for _ in range(2):
                net.reset(); net.show(None, 0.3, Poff); r += net.show(code, 1.0, Poff) / 2; h += net.show(None, 1.0, Poff) / 2
            R[(s, name)] = r; H[(s, name)] = h
    ws[kret, 0] = keep; net.reset(); out = []; res = []
    for lv, sl in ((0, l0), (1, l1)):
        lit = {k_: set(np.nonzero(v[sl] > 10)[0]) for k_, v in R.items()}; sh = lambda x, y: len(lit[x] & lit[y]) / max(1, min(len(lit[x]), len(lit[y])))
        held = np.mean([H[k_][sl][list(lit[k_])].mean() if lit[k_] else 0.0 for k_ in R])
        res.append(np.mean([sh((s, x), (t, y)) for s in range(n) for t in range(n) if s != t for x in 'ab' for y in 'ab']))
        out.append('level %d: lit by a member %.0f, by a cue %.0f; members of ONE situation share %.2f, of DIFFERENT situations %.2f; member and its own cue %.2f, another cue %.2f; held %.1f Hz' % (
            lv, np.mean([len(lit[(s, x)]) for s in range(n) for x in 'ab']), np.mean([len(lit[(s, 'c')]) for s in range(n)]), np.mean([sh((s, 'a'), (s, 'b')) for s in range(n)]),
            np.mean([sh((s, x), (t, y)) for s in range(n) for t in range(n) if s != t for x in 'ab' for y in 'ab']), np.mean([sh((s, x), (s, 'c')) for s in range(n) for x in 'ab']),
            np.mean([sh((s, x), (t, 'c')) for s in range(n) for t in range(n) if s != t for x in 'ab']), held))
    print(label + (' (returns cut)' if cut else '') + ' | ' + ' | '.join(out), flush=True)
    return res


# first: for this many seconds only level 0 learns (all links into, inside and out of level 1 are kept fixed) - the lower level first, then the upper one
upper = np.concatenate([net.block(0, 1)[0], net.block(1, 1)[0], kret]); net.S['ee_pl'][upper] = 0 if C['first'] > 0 else 1
if C['first'] <= 0: net.S['ee_pl'][kret] = PL
save = lambda name: pickle.dump(dict(S=net.S, C=C, cue=cue, mem=mem), open(os.path.join(Z.HERE, 'data/zenke_run/%s_state%s.pkl' % (C['tag'], name)), 'wb'), protocol=4)
test('before training'); t = 0.0; t0 = time.time(); nxt = 600.0; s = int(rng.integers(n)); bad = 0
while t < T - 1e-9:
    if C['stay'] <= 0: s = int(rng.integers(n))
    elif rng.random() > C['stay']: s = int((s + 1 + rng.integers(n - 1)) % n)             # leave for one of the other situations
    j = int(rng.integers(2)); net.show(cue[s], 1.0, P); net.show(mem[s][j], 1.0, P); net.show(None, 2.0, P); t += 4.0
    if C['first'] > 0 and t >= C['first'] and net.S['ee_pl'][upper[0]] == 0:
        save('_first'); net.S['ee_pl'][upper] = 1; net.S['ee_pl'][kret] = PL; print('--- from here on level 1 learns too', flush=True)
    if t >= nxt:
        nxt += 600.0; ws = net.S['ee_ws'][:, 0]
        print('t=%5.0f s | weights: input %.3f, 0->1 %.3f, 1->1 %.3f, 1->0 %.3f, I-E %.3f | %.0f s' % (t, net.S['se_ws'][:, 0].mean(), ws[net.block(0, 1)[0]].mean(), ws[net.block(1, 1)[0]].mean(), ws[kret].mean(), net.S['ie_w'].mean(), time.time() - t0), flush=True)
        res = test('  test at %5.0f s' % t); test('  test at %5.0f s' % t, cut=True); bad = bad + 1 if res[1] > 0.4 else 0
        if abs(t - 3000.0) < 1.0: save('_3000')
        if bad >= 2: print('stopped: the situations have merged in level 1 (two tests in a row above 0.4)', flush=True); break
save('')
