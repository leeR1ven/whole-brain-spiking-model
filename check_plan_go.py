"""no training: 'plan x go' in a saved TWO-level state. A cue is shown for 1 s and switched off; if its group stays
active in level 1, its returns (capped: they cannot light a cell alone) pre-activate cells in level 0 = the 'plan'.
Then every level-0 cell gets the same extra input for 0.3 s (the rate of ALL input units is raised) = 'go'.
Which level-0 cells fire: those of the held situation, or also those of the other situations?
usage: python3 plan_go2.py state.pkl"""
import sys, pickle, numpy as np
import levels as L
d = pickle.load(open(sys.argv[1], 'rb')); C, cue, mem = d['C'], d['cue'], d['mem']; n = C['nsit']
ee = {(0, 1): [(C['k01'] / C['n0'], C['w01']), (C['k01weak'] / C['n0'], C['w01weak'])], (1, 1): (C['p11'], C['w11']), (1, 0): (C['k10weak'] / C['n1'], C['w10weak'])}
net = L.Net(C['seed'], [C['n0'], C['n1']], [C['n0'] // 4, C['n1'] // 4], {0: [(C['K'], C['w']), (C['kweak'], C['wweak'])]}, ee, alpha=C['alpha']); net.S = d['S']
par = lambda bg: net.par(plastic=0.0, bg=bg, alpha=C['alpha'], beta=C['beta'], wmax2=C.get('wret', 0) or 5.0); P = par(C['bg']); l0, l1 = net.level(0), net.level(1)
ws = net.S['ee_ws'][:, 0]; kret, rpre, rpost = net.block(1, 0); print('returns: mean weight %.3f, max %.3f; sum onto a level-0 cell: mean %.1f' % (ws[kret].mean(), ws[kret].max(), np.bincount(rpost, weights=ws[kret], minlength=net.ne)[l0].mean()))
lit0 = {}; lit1 = {}
for s in range(n):
    for name, code in (('c', cue[s]), ('a', mem[s][0]), ('b', mem[s][1])):
        net.reset(); net.show(None, 0.3, P); r = net.show(code, 1.0, P); lit0[(s, name)] = r[l0] > 10; lit1[(s, name)] = r[l1] > 10
own = [lit0[(s, 'c')] | lit0[(s, 'a')] | lit0[(s, 'b')] for s in range(n)]; memonly = [(lit0[(s, 'a')] | lit0[(s, 'b')]) & ~lit0[(s, 'c')] for s in range(n)]
other = [np.any([own[t] for t in range(n) if t != s], 0) & ~own[s] for s in range(n)]; rest = ~np.any(own, 0); g1 = [lit1[(s, 'c')] | lit1[(s, 'a')] | lit1[(s, 'b')] for s in range(n)]
print('level-0 cells per situation %.0f (of them lit by a member but not by the cue %.0f), of other situations %.0f, of none %.0f | level-1 cells per situation %.0f' % (np.mean([o.sum() for o in own]), np.mean([m.sum() for m in memonly]), np.mean([o.sum() for o in other]), rest.sum(), np.mean([g.sum() for g in g1])))
print('input rate | level 1, held group Hz | level 0: cells of the held situation Hz, share > 10 Hz | its member-only cells Hz, share | other situations Hz, share | of none Hz')
for bg in (C['bg'], 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    r = np.zeros((n, 8))
    for s in range(n):
        net.reset(); net.show(None, 0.3, P); net.show(cue[s], 1.0, P); net.show(None, 0.5, P); x = net.show(None, 0.3, par(bg)); a, b = x[l0], x[l1]
        r[s] = (b[g1[s]].mean(), a[own[s]].mean(), (a[own[s]] > 10).mean(), a[memonly[s]].mean() if memonly[s].any() else 0, (a[memonly[s]] > 10).mean() if memonly[s].any() else 0, a[other[s]].mean(), (a[other[s]] > 10).mean(), a[rest].mean())
    m = r.mean(0); print('  %4.0f Hz | %5.1f | %5.1f  %.2f | %5.1f  %.2f | %5.1f  %.2f | %5.1f' % ((bg,) + tuple(m)), flush=True)
print('control, nothing held (no cue before):')
for bg in (3.0, 5.0):
    net.reset(); net.show(None, 1.8, P); x = net.show(None, 0.3, par(bg))[l0]; print('  %4.0f Hz | cells of the situations %.1f Hz, share > 10 Hz %.2f | of none %.1f Hz' % (bg, x[~rest].mean(), (x[~rest] > 10).mean(), x[rest].mean()), flush=True)
