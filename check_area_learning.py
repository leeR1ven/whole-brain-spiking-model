"""Does one cortical area of the brain model learn cell groups in layer 2/3, and do they stay active after the input stops?
(area_learn.py: the layered circuit + the plasticity rules reproduced in zenke.py, sliding-threshold variant.)
Protocol as in the reproduced paper: stage 1 - one of 4 inputs (thalamic fibres of one quarter of the sheet) ~1 s on,
~2 s off; stage 2 - 0.2 s on, ~5 s off.     usage: python3 check_area_learning.py [T1=1800] [T2=600] [fibre rate=100] [settings dict]"""
import sys, time, numpy as np
import area_learn as AL, cortex

T1 = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0; T2 = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
R = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
kw = dict(stp=False, a_fast=0.5, exc=dict(hom=True)); kw.update(eval(sys.argv[4]) if len(sys.argv) > 4 else {})
net, info = AL.build(seed=1, learn=True, **kw); rng = np.random.default_rng(5); t0 = time.time(); nq = info['n']['L23e']; q23 = info['quad']['L23e']
print('settings', kw, '| fibres %.0f Hz | cells' % R, info['n'], flush=True)
net.run(500.0); r = net.run(5000.0); print('before: ' + ' '.join('%s %.2f' % (p, r['cx.' + p]) for p in cortex.POPS), flush=True)


def stage(T, on, off, tag, rec_last):
    """returns per-cell counts while each quarter is shown / in the pause after it (from 1 s after switch-off), and the times"""
    t = 0.0; nxt = 60.0; c_on = np.zeros((4, nq)); t_on = np.zeros(4); c_off = np.zeros((4, nq)); t_off = np.zeros(4); pauses = [[] for _ in range(4)]
    a_on = np.zeros(2); a_off = 0.0; k_on = np.zeros(2); k_off = 0
    while t < T:
        p = int(rng.integers(4)); d_on = max(50.0, rng.normal(on, on / 3)); d_off = max(50.0, rng.exponential(off)); rec = t >= T - rec_last
        AL.show(net, info, p, R, d_on); c = net.pops['cx.L23e']['count'].copy(); own = c[q23 == p].mean() / d_on * 1e3; oth = c[q23 != p].mean() / d_on * 1e3
        a_on += (own, oth); k_on += 1
        if rec: c_on[p] += c; t_on[p] += d_on * 1e-3
        if d_off > 2000.0:
            AL.show(net, info, None, 0, 1000.0); c1 = net.pops['cx.L23e']['count'].copy(); AL.show(net, info, None, 0, d_off - 1000.0); c = net.pops['cx.L23e']['count'].copy()
            a_off += (c1.mean() + c.mean()) / d_off * 1e3; k_off += 1
            if rec: c_off[p] += c; t_off[p] += (d_off - 1000.0) * 1e-3; pauses[p].append(c / (d_off - 1000.0) * 1e3)
        else:
            AL.show(net, info, None, 0, d_off); a_off += net.pops['cx.L23e']['count'].mean() / d_off * 1e3; k_off += 1
        t += (d_on + d_off) * 1e-3
        if t >= nxt:
            nxt += 60.0; ff, w1 = AL.weights(net, info, 'L4e->L23e'); rc, w2 = AL.weights(net, info, 'L23e->L23e'); ih, w3 = AL.weights(net, info, 'L23i->L23e')
            print('%s t=%4.0f s | L2/3e: own quarter shown %4.1f Hz, another %4.1f, pauses %4.2f | L4e->L2/3e w %.3f (same quarter %.3f, other %.3f) | L2/3e->L2/3e w %.3f (same %.3f, other %.3f, >0.25: %4.1f %%) | inh w %.3f | %.0f s' % (
                tag, t, a_on[0] / k_on[0], a_on[1] / k_on[1], a_off / max(k_off, 1), w1.mean(), np.diag(ff).mean(), (ff.sum() - np.trace(ff)) / 12, w2.mean(), np.diag(rc).mean(),
                (rc.sum() - np.trace(rc)) / 12, 100 * (w2 > 0.25).mean(), w3.mean(), time.time() - t0), flush=True)
            a_on[:] = 0; k_on[:] = 0; a_off = 0.0; k_off = 0
    return c_on / np.maximum(t_on, 1e-9)[:, None], pauses


def report(ev, pauses, M):
    rest = np.setdiff1d(np.arange(nq), np.concatenate(M)); groups = M + [rest]
    print('  activity (Hz) of each group in the pauses (from 1 s after switch-off to the next input; pauses > 2 s); rows = input shown before:')
    print('  %-9s' % '' + ' '.join('%8s' % ('group %d' % g) for g in range(4)) + '     none   pauses   own group > 10 Hz')
    for p in range(4):
        if not pauses[p]: continue
        x = np.array([[c[g].mean() if len(g) else 0.0 for g in groups] for c in pauses[p]])
        print('  input %d  ' % p + ' '.join('%8.1f' % v for v in x.mean(0)) + '   %5d   %5.0f %%' % (len(x), 100 * (x[:, p] > 10).mean()))


ev, pauses = stage(T1, 1000.0, 2000.0, 'stage 1', 300.0)
M = [np.nonzero(ev[p] > 10.0)[0] for p in range(4)]
print('=== after stage 1 (last 300 s). L2/3e cells coding for each input (rate > 10 Hz while it is shown): ' + ', '.join(str(len(m)) for m in M) +
      '; for none %d of %d; for more than one %d' % ((np.sum(ev > 10, 0) == 0).sum(), nq, (np.sum(ev > 10, 0) > 1).sum()))
print('  of the coding cells, share lying in the shown quarter: ' + ', '.join('%.0f %%' % (100 * (q23[M[p]] == p).mean()) if len(M[p]) else 'nan' for p in range(4)))
print('  mean rate of each group (rows) while each input is shown (columns), Hz:')
for g in range(4): print('  group %d  ' % g + ' '.join('%6.1f' % ev[p, M[g]].mean() if len(M[g]) else '   nan' for p in range(4)))
lab = np.full(nq, 4); only = np.sum(ev > 10, 0) == 1
for p in range(4): lab[M[p][only[M[p]]]] = p
i_, j_, w = AL.syn(net, 'L23e', 'L23e'); la, lb = lab[i_], lab[j_]
print('  mean L2/3e -> L2/3e weight from group (row) to group (column), cells coding for exactly one input; last = none; start 0.100:')
for a in range(5): print('  ' + ('group %d  ' % a if a < 4 else 'none     ') + ' '.join('%6.3f' % w[(la == a) & (lb == b)].mean() if ((la == a) & (lb == b)).any() else '   nan' for b in range(5)))
report(ev, pauses, M)
if T2 > 0:
    ev2, pauses2 = stage(T2, 200.0, 5000.0, 'stage 2', T2)
    print('=== stage 2 (0.2 s inputs, ~5 s pauses), groups as found after stage 1'); report(ev2, pauses2, M)
net.learn = False; G = net.pops['cx.L23e']['grp']; bt, dc = G['btot'].copy(), G['dc'].copy()
print('=== learning off. Each input shown 0.5 s, then ALL background input of the area cut for 3 s: Hz of each group (and of the other cells) in seconds 1-3 after the cut')
for p in range(4):
    AL.show(net, info, None, 0, 2000.0); AL.show(net, info, p, R, 500.0); G['btot'][:] = 0.0; G['dc'][:] = 0.0
    AL.show(net, info, None, 0, 1000.0); AL.show(net, info, None, 0, 2000.0); c = net.pops['cx.L23e']['count'] / 2.0; G['btot'][:] = bt; G['dc'][:] = dc
    print('  input %d: ' % p + ' '.join('%6.1f' % (c[m].mean() if len(m) else 0.0) for m in M) + ' | %6.1f' % c[np.setdiff1d(np.arange(nq), np.concatenate(M))].mean())
AL.show(net, info, None, 0, 2000.0); r = net.run(10000.0)
print('at rest afterwards (learning off, 10 s): ' + ' '.join('%s %.2f' % (p, r['cx.' + p]) for p in cortex.POPS))
np.savez_compressed('data/area_learning_weights_%d.npz' % T1, ev=ev, quad=q23, **{'%s_%s_%s' % (a, b, k): v for a, b in (('L23e', 'L23e'), ('L4e', 'L23e'), ('L23i', 'L23e'))
                                                                         for k, v in zip(('pre', 'post', 'w'), AL.syn(net, a, b))})
print('total %.0f s' % (time.time() - t0))
