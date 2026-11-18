"""Does the learning network come to pass its activity on to the NEXT item of a learned order by itself?
(the designer's loop: what is held activates what usually comes next - IDEAS.md, section 4)

Reproduced network (zenke.py) with three settings changed, each for a stated reason:
  beta = 0.01      the heterosynaptic term at 1/5 of the paper's value (at 0.05 it erases links between cells that do
                   not fire together - found in the concept experiments, MODEL.md)
  tau_pre = 0.15   the sending cell's trace in the potentiation lasts 150 ms instead of 20 ms, so that 'A just before B'
                   strengthens A -> B (Tully et al. 2016; Maes et al. 2020: order is learned only inside this window)
  adapt2           the paper's second, long-lasting adaptation (a group that has been active for seconds tires)
Four inputs that do not overlap (two 16 x 16 patches each). Training: the four inputs one after the other, 1 s each with
0.2 s between them, then a pause of 8 s (so that the mean activity stays as in the paper's protocol), again and again.
Test, learning off: only the FIRST input is shown briefly; which group is active second by second afterwards?
usage: python3 check_order.py [seconds = 3600] [adaptation jump for the test = 0.002]"""
import sys, os, time, pickle, numpy as np
import zenke as Z, zenke_ana as A

T = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0; AD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
X = eval(sys.argv[3]) if len(sys.argv) > 3 else {}; tag = 'order' + ''.join('_%s%g' % kv for kv in X.items())
def patch(k):
    cx, cy = (10, 32, 54)[k % 3], (10, 32, 54)[k // 3]; return np.array([x + 64 * y for y in range(cy - 8, cy + 8) for x in range(cx - 8, cx + 8)], dtype=np.int64)
items = [np.sort(np.concatenate([patch(2 * k), patch(2 * k + 1)])) for k in range(4)]; names = ['one', 'two', 'three', 'four']
sim = Z.Sim(1); S = sim.S; kw = dict(beta=0.01, tau_pre=0.15); kw.update(X); P = Z.params(on=1.0, off=1e7, **kw)


def show(k, seconds, P, record=False):
    """item k (None: only the background) for `seconds`; returns the rate of every E cell"""
    S['sched'][:] = (0.0 if k is None else 1.0, 0.0 if k is None else float(k), 1e18, 0.0)
    sim.run(seconds, P, items, record=False); return sim.cntE / seconds


t = 0.0; t0 = time.time(); nxt = 600.0; ev = np.zeros((4, Z.NE)); nev = 0
while t < T - 1e-9:
    for k in range(4):
        r = show(k, 1.0, P); show(None, 0.2, P)
        if t >= T - 300.0: ev[k] += r
    if t >= T - 300.0: nev += 1
    show(None, 8.0, P); t += 12.8
    if t >= nxt:
        nxt += 600.0; print('t=%5.0f s | E-E weight %.3f, input weight %.3f, I-E weight %.3f | %.0f s' % (t, S['ee_ws'][:, 0].mean(), S['se_ws'][:, 0].mean(), S['ie_w'].mean(), time.time() - t0), flush=True)
ev /= max(nev, 1); M = [np.nonzero(ev[k] > 10.0)[0] for k in range(4)]; only = np.sum(ev > 10, 0) == 1; lab = np.full(Z.NE, 4)
for k in range(4): lab[M[k][only[M[k]]]] = k
pre, post, w = S['ee'][5], S['ee'][1], S['ee_ws'][:, 0]
W = np.array([[w[(lab[pre] == a) & (lab[post] == b)].mean() if ((lab[pre] == a) & (lab[post] == b)).any() else np.nan for b in range(4)] for a in range(4)])
print('cells coding for the four inputs: ' + ', '.join(str(len(m)) for m in M) + '; for more than one %d' % (np.sum(ev > 10, 0) > 1).sum())
print('mean weight from group (row) to group (column):'); [print('  %-6s' % names[a] + ' '.join('%6.3f' % x for x in W[a])) for a in range(4)]
print('to the NEXT group: ' + ' '.join('%.3f' % W[a, a + 1] for a in range(3)) + '   to the PREVIOUS one: ' + ' '.join('%.3f' % W[a + 1, a] for a in range(3)))
pickle.dump(dict(S=S, M=M), open(os.path.join(A.OUT, tag + '_state.pkl'), 'wb'), protocol=4)


def watch(first, seconds, P):
    out = []; sim.S['E'][6][:] = 0.0
    if first is not None: show(first, 0.5, P)
    for _ in range(int(seconds)):
        r = show(None, 1.0, P); x = [r[m].mean() for m in M]; out.append('%s%2.0f' % (names[int(np.argmax(x))][:3], max(x)) if max(x) > 5 else ' -   ')
    return ' '.join(out)


for ad in (0.0, AD, 2 * AD):
    Pt = Z.params(on=1.0, off=1e7, plastic=0.0, adapt2=ad, **kw)
    print('learning off, adaptation jump %.4f. Most active group every second (name + Hz; - = none above 5 Hz):' % ad)
    for first in (0, 1, None):
        show(None, 3.0, Z.params(on=1.0, off=1e7, plastic=0.0, bg=0.0, **kw))         # everything off for 3 s: back to rest
        print('  %-22s' % ('"%s" shown 0.5 s:' % names[first] if first is not None else 'nothing shown:') + watch(first, 40, Pt), flush=True)
