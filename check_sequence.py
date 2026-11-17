"""Does the learning network (zenke.py) learn the ORDER of its inputs?  Same training as stage 1 of the reproduced paper
(shapes ~1 s on, ~2 s off, 30 min) but the four shapes always come in the same order circle -> square -> cross -> triangle.
Afterwards: are the synapses from a group to the NEXT group stronger than to the previous one, and does the activity
move on to the next group by itself?   (This test is mine; the paper only shows shapes in random order.)
usage: python3 check_sequence.py [seconds=1800] [gap between shapes in s = 2.0] [time window of the sending trace in s = 0.02] [duration of each shape in s = 1.0]"""
import sys, os, pickle, time, numpy as np
import zenke as Z, zenke_ana as A
T = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0; OFF = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
TP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02           # time window (s) of the sending cell's trace in the potentiation (paper: 0.02)
ON = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0             # how long each shape is shown (s)
tag = 'seq%g' % OFF + ('' if TP == 0.02 else '_tp%g' % TP) + ('' if ON == 1.0 else '_on%g' % ON); OUT = A.OUT; shapes = Z.load_patterns(Z.CODE + '/shapes.pat'); names = A.NAMES
sim = Z.Sim(1); P = Z.params(on=ON, off=OFF, scale=25.0, seq=1.0, tau_pre=TP); t_done = 0.0; spikes = []; evs = []; rec_last = min(300.0, T)
while t_done < T - 1e-9:
    chunk = min(60.0, T - t_done); rec = t_done >= T - rec_last - 1e-9; t0 = time.time()
    re, ri, ev, sp = sim.run(chunk, P, shapes, record=rec, max_spikes=10_000_000); t_done += chunk; S = sim.S; evs.append(ev)
    if rec: spikes.append(sp)
    print('t=%5.0f s  E %5.2f Hz  I %5.2f Hz | E-E w %.3f (>0.25: %4.1f %%) | %.0f s' % (S['step'] * Z.DT, re, ri, S['ee_ws'][:, 0].mean(), 100 * (S['ee_ws'][:, 0] > 0.25).mean(), time.time() - t0), flush=True)
np.savez_compressed(os.path.join(OUT, tag + '_init_spikes.npz'), t=np.concatenate([s[0] for s in spikes]), i=np.concatenate([s[1] for s in spikes]), ev=np.concatenate(evs), t_end=sim.S['step'])
sim.save(os.path.join(OUT, tag + '_init_weights.npz')); pickle.dump(sim.S, open(os.path.join(OUT, tag + '_state.pkl'), 'wb'), protocol=4)
M, r = A.sets(tag); S = sim.S; pre, post, w = S['ee'][5], S['ee'][1], S['ee_ws'][:, 0]
print('=== cells coding for circle, square, cross, triangle: ' + ', '.join(str(len(m)) for m in M) + '; for more than one %d' % (np.sum(r > 10, 0) > 1).sum())
lab = np.full(Z.NE, 4); only = np.sum(r > 10, 0) == 1
for p in range(4): lab[M[p][only[M[p]]]] = p
W = np.array([[w[(lab[pre] == a) & (lab[post] == b)].mean() if ((lab[pre] == a) & (lab[post] == b)).any() else np.nan for b in range(4)] for a in range(4)])
print('mean weight from group (row) to group (column); cells coding for exactly one shape; start 0.100:')
for a in range(4): print('  %-9s' % names[a] + ' '.join('%6.3f' % x for x in W[a]))
print('to the NEXT group in the order: ' + ' '.join('%.3f' % W[a, (a + 1) % 4] for a in range(4)) + '   to the PREVIOUS one: ' + ' '.join('%.3f' % W[a, (a - 1) % 4] for a in range(4)))
groups = M + [np.setdiff1d(np.arange(Z.NE), np.concatenate(M))]
def watch(seconds, label):
    P2 = Z.params(on=0.2, off=1e7, scale=25.0, plastic=0.0); s0 = sim.S['step']; re, ri, ev, sp = sim.run(seconds, P2, shapes, record=True); t, i = sp; out = []
    for k in range(int(seconds)):
        c = A.counts(t, i, s0 + k * 10000, s0 + (k + 1) * 10000); x = [c[g].mean() for g in groups[:4]]; out.append(names[int(np.argmax(x))][:3] + '%2.0f' % max(x) if max(x) > 5 else ' -   ')
    print('  %-32s' % label + ' '.join(out), flush=True)
print('no shapes any more, only the usual background; most active group in every second (name + Hz, - = none above 5 Hz):')
sim.S['sched'][:] = (0.0, 0.0, 1e18, 0.0); watch(40.0, 'left alone, 40 s')
for p in range(4):
    sim.S['sched'][:] = (1.0, float(p), sim.S['step'] + 2000, 0.0); watch(20.0, '%s shown 0.2 s, then 20 s' % names[p])
