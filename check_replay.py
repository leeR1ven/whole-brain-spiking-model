"""After check_sequence.py: does the trained network walk through its groups in the trained order by itself?
The excitatory cells get the paper's second, long-lasting adaptation (decay 20 s), which tires an active group.
usage: python3 check_replay.py TAG [adapt2 = 0.002] [seconds left alone = 200]     (TAG e.g. seq0.2_tp0.15)"""
import sys, os, pickle, numpy as np
import zenke as Z, zenke_ana as A
tag = sys.argv[1]; AD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002; T = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
M, r = A.sets(tag); shapes = Z.load_patterns(Z.CODE + '/shapes.pat'); names = A.NAMES; order = {0: 1, 1: 2, 2: 3, 3: 0}
sim = Z.Sim(1); sim.S = pickle.load(open(os.path.join(A.OUT, tag + '_state.pkl'), 'rb')); Z.seed(11)
def watch(seconds, label, bg=5.0):
    P2 = Z.params(on=0.2, off=1e7, scale=25.0, plastic=0.0, adapt2=AD, bg=bg); s0 = sim.S['step']; re, ri, ev, sp = sim.run(seconds, P2, shapes, record=True, max_spikes=30_000_000); t, i = sp; seq = []
    for k in range(int(seconds)):
        c = A.counts(t, i, s0 + k * 10000, s0 + (k + 1) * 10000); x = [c[g].mean() for g in M]; seq.append(int(np.argmax(x)) if max(x) > 5 else -1)
    runs = []
    for g in seq:
        if runs and runs[-1][0] == g: runs[-1][1] += 1
        else: runs.append([g, 1])
    print('  %-34s' % label + ' -> '.join('%s %ds' % (names[g] if g >= 0 else 'none', n) for g, n in runs), flush=True)
    tr = [(a[0], b[0]) for a, b in zip([x for x in runs if x[0] >= 0][:-1], [x for x in runs if x[0] >= 0][1:])]
    return sum(order[a] == b for a, b in tr), len(tr)
print('%s, slow adaptation %.4f, learning off; group that is most active (> 5 Hz), second by second:' % (tag, AD))
sim.S['sched'][:] = (0.0, 0.0, 1e18, 0.0); ok, n = watch(T, 'left alone, usual background')
print('  changes from one group to another: %d, of which to the NEXT group of the trained order: %d (chance: 1 in 3)' % (n, ok))
for p in range(4):
    sim.S['sched'][:] = (1.0, float(p), sim.S['step'] + 2000, 0.0); watch(60.0, '%s shown 0.2 s, then 60 s' % names[p])
