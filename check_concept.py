"""Test of the designer's hypothesis of how 'concept cells' form (memory: concept-formation-hypothesis), in the reproduced
learning network (zenke.py, unchanged rules): cells that the loop keeps re-activating become linked to whatever inputs
are active at the same time, so that different forms of the same thing come to drive the same cells.

Inputs: the 64 x 64 input field carries 9 separate patches (16 x 16 units). Two 'things', each with two 'forms':
    A1 = shared patch SA + patch UA1        A2 = SA + UA2        B1 = SB + UB1        B2 = SB + UB2
A cell whose inputs lie in UA1 gets NO direct input when A2 is shown. Question: after training, does it fire when A2 is
shown (and not when B is shown)?  Run once with the loop among the excitatory cells and once with it switched off.
usage: python3 check_concept.py [loop = 1 | 0] [seconds = 1800]"""
import sys, os, time, pickle, numpy as np
import zenke as Z, zenke_ana as A
LOOP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; T = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
tag = 'concept_loop%d' % LOOP; C0 = (10, 32, 54)
def patch(k):
    cx, cy = C0[k % 3], C0[k // 3]; return np.array([x + 64 * y for y in range(cy - 8, cy + 8) for x in range(cx - 8, cx + 8)], dtype=np.int64)
PN = ['SA', 'UA1', 'UA2', 'SB', 'UB1', 'UB2', 'spare1', 'spare2', 'spare3']; pat = [patch(k) for k in range(9)]
stim = [np.sort(np.concatenate([pat[0], pat[1]])), np.sort(np.concatenate([pat[0], pat[2]])), np.sort(np.concatenate([pat[3], pat[4]])), np.sort(np.concatenate([pat[3], pat[5]]))]
SN = ['A1', 'A2', 'B1', 'B2']
sim = Z.Sim(1); S = sim.S; pre, post = S['se'][5], S['se'][1]
which = np.full(Z.NS, -1); 
for k in range(9): which[pat[k]] = k
cnt = np.zeros((Z.NE, 10)); np.add.at(cnt, (post, which[pre] + 1), 1.0)            # inputs of every E cell from: no patch, patch 0..8
cells = [np.nonzero((cnt[:, k + 1] >= 100) & (cnt[:, 1:].sum(1) - cnt[:, k + 1] == 0))[0] for k in range(9)]
print('loop among excitatory cells: %s. Cells whose inputs come from one patch only (>= 100 inputs from it, none from the other patches): ' % ('ON' if LOOP else 'OFF') +
      ', '.join('%s %d' % (PN[k], len(cells[k])) for k in range(9)), flush=True)
P = Z.params(on=1.0, off=2.0, scale=25.0, ee_on=LOOP); t_done = 0.0; spikes = []; evs = []; rec_last = min(300.0, T)
def table(rate, title):
    print(title); print('  %-22s' % 'cells with inputs in' + ' '.join('%6s' % n for n in SN))
    for k in range(9):
        if len(cells[k]): print('  %-22s' % PN[k] + ' '.join('%6.1f' % rate[p, cells[k]].mean() for p in range(4)))
while t_done < T - 1e-9:
    chunk = min(60.0, T - t_done); rec = t_done >= T - rec_last - 1e-9 or t_done < 60.0; t0 = time.time()
    re, ri, ev, sp = sim.run(chunk, P, stim, record=rec, max_spikes=10_000_000); t_done += chunk
    if rec and t_done > 60.0: spikes.append(sp); evs.append(ev)
    elif rec:                                                        # the first minute: the responses before anything is learned
        t, i = sp; on, off = A.intervals(ev, 0, S['step']); c = np.zeros((4, Z.NE)); tt = np.zeros(4)
        for a, b, p in on: c[p] += A.counts(t, i, a, b); tt[p] += (b - a) * Z.DT
        table(c / np.maximum(tt, 1e-9)[:, None], 'first minute - rate (Hz) while each input is shown:')
    if int(t_done) % 300 == 0: print('t=%5.0f s  E %5.2f Hz  I %5.2f Hz | E-E w %.3f  in-E w %.3f | %.0f s' % (S['step'] * Z.DT, re, ri, S['ee_ws'][:, 0].mean(), S['se_ws'][:, 0].mean(), time.time() - t0), flush=True)
t = np.concatenate([s[0] for s in spikes]); i = np.concatenate([s[1] for s in spikes]); ev = np.concatenate(evs)
on, off = A.intervals(ev, t[0], S['step']); c = np.zeros((4, Z.NE)); tt = np.zeros(4)
for a, b, p in on: c[p] += A.counts(t, i, a, b); tt[p] += (b - a) * Z.DT
table(c / np.maximum(tt, 1e-9)[:, None], 'last %d s of %d s of training - rate (Hz) while each input is shown:' % (rec_last, T))
lab = np.full(Z.NE, -1)
for k in range(9): lab[cells[k]] = k
ep, eq, w = S['ee'][5], S['ee'][1], S['ee_ws'][:, 0]
print('mean E-E weight from cells of (row) to cells of (column); start 0.100:'); print('  %-8s' % '' + ' '.join('%6s' % PN[k] for k in range(6)))
for a in range(6): print('  %-8s' % PN[a] + ' '.join('%6.3f' % w[(lab[ep] == a) & (lab[eq] == b)].mean() if ((lab[ep] == a) & (lab[eq] == b)).any() else '   nan' for b in range(6)))
pickle.dump(S, open(os.path.join(A.OUT, tag + '_state.pkl'), 'wb'), protocol=4)
print('learning off. One patch alone is shown for 0.5 s (8 times each): rate (Hz) of the cells of every patch during it'); print('  %-12s' % 'shown alone' + ' '.join('%6s' % PN[k] for k in range(6)))
P2 = Z.params(on=0.5, off=1.5, scale=25.0, plastic=0.0, ee_on=LOOP, seq=1.0)
for k in (0, 1, 2, 3, 4):
    S['sched'][:] = (0.0, -1.0, S['step'] + 5000, 0.0); re, ri, ev, sp = sim.run(16.5, P2, [pat[k]] , record=True); t, i = sp; on, off = A.intervals(ev, 0, S['step']); c = np.zeros(Z.NE); tt = 0.0
    for a, b, p in on: c += A.counts(t, i, a, b); tt += (b - a) * Z.DT
    print('  %-12s' % PN[k] + ' '.join('%6.1f' % (c[cells[m]].mean() / max(tt, 1e-9)) for m in range(6)), flush=True)
