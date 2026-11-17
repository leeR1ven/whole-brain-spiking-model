"""Train the order-learning network of maes.py the way the authors' run script does (runsim.jl):
  5 s to settle; then N x 2 min with the 30 groups stimulated one after the other (9 ms each, every 15 ms);
  then N x 2 min left alone (plasticity on).  After every 2 min: 2 s without stimulation and without learning, in which
  it is measured whether the groups fire one after the other in the trained order.
usage: python3 run_maes.py [N = 30] [TAG = m1]"""
import sys, os, time, pickle, numpy as np
import maes as Mx
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30; tag = sys.argv[2] if len(sys.argv) > 2 else 'm1'
OUT = os.path.join(Mx.HERE, 'data/maes_run'); os.makedirs(OUT, exist_ok=True)


def order(sp, t0, ms, bin_ms=5.0, thr=30.0):
    """from the E spikes: which group is the most active one in every 5 ms (if above thr Hz); the list of groups in the
    order they came; share of the changes that go to the NEXT group; mean time a group stays on (ms)"""
    t, i = sp; nb = int(ms / bin_ms); b = ((t - t0) * 0.1 / bin_ms).astype(np.int64); ok = (b >= 0) & (b < nb)
    r = np.zeros((nb, Mx.NPOP)); np.add.at(r, (b[ok], i[ok] // (Mx.NE // Mx.NPOP)), 1.0); r /= (Mx.NE // Mx.NPOP) * bin_ms * 1e-3
    top = np.where(r.max(1) > thr, r.argmax(1), -1); runs = []
    for g in top:
        if g < 0: continue
        if runs and runs[-1][0] == g: runs[-1][1] += 1
        else: runs.append([g, 1])
    d = np.array([(b_[0] - a_[0]) % Mx.NPOP for a_, b_ in zip(runs[:-1], runs[1:])])
    return (len(d), float((d == 1).mean()) if len(d) else 0.0, float(np.mean([n for g, n in runs]) * bin_ms) if runs else 0.0, float((top >= 0).mean()), [g for g, n in runs][:40])


sim = Mx.Sim(1); t00 = time.time(); sim.run(5000.0, False, 1000.0)
for phase, stim, delay in (('train', True, 0.0), ('alone', False, 1000.0)):
    for it in range(1, N + 1):
        t0 = time.time(); re, ri, _ = sim.run(120000.0, stim, delay); s0 = sim.S['step']; r2e, r2i, sp = sim.run(2000.0, False, -1.0, record=True)
        nch, fnext, stay, frac, seq = order(sp, s0, 2000.0); st = sim.structure()
        print('%s %2d (%4.0f min) | E %.2f I %.2f Hz | E-E w: inside group %.2f, to next %.2f, to previous %.2f, others %.2f; I-E %.1f | 2 s alone: E %.2f Hz, a group is on %.0f %% of the time, stays %.0f ms, %d changes, %.0f %% of them to the NEXT group | %.0f s' % (
            phase, it, sim.S['step'] * 1e-4 / 60, re, ri, *st, sim.S['ie_w'].mean(), r2e, 100 * frac, stay, nch, 100 * fnext, time.time() - t0), flush=True)
        print('      order of groups: ' + ' '.join(str(g) for g in seq), flush=True)
        if it % 5 == 0 or it == N:
            pickle.dump(sim.S, open(os.path.join(OUT, tag + '_state.pkl'), 'wb'), protocol=4)
print('total %.0f s' % (time.time() - t00))
