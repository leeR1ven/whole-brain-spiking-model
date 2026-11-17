"""Runs the learning network of zenke.py through the authors' three stages (their run scripts 1run_init / 2run_learn / 3run_cued):
  1 init   30 min, shapes shown ~1 s every ~2 s          -> do cells become selective, do assemblies form?
  2 learn  30 min, shapes shown 0.2 s every ~5 s         -> does the shown assembly stay active in the pause (delay activity)?
  3 cue    30 min, distorted / partial shapes, 0.2 s      -> does a partial cue call up the right assembly?
usage: python3 run_zenke.py TAG [hom=0|1] [first_stage=1] [seconds_per_stage=1800] [network changes]
writes data/zenke_run/TAG_*.npz (weights, spikes of the last minutes, stimulus log) and prints a line per minute."""
import sys, os, time, pickle, numpy as np
import zenke as Z
import zenke_ana as A

tag = sys.argv[1]; hom = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
first = int(sys.argv[3]) if len(sys.argv) > 3 else 1; T = float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0
BUILD = eval(sys.argv[5]) if len(sys.argv) > 5 else {}                    # changes to the network, e.g. "dict(sparse_ee=0.1, wee=0.05)"
OUT = os.path.join(Z.HERE, 'data/zenke_run'); os.makedirs(OUT, exist_ok=True)
shapes = Z.load_patterns(Z.CODE + '/shapes.pat'); cues = Z.load_patterns(Z.CODE + '/shapes_cues.pat')
STAGES = {1: ('init', shapes, dict(on=1.0, off=2.0, scale=25.0), 300.0),
          2: ('learn', shapes, dict(on=0.2, off=5.0, scale=25.0), 600.0),
          3: ('cue', cues, dict(on=0.2, off=5.0, scale=50.0), 900.0)}
state = os.path.join(OUT, tag + '_state.pkl')
if first > 1:
    sim = Z.Sim(1); sim.S = pickle.load(open(state, 'rb')); print('loaded state at t = %.0f s' % (sim.S['step'] * Z.DT), flush=True)
else:
    sim = Z.Sim(1, **BUILD)
for st in range(first, 4):
    name, pats, kw, rec_last = STAGES[st]; P = Z.params(hom=hom, **kw); t_done = 0.0; spikes = []; evs = []
    rec_last = min(rec_last, T)
    print('--- stage %d (%s): %.0f s, %s, sliding threshold %s' % (st, name, T, kw, 'on' if hom else 'off'), flush=True)
    while t_done < T - 1e-9:
        chunk = min(60.0, T - t_done); rec = t_done >= T - rec_last - 1e-9; t0 = time.time()
        re, ri, ev, sp = sim.run(chunk, P, pats, record=rec, max_spikes=10_000_000); t_done += chunk; S = sim.S
        evs.append(ev)
        if rec: spikes.append(sp)
        w = S['ee_ws'][:, 0]
        print('t=%5.0f s  E %5.2f Hz  I %5.2f Hz | E-E w %.3f (ref %.3f, >0.25: %4.1f %%)  in-E w %.3f  I-E w %.3f | %.0f s' % (
            S['step'] * Z.DT, re, ri, w.mean(), S['ee_ws'][:, 1].mean(), 100 * (w > 0.25).mean(), S['se_ws'][:, 0].mean(), S['ie_w'].mean(), time.time() - t0), flush=True)
    sim.save(os.path.join(OUT, '%s_%s_weights.npz' % (tag, name)))
    np.savez_compressed(os.path.join(OUT, '%s_%s_spikes.npz' % (tag, name)), t=np.concatenate([s[0] for s in spikes]), i=np.concatenate([s[1] for s in spikes]),
                        ev=np.concatenate(evs), t_end=sim.S['step'])
    pickle.dump(sim.S, open(state, 'wb'), protocol=4)
    A.report(tag, name)
