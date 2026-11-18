"""no training (learning off): measurements on a saved state of check_voice.py.
  python3 check_voice_look.py look STATE.pkl   the tables re-computed from the weights; what the loop level does during 'F then A',
                                               'A then F', 'F and A together', 'G then B' twice (rates per 0.1 s)
  python3 check_voice_look.py link STATE.pkl   the loop link 'cells lit by F -> cells lit by A' is set by hand to a given summed
                                               weight per A cell and F is played: does A's group take over, does the mouth say A?
                                               Then the same with fatigue also in the loop level (a held group lets go by itself).
Mine (not the designer's, not from a paper): the hand-set link values and the loop-level fatigue jumps 0.002 (the simulator
default of the reproduced network's slow adaptation) and 0.005."""
import sys, os, pickle, numpy as np
import levels as L, zenke as Z
mode, path = sys.argv[1], sys.argv[2]
d = pickle.load(open(path if os.path.exists(path) else os.path.join(Z.HERE, 'data/zenke_run', path), 'rb')); C, code, ins, cells, M, lit = d['C'], d['code'], d['ins'], d['cells'], d['M'], d['lit']
n0, n1, n2 = C['n0'], C['n1'], C['n2']; ear = np.arange(C['near'])
ee = {(0, 1): [(C['k01'] / n0, C['w01']), (C['k01weak'] / n0, C['w01weak'])], (1, 1): (C['p11'], C['w11']), (1, 0): (C['k10weak'] / n1, C['w10weak']), (1, 2): (C['k12weak'] / n1, C['w12weak'])}
net = L.Net(C['seed'], [n0, n1, n2], [n0 // 4, n1 // 4, n2 // 4], {0: [(C['K'], C['w'], ear), (C['kweak'], C['wweak'], ear)]}, ee, alpha=C['alpha'],
            s_fixed=[(ins[x], 2, cells[x], C['kins'], C['wins']) for x in 'AB'])
net.S = d['S']; S = net.S; l0, l1, l2 = net.level(0), net.level(1), net.level(2)
Poff = net.par(plastic=0.0, bg=C['bg'], alpha=C['alpha'], beta=C['beta'], wmax2=C['wret'], tau2=C['tau2'])
SND = 'ABFG'; mouth = [False, False]


if mode == 'look':
    # ---- independent re-computation of the tables from the weights
    pre, post, lev, w = S['ee'][5], S['ee'][1], S['lev'], S['ee_ws'][:, 0]
    print('fatigue jump: level 0 / 1 / 2 mean %.4f %.4f %.4f' % tuple(S['ad2'][s].mean() for s in (l0, l1, l2)))
    print('lit sizes level 1:', {x: len(lit[1][x]) for x in SND}, ' level 0:', {x: len(lit[0][x]) for x in SND})
    m12 = (lev[pre] == 1) & (lev[post] == 2)
    for g in 'AB':
        row = []
        for x in SND:
            k = m12 & np.isin(pre, lit[1][x]); row.append(np.bincount(post[k], weights=w[k], minlength=len(lev))[M[g]].mean())
        print('summed 1->2 weight per M_%s cell from level-1 cells lit by A B F G: ' % g + ' '.join('%7.2f' % v for v in row))
    m11 = (lev[pre] == 1) & (lev[post] == 1)
    for a in SND:
        k = m11 & np.isin(pre, lit[1][a]); tot = np.bincount(post[k], weights=w[k], minlength=len(lev))
        print('loop: summed weight from cells lit by %s onto a cell lit by A B F G: ' % a + ' '.join('%6.2f' % tot[lit[1][b]].mean() for b in SND))
    m01 = (lev[pre] == 0) & (lev[post] == 1)
    for a in SND:
        k = m01 & np.isin(pre, lit[0][a]); tot = np.bincount(post[k], weights=w[k], minlength=len(lev))
        print('0->1: summed weight from level-0 cells lit by %s onto a level-1 cell lit by A B F G: ' % a + ' '.join('%6.2f' % tot[lit[1][b]].mean() for b in SND))
    ie = np.bincount(S['ie'][1], weights=S['ie_w'], minlength=len(lev))
    print('I->E summed per cell: level 0 %.1f, level 1 %.1f, level 2 %.1f | level-1 cells lit by A %.1f B %.1f F %.1f G %.1f, lit by none %.1f' % (
        ie[l0].mean(), ie[l1].mean(), ie[l2].mean(), *[ie[lit[1][x]].mean() for x in SND], ie[np.setdiff1d(np.arange(l1.start, l1.stop), np.concatenate([lit[1][x] for x in SND]))].mean()))


    def play(ext, seconds, rec, speak=True):
        for _ in range(int(round(seconds / C['chunk']))):
            said = ''.join(x for x, on in zip('AB', mouth) if on and speak)
            parts = ([ext] if ext is not None else []) + [code[x] for x in said]
            r = net.show(np.unique(np.concatenate(parts)) if parts else None, C['chunk'], Poff)
            ra, rb = r[M['A']].mean(), r[M['B']].mean(); mouth[0], mouth[1] = bool(ra >= C['emit']), bool(rb >= C['emit'])
            rec.append([ra, rb] + [r[lit[1][x]].mean() for x in SND] + [r[lit[0][x]].mean() for x in SND] + [said or '.'])


    def trial(label, seq, speak=True):
        net.reset(); mouth[:] = [False, False]; rec = []
        play(None, 0.5, rec, speak)
        for snd, sec in seq:
            ext = None if snd is None else np.unique(np.concatenate([code[x] for x in snd]))
            play(ext, sec, rec, speak)
        a = np.array([r[:10] for r in rec]); print('%s   (0.5 s rest, then %s; mouth %s)' % (label, ', '.join('%s %.0f s' % (s or 'nothing', t) for s, t in seq), 'on' if speak else 'off'))
        for j, name in enumerate(('M_A', 'M_B', 'L1:A', 'L1:B', 'L1:F', 'L1:G', 'L0:A', 'L0:B', 'L0:F', 'L0:G')):
            print('    %-5s %s' % (name, ' '.join('%2.0f' % v for v in a[:, j])))
        print('    said  %s' % ' '.join('%2s' % r[-1] for r in rec), flush=True)
        return a


    a = trial('F then A', [('F', 1.0), ('A', 1.0), (None, 4.0)])
    both = ((a[:, 4] > 5) & (a[:, 2] > 5)).sum() * C['chunk']; print('    time with level-1 groups of F and of A both above 5 Hz: %.1f s' % both)
    a = trial('F then A, mouth off', [('F', 1.0), ('A', 1.0), (None, 2.0)], speak=False)
    both = ((a[:, 4] > 5) & (a[:, 2] > 5)).sum() * C['chunk']; print('    time with level-1 groups of F and of A both above 5 Hz: %.1f s' % both)
    trial('A then F', [('A', 1.0), (None, 5.0), ('F', 1.0), (None, 2.0)])
    trial('F and A together, mouth off', [('FA', 1.0), (None, 2.0)], speak=False)
    trial('G then B, twice', [('G', 1.0), ('B', 1.0), (None, 4.0), ('G', 1.0), ('B', 1.0), (None, 2.0)])

else:
    pre, post, lev = S['ee'][5], S['ee'][1], S['lev']
    onlyF = np.setdiff1d(lit[1]['F'], lit[1]['A']); onlyA = np.setdiff1d(lit[1]['A'], lit[1]['F'])
    kFA = np.nonzero((lev[pre] == 1) & (lev[post] == 1) & np.isin(pre, onlyF) & np.isin(post, onlyA))[0]
    w0 = S['ee_ws'][kFA, 0].copy(); cnt = np.bincount(post[kFA], minlength=len(lev))[onlyA]
    print('F -> A loop synapses: %d, per A cell %.1f (min %d); summed weight now %.2f' % (len(kFA), cnt.mean(), cnt.min(), np.bincount(post[kFA], weights=w0, minlength=len(lev))[onlyA].mean()), flush=True)
    ad0 = S['ad2'].copy()


    def play(ext, seconds, rec):
        for _ in range(int(round(seconds / C['chunk']))):
            said = ''.join(x for x, on in zip('AB', mouth) if on)
            parts = ([ext] if ext is not None else []) + [code[x] for x in said]
            r = net.show(np.unique(np.concatenate(parts)) if parts else None, C['chunk'], Poff)
            ra, rb = r[M['A']].mean(), r[M['B']].mean(); mouth[0], mouth[1] = bool(ra >= C['emit']), bool(rb >= C['emit'])
            rec.append([ra, rb, r[onlyA].mean(), r[onlyF].mean(), said or '.'])


    def trial(label, seconds=8.0):
        net.reset(); S['E'][6][:] = 0.0; mouth[:] = [False, False]; rec = []
        play(None, 0.5, rec); play(code['F'], 1.0, rec); play(None, seconds, rec)
        a = np.array([r[:4] for r in rec]); said = [r[-1] for r in rec]; on = [i for i, s in enumerate(said) if 'A' in s]
        print('%s: mouth said A %s' % (label, 'from %.1f to %.1f s after F began (%.1f s in all)' % (on[0] * 0.1 - 0.5, on[-1] * 0.1 - 0.4, len(on) * 0.1) if on else 'never'))
        for j, name in enumerate(('M_A', 'L1:A', 'L1:F')): print('    %-5s %s' % (name, ' '.join('%2.0f' % v for v in a[:, (0, 2, 3)[j]])))
        print('    said  %s' % ' '.join('%2s' % s for s in said), flush=True)


    for target in (8.0, 16.0, 24.0, 32.0, 48.0):
        S['ee_ws'][kFA, 0] = target / cnt.mean(); trial('link F -> A set to %.0f per A cell' % target)
    print('--- with fatigue in the loop level as well (jump per spike, decay %.0f s)' % C['tau2'], flush=True)
    for jump, target in ((0.002, 1.44), (0.002, 16.0), (0.002, 32.0), (0.005, 16.0), (0.005, 32.0)):
        S['ad2'][:] = ad0; S['ad2'][l1] = jump
        if target == 1.44: S['ee_ws'][kFA, 0] = w0
        else: S['ee_ws'][kFA, 0] = target / cnt.mean()
        trial('loop-level fatigue %.3f, link F -> A %.1f' % (jump, target), seconds=14.0)
