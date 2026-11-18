"""A closed loop: ear -> level 0 (hearing) -> level 1 (loop level) -> level 2 (motor cells) -> mouth -> ear. Cells, rules and
kernel of the reproduced network (zenke.py), wiring in levels.py.
The designer's (protocol of 2026-11-18): three levels and their sizes and wiring (level 0: 60 strong 1.5 + 350 weak 0.05
inputs from the ear units, no loop; level 1: 150 strong 1.0 + 350 weak 0.05 from level 0, loop p 0.05 w 0.1, returns 350
weak 0.05 per level-0 cell with ceiling 0.25; level 2: 1024 + 256 cells, 350 weak 0.05 from level 1, nothing else); ear
units 0..3071, instinct units above (64 per motor group, 30 inputs per motor cell); two motor groups M_A, M_B (cells 0-199,
200-399 of level 2); four sounds A, B, F, G (disjoint, 256 ear units each); the mouth (0.1 s chunks: a group at >= 10 Hz in a
chunk makes the mouth say its sound in the next chunk, added to whatever is played); motor fatigue (only the level-2 cells
get the long-lasting adaptation); the stages H / 1 / 3, the tests 2 / 4, stay = 0.8; the expectations; the checks.
Taken from the earlier two-level run (check_concept3.py, c3_cap): bg 1 Hz, alpha 0.5 Hz, beta 0.01, sliding threshold
(hom), return ceiling 0.25 (ee_pl = 2), staging with ee_pl = 0, the test (lit = > 10 Hz, 2 presentations, 0.3 s rest, held =
the second after the sound).
Mine: instinct weight 0.75 and fatigue jump 0.015 with the kernel's decay of 20 s (calibrated in short tests: a drive that
makes the group fire 20 Hz falls below 5 Hz after 1.6-1.9 s, the instinct keeps it >= 10 Hz for ~0.9 s; on a repeated instinct
trial 6 s later the group fires much less - a decay of 5 s, tried first, avoided that but let a 2-minute closed-loop check run
away: motor cells 100 Hz, summed 1->2 weights > 100); each motor cell's 30 instinct inputs are 30
different units of its group's 64; the instinct synapses are input synapses and therefore plastic like all of them (the
kernel has no fixed flag for input synapses); the checkpoint test plays the sound without the mouth (one call, as in
check_concept3); every test runs with learning off on the current state and the state is restored afterwards (the random
numbers are not); an extra repeat test after stage H as a baseline; the start-up checks.
usage: python3 check_voice.py ["dict(...)" settings]"""
import sys, time, pickle, os, numpy as np
import levels as L, zenke as Z

C = dict(n0=4096, n1=4096, n2=1024, near=3072, m=256, K=60, w=1.5, kweak=350, wweak=0.05, k01=150, w01=1.0, k01weak=350, w01weak=0.05, p11=0.05, w11=0.1,
         k10weak=350, w10weak=0.05, k12weak=350, w12weak=0.05, nm=200, nins=64, kins=30, wins=0.75, wsteady=0.70, ad2=0.015, tau2=20.0,
         bg=1.0, alpha=0.5, beta=0.01, hom=1.0, wret=0.25, stay=0.8, chunk=0.1, emit=10.0, TH=900.0, T1=1200.0, T3=900.0, every=300.0, reps=3, seed=1, tag='voice', out='')
C.update(eval(sys.argv[1]) if len(sys.argv) > 1 else {})
OUT = C['out'] or os.path.join(Z.HERE, 'data/zenke_run'); T0 = time.time()
rng = np.random.default_rng(7 + 10 * (C['seed'] - 1)); perm = rng.permutation(C['near']); m = C['m']; SND = 'ABFG'
code = {x: np.sort(perm[i * m:(i + 1) * m]) for i, x in enumerate(SND)}                     # four disjoint sounds over the ear units
ear = np.arange(C['near']); ins = {x: C['near'] + i * C['nins'] + np.arange(C['nins']) for i, x in enumerate('AB')}
cells = {x: i * C['nm'] + np.arange(C['nm']) for i, x in enumerate('AB')}                  # motor groups, inside level 2
n0, n1, n2 = C['n0'], C['n1'], C['n2']
ee = {(0, 1): [(C['k01'] / n0, C['w01']), (C['k01weak'] / n0, C['w01weak'])], (1, 1): (C['p11'], C['w11']), (1, 0): (C['k10weak'] / n1, C['w10weak']), (1, 2): (C['k12weak'] / n1, C['w12weak'])}
net = L.Net(C['seed'], [n0, n1, n2], [n0 // 4, n1 // 4, n2 // 4], {0: [(C['K'], C['w'], ear), (C['kweak'], C['wweak'], ear)]}, ee, alpha=C['alpha'],
            s_fixed=[(ins[x], 2, cells[x], C['kins'], C['wins']) for x in 'AB'])
S = net.S; l0, l1, l2 = net.level(0), net.level(1), net.level(2); M = {x: l2.start + cells[x] for x in 'AB'}
S['ad2'][l2] = C['ad2']                                                                    # fatigue: only the motor level
kw = dict(bg=C['bg'], alpha=C['alpha'], beta=C['beta'], wmax2=C['wret'], tau2=C['tau2']); P = net.par(hom=C['hom'], **kw); Poff = net.par(plastic=0.0, **kw)
(k01, _, _), (k11, _, _), (k10, _, _), (k12, pre12, post12) = (net.block(a, b) for a, b in ((0, 1), (1, 1), (1, 0), (1, 2)))
se_pre, se_post = S['se'][5], S['se'][1]; kins = {x: np.nonzero(np.isin(se_pre, ins[x]))[0] for x in 'AB'}; kear = np.nonzero(se_pre < C['near'])[0]
ie_lev = S['lev'][S['ie'][1]]
print('settings', C, flush=True)
mouth = [False, False]                                                                     # does the mouth say A / B in the next chunk
LIT = {}; REC = {}


def snap():
    return dict({k: [x.copy() for x in S[k]] for k in ('E', 'I', 'tr', 'stp', 'dl')}, **{k: S[k].copy() for k in ('sched', 'pf', 'ee_ws', 'se_ws', 'ie_w', 'ee_pl', 'ad2')}, step=S['step'], mouth=list(mouth))


def restore(s):
    for k in ('E', 'I', 'tr', 'stp', 'dl'):
        for x, y in zip(S[k], s[k]): x[...] = y
    for k in ('sched', 'pf', 'ee_ws', 'se_ws', 'ie_w', 'ee_pl', 'ad2'): S[k][...] = s[k]
    S['step'] = s['step']; mouth[:] = s['mouth']


def play(ext, seconds, PP, drive=None, rec=None, speak=True, track=None):
    """`seconds` in chunks of 0.1 s. ext: S units played from outside (None: nothing); drive: 'A' / 'B' = that group's instinct
    units on. The mouth adds the sound of every group that fired >= 10 Hz in the previous chunk. Returns what the mouth said
    in each chunk ('.', 'A', 'B', 'AB'). rec: per chunk [M_A, M_B, level-1 cells lit by A, B, F, G (Hz), said];
    track: per chunk, which S units sent a spike in it."""
    said_all = []
    for _ in range(int(round(seconds / C['chunk']))):
        said = ''.join(x for x, on in zip('AB', mouth) if on and speak)
        parts = ([ext] if ext is not None else []) + ([ins[drive]] if drive else []) + [code[x] for x in said]
        st0 = S['step']; r = net.show(np.unique(np.concatenate(parts)) if parts else None, C['chunk'], PP)
        ra, rb = r[M['A']].mean(), r[M['B']].mean(); mouth[0], mouth[1] = bool(ra >= C['emit']), bool(rb >= C['emit'])
        said_all.append(said or '.')
        if rec is not None: rec.append([ra, rb] + [r[LIT[1][x]].mean() if len(LIT.get(1, {}).get(x, [])) else 0.0 for x in SND] + [said or '.'])
        if track is not None: track.append(S['stp'][7] >= st0)
    return said_all


def runs(em, x, t0=0.0):
    """time intervals (s) in which the mouth said x"""
    on = np.array([x in e for e in em] + [False]); out = []; i = 0
    while i < len(em):
        if on[i]:
            j = i
            while on[j]: j += 1
            out.append((t0 + i * C['chunk'], t0 + j * C['chunk'])); i = j
        else: i += 1
    return out


fmtr = lambda iv: ','.join('%.1f-%.1f' % ab for ab in iv) or '-'
ov = lambda a, b: len(np.intersect1d(a, b)) / max(1, min(len(a), len(b)))


def weights():
    ws = S['ee_ws'][:, 0]
    return 'weights: ear->0 %.3f, instinct->M_A %.3f, ->M_B %.3f, 0->1 %.3f, 1->1 %.3f, 1->0 %.3f (max %.3f), 1->2 %.4f, I->E level 0 %.3f, 1 %.3f, 2 %.3f' % (
        S['se_ws'][kear, 0].mean(), S['se_ws'][kins['A'], 0].mean(), S['se_ws'][kins['B'], 0].mean(), ws[k01].mean(), ws[k11].mean(), ws[k10].mean(), ws[k10].max(), ws[k12].mean(),
        *[S['ie_w'][ie_lev == lv].mean() for lv in range(3)])


def motor_table(lit1):
    """summed 1->2 weight per M_A / M_B cell from the level-1 cells lit by each sound"""
    ws = S['ee_ws'][k12, 0]; tab = {}
    for x in SND:
        msk = np.isin(pre12, lit1[x]); tot = np.bincount(post12[msk], weights=ws[msk], minlength=net.ne); n = np.bincount(post12[msk], minlength=net.ne)
        for g in 'AB': tab[(g, x)] = (tot[M[g]].mean(), n[M[g]].mean())
    return tab


def test(label):
    """learning off, every sound alone after a reset (no mouth): lit cells, overlaps, held rate, weights, motor table"""
    s0 = snap(); R = {}; H = {}
    for x in SND:
        r = np.zeros(net.ne); h = np.zeros(net.ne)
        for _ in range(2):
            net.reset(); net.show(None, 0.3, Poff); r += net.show(code[x], 1.0, Poff) / 2; h += net.show(None, 1.0, Poff) / 2
        R[x] = r; H[x] = h
    restore(s0)
    lit = {lv: {x: np.nonzero(R[x][sl] > 10)[0] + sl.start for x in SND} for lv, sl in ((0, l0), (1, l1), (2, l2))}
    print('%s (t_total %.0f s, wall %.0f s)' % (label, S['step'] * Z.DT, time.time() - T0), flush=True)
    for lv in (0, 1):
        lt = lit[lv]
        print('  level %d: lit A %d, B %d, F %d, G %d | overlap AB %.2f AF %.2f AG %.2f BF %.2f BG %.2f FG %.2f | held (lit cells, 1 s after) A %.1f B %.1f F %.1f G %.1f Hz | mean rate during sound %.2f Hz' % (
            lv, *[len(lt[x]) for x in SND], *[ov(lt[a], lt[b]) for a, b in ('AB', 'AF', 'AG', 'BF', 'BG', 'FG')], *[H[x][lt[x]].mean() if len(lt[x]) else 0.0 for x in SND],
            np.mean([R[x][(l0, l1)[lv]].mean() for x in SND])), flush=True)
    print('  motor during the sound (M_A / M_B Hz): ' + ', '.join('%s %.1f/%.1f' % (x, R[x][M['A']].mean(), R[x][M['B']].mean()) for x in SND) +
          ' | 1 s after: ' + ', '.join('%s %.1f/%.1f' % (x, H[x][M['A']].mean(), H[x][M['B']].mean()) for x in SND), flush=True)
    print('  ' + weights(), flush=True)
    tab = motor_table(lit[1])
    for g in 'AB':
        print('  summed 1->2 weight per M_%s cell from level-1 cells lit by  A %.2f  B %.2f  F %.2f  G %.2f   (synapses per cell %.1f %.1f %.1f %.1f)' % (
            g, *[tab[(g, x)][0] for x in SND], *[tab[(g, x)][1] for x in SND]), flush=True)
    LIT.clear(); LIT.update(lit); REC.setdefault('tests', []).append(dict(label=label, lit=lit, tab=tab, R={x: R[x].astype(np.float32) for x in SND}, H={x: H[x].astype(np.float32) for x in SND}))
    return lit


def motor_test(label, sounds):
    """learning off; per sound and repetition: reset, 0.5 s rest, the sound 1 s from outside, 8 s nothing; mouth on all the time"""
    s0 = snap(); print('%s: learning off, mouth on; times from the onset of the sound; rates per 0.1 s (lit groups from the last checkpoint)' % label, flush=True); out = {}
    for x in sounds:
        for rep in range(C['reps']):
            net.reset(); mouth[:] = [False, False]; rec = []
            play(None, 0.5, Poff, rec=rec); play(code[x], 1.0, Poff, rec=rec); play(None, 8.0, Poff, rec=rec)
            em = [r[-1] for r in rec]; t0 = -0.5; a = np.array([r[:6] for r in rec]); iv = {g: runs(em, g, t0) for g in 'AB'}
            allv = sorted(iv['A'] + iv['B']); first = '-' if not allv else ''.join(g for g in 'AB' if iv[g] and abs(iv[g][0][0] - allv[0][0]) < 1e-9)
            print('  heard %s rep %d: first said %s, start %s, stop %s | said A %s (%.1f s), B %s (%.1f s) | peak M_A %.1f, M_B %.1f Hz (after the sound: %.1f, %.1f)' % (
                x, rep, first, '%.1f' % allv[0][0] if allv else '-', '%.1f' % max(b for _, b in allv) if allv else '-', fmtr(iv['A']), sum(b - a_ for a_, b in iv['A']),
                fmtr(iv['B']), sum(b - a_ for a_, b in iv['B']), a[5:, 0].max(), a[5:, 1].max(), a[15:, 0].max(), a[15:, 1].max()), flush=True)
            for j, name in enumerate(('M_A', 'M_B', 'L1:A', 'L1:B', 'L1:F', 'L1:G')): print('    %-5s %s' % (name, ' '.join('%.0f' % v for v in a[:, j])))
            print('    said  %s' % ' '.join(e for e in em), flush=True)
            out[(x, rep)] = dict(rec=rec, iv=iv, first=first)
    restore(s0); REC.setdefault('motor', []).append(dict(label=label, out=out))
    return out


def save(name):
    pickle.dump(dict(S=S, C=C, code=code, ins=ins, cells=cells, M=M, lit=LIT, rec=REC), open(os.path.join(OUT, '%s_state_%s.pkl' % (C['tag'], name)), 'wb'), protocol=4)
    print('saved %s_state_%s.pkl' % (C['tag'], name), flush=True)


# ---------------- start-up checks (learning off, state restored)
nE = np.bincount(se_post[kear], minlength=net.ne); nI_ = {x: np.bincount(se_post[kins[x]], minlength=net.ne) for x in 'AB'}; n12 = np.bincount(post12, minlength=net.ne)
print('wiring: ear -> level-0 cell %.1f per cell (min %d, max %d); ear -> level 1 / 2: %d / %d synapses; instinct -> level 0 / 1: %d / %d; instinct -> level 2 outside M_A, M_B: %d' % (
    nE[l0].mean(), nE[l0].min(), nE[l0].max(), nE[l1].sum(), nE[l2].sum(), (nI_['A'] + nI_['B'])[l0].sum(), (nI_['A'] + nI_['B'])[l1].sum(), (nI_['A'] + nI_['B'])[l2.start + 2 * C['nm']:l2.stop].sum()), flush=True)
print('        instinct A -> each M_A cell %s, -> M_B cells %d in all; instinct B -> each M_B cell %s, -> M_A cells %d in all; start weight %.2f' % (
    set(nI_['A'][M['A']]), nI_['A'][M['B']].sum(), set(nI_['B'][M['B']]), nI_['B'][M['A']].sum(), S['se_ws'][kins['A'], 0].mean()), flush=True)
r500 = l1.start + rng.choice(n1, 500, replace=False); msk = np.isin(pre12, r500); tot = np.bincount(post12[msk], weights=S['ee_ws'][k12[msk], 0], minlength=net.ne)[l2]
print('        level-1 inputs per motor cell %.1f (min %d, max %d); 1->2 weights from levels other than 1: %d; summed start weight from a random 500-cell level-1 set onto a motor cell %.2f (sd %.2f, min %.2f, max %.2f)' % (
    n12[l2].mean(), n12[l2].min(), n12[l2].max(), int(np.isin(S['lev'][S['ee'][1]], [2]).sum() - len(k12)), tot.mean(), tot.std(), tot.min(), tot.max()), flush=True)
s0 = snap()
# fatigue: the group's own response to its instinct units (no mouth), and to a steady drive that makes it fire about 20 Hz
for lab, w_, ad, seq in (('instinct (w %.2f), no fatigue' % C['wins'], C['wins'], 0.0, [(1, 2.0)]), ('instinct (w %.2f), with fatigue' % C['wins'], C['wins'], C['ad2'], [(1, 1.5), (0, 4.5)] * 3),
                         ('steady drive (w %.2f), no fatigue' % C['wsteady'], C['wsteady'], 0.0, [(1, 4.0)]), ('steady drive (w %.2f), with fatigue' % C['wsteady'], C['wsteady'], C['ad2'], [(1, 4.0)])):
    S['se_ws'][kins['A'], 0] = w_; S['ad2'][l2] = ad; net.reset(); rec = []; play(None, 0.3, Poff, speak=False)
    for on, sec in seq: play(None, sec, Poff, drive='A' if on else None, rec=rec, speak=False)
    ra = np.array([r[0] for r in rec]); out = []; i = 0
    for on, sec in seq:
        n_ = int(round(sec / C['chunk'])); x = ra[i:i + n_]; i += n_
        if on:
            sm = np.convolve(x, np.ones(3) / 3, 'valid'); b = np.nonzero(sm < 5)[0]
            out.append('%.1f s on: mean %.1f Hz, first 1 s %.1f Hz, 0.1-s bins >= 10 Hz %d, 0.3-s mean first < 5 Hz at %s, last 2 s %.1f Hz' % (
                sec, x.mean(), x[:10].mean(), (x >= 10).sum(), '%.1f s' % ((b[0] + 3) * C['chunk']) if len(b) else 'never', x[-20:].mean()))
    print('calibration, M_A, %s (M_B mean %.1f Hz): %s' % (lab, np.mean([r[1] for r in rec]), ' || '.join(out)), flush=True)
    print('    M_A %s' % ' '.join('%.0f' % v for v in ra), flush=True)
restore(s0)
# mouth loop: instinct A on -> which ear units fire in the chunks in which the mouth says A; which level-0 cells they light
net.reset(); mouth[:] = [False, False]; trk = []; rr = []; play(None, 0.3, Poff)
for drive, sec in (('A', 1.5), (None, 1.0)):
    for _ in range(int(round(sec / C['chunk']))):
        said = play(None, C['chunk'], Poff, drive=drive, track=trk)[0]; rr.append((said, net.cnt / C['chunk']))
ch = [i for i, (s_, _) in enumerate(rr) if s_ == 'A']; trk = trk[-len(rr):]
if ch:
    frac = np.mean([trk[i] for i in ch], axis=0); on_ = np.nonzero(frac >= 0.5)[0]; on_ear = on_[on_ < C['near']]
    self_r = np.mean([rr[i][1] for i in ch], axis=0); lit_self = np.nonzero(self_r[l0] > 10)[0]
    lit_ext = []
    for _ in range(2):
        net.reset(); net.show(None, 0.3, Poff); lit_ext.append(np.nonzero(net.show(code['A'], 1.0, Poff)[l0] > 10)[0])
    print('mouth check: said %s | chunks with A: %d; ear units firing in >= half of them: %d, of which in code A %d (code A has %d; code B / F / G units among them: %d / %d / %d); other S units: %s' % (
        ' '.join(s_ for s_, _ in rr), len(ch), len(on_ear), np.isin(on_ear, code['A']).sum(), len(code['A']), np.isin(on_ear, code['B']).sum(), np.isin(on_ear, code['F']).sum(),
        np.isin(on_ear, code['G']).sum(), 'instinct A %d of %d, anything else %d' % (np.isin(on_, ins['A']).sum(), C['nins'], (~np.isin(on_, np.concatenate([code['A'], ins['A']]))).sum())), flush=True)
    print('             level-0 cells lit by the self-produced A %d, by A played from outside %d / %d; overlap self vs outside %.2f / %.2f, outside vs outside %.2f' % (
        len(lit_self), len(lit_ext[0]), len(lit_ext[1]), ov(lit_self, lit_ext[0]), ov(lit_self, lit_ext[1]), ov(lit_ext[0], lit_ext[1])), flush=True)
else:
    print('mouth check: the mouth never said A', flush=True)
restore(s0); del s0
print('start-up checks done, %.0f s' % (time.time() - T0), flush=True)


def nxt(cur, n):
    return int(rng.integers(n)) if cur is None or (C['stay'] <= 0) else (cur if rng.random() < C['stay'] else int((cur + 1 + rng.integers(n - 1)) % n))


def block_summary(stats):
    if not stats: return ''
    return ' | '.join('%s: %d trials, said A in %d (%.2f s per trial), B in %d (%.2f s)' % (it, len(v), sum(a > 0 for a, _ in v), np.mean([a for a, _ in v]), sum(b > 0 for _, b in v), np.mean([b for _, b in v]))
                      for it, v in sorted(stats.items()))


test('before training')
# ---------------- stage H: only level 0 learns
upper = np.concatenate([k01, k11, k10, k12]); S['ee_pl'][upper] = 0
t = 0.0; it = None; stats = {}
while t < C['TH'] - 1e-9:
    it = nxt(it, 4); x = SND[it]; em = play(code[x], 1.0, P) + play(None, 2.0, P); t += 3.0
    iv = {g: runs(em, g) for g in 'AB'}; stats.setdefault(x, []).append(tuple(sum(b - a for a, b in iv[g]) for g in 'AB'))
    if iv['A'] or iv['B']: print('    emit H t=%6.1f heard %s: A %s | B %s' % (t - 3.0, x, fmtr(iv['A']), fmtr(iv['B'])), flush=True)
print('stage H done: ' + block_summary(stats), flush=True)
test('end of stage H (%.0f s)' % t); save('H')
motor_test('baseline repeat test after stage H (my addition)', 'ABF')
# ---------------- stage 1: instinctive voice, everything learns
S['ee_pl'][k01] = 1; S['ee_pl'][k11] = 1; S['ee_pl'][k12] = 1; S['ee_pl'][k10] = 2
print('--- stage 1: everything learns', flush=True)
t = 0.0; it = None; stats = {}; nxt_t = C['every']
while t < C['T1'] - 1e-9:
    it = nxt(it, 2); g = 'AB'[it]; em = play(None, 1.5, P, drive=g) + play(None, 4.5, P); t += 6.0
    iv = {h: runs(em, h) for h in 'AB'}; stats.setdefault('drive ' + g, []).append(tuple(sum(b - a for a, b in iv[h]) for h in 'AB'))
    print('    emit 1 t=%6.1f drive %s: A %s | B %s' % (t - 6.0, g, fmtr(iv['A']), fmtr(iv['B'])), flush=True)
    if t >= nxt_t - 1e-9:
        nxt_t += C['every']; print('stage 1, last %.0f s: %s' % (C['every'], block_summary(stats)), flush=True); stats = {}
        test('stage 1 checkpoint at %.0f s' % t)
save('1')
motor_test('TEST 2 (repeat) after stage 1', 'ABF')
# ---------------- stage 3: hearing a regularity
print('--- stage 3: F then A, or G then B', flush=True)
t = 0.0; it = None; stats = {}; nxt_t = C['every']
while t < C['T3'] - 1e-9:
    it = nxt(it, 2); a, b = ('F', 'A') if it == 0 else ('G', 'B'); em = play(code[a], 1.0, P) + play(code[b], 1.0, P) + play(None, 4.0, P); t += 6.0
    iv = {h: runs(em, h) for h in 'AB'}; stats.setdefault(a + b, []).append(tuple(sum(e - s_ for s_, e in iv[h]) for h in 'AB'))
    if iv['A'] or iv['B']: print('    emit 3 t=%6.1f heard %s%s: A %s | B %s' % (t - 6.0, a, b, fmtr(iv['A']), fmtr(iv['B'])), flush=True)
    if t >= nxt_t - 1e-9:
        nxt_t += C['every']; print('stage 3, last %.0f s: %s' % (C['every'], block_summary(stats)), flush=True); stats = {}
        test('stage 3 checkpoint at %.0f s' % t)
save('3')
motor_test('TEST 4 (association + repeat) after stage 3', 'FGAB')
save('3')
print('done, wall %.0f s' % (time.time() - T0), flush=True)
