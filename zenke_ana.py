"""Analysis of the runs of run_zenke.py, done the way the paper describes (Methods, "Determining readout populations"):
a cell codes for a shape if its firing rate while that shape is shown is above 10 Hz; the 'activity' of a shape is the
mean rate of its cells.   usage: python3 zenke_ana.py TAG [init|learn|cue]"""
import sys, os, numpy as np
import zenke as Z
OUT = os.path.join(Z.HERE, 'data/zenke_run'); NAMES = ['circle', 'square', 'cross', 'triangle']


def load(tag, name):
    d = np.load(os.path.join(OUT, '%s_%s_spikes.npz' % (tag, name))); return d['t'], d['i'], d['ev'], int(d['t_end'])


def intervals(ev, t_first, t_end):
    """lists of (start step, end step, pattern) for the times a pattern is shown / for the pauses after it"""
    on, off = [], []
    for k in range(len(ev)):
        a = int(ev[k, 0]); b = int(ev[k + 1, 0]) if k + 1 < len(ev) else t_end
        if a < t_first: continue
        if ev[k, 1] > 0: on.append((a, b, int(ev[k, 2])))
        elif k > 0: off.append((a, b, int(ev[k, 2])))          # a pause counts only if its shape was shown in this stage
    return on, off


def counts(t, i, a, b):
    lo, hi = np.searchsorted(t, [a, b]); return np.bincount(i[lo:hi], minlength=Z.NE)


def evoked(tag, name='init', npat=4):
    """rate of every E cell while each pattern is shown (Hz), from the recorded part of a stage"""
    t, i, ev, t_end = load(tag, name); on, off = intervals(ev, t[0] if len(t) else 0, t_end)
    c = np.zeros((npat, Z.NE)); T = np.zeros(npat)
    for a, b, p in on: c[p] += counts(t, i, a, b); T[p] += (b - a) * Z.DT
    return c / np.maximum(T, 1e-9)[:, None], T


def sets(tag):
    r, T = evoked(tag, 'init'); return [np.nonzero(r[p] > 10.0)[0] for p in range(4)], r


def report(tag, name):
    M, r = sets(tag); t, i, ev, t_end = load(tag, name); on, off = intervals(ev, t[0], t_end)
    rest = np.setdiff1d(np.arange(Z.NE), np.concatenate(M)); groups = M + [rest]
    print('=== %s / %s: recorded %.0f s, %d presentations' % (tag, name, (t_end - t[0]) * Z.DT, len(on)))
    if name == 'init':
        print('cells coding for each shape (rate > 10 Hz while it is shown): ' + ', '.join('%s %d' % (NAMES[p], len(M[p])) for p in range(4)) +
              '; for none %d (%.0f %%); for more than one %d' % (len(rest), 100 * len(rest) / Z.NE, (np.sum(r > 10, axis=0) > 1).sum()))
        print('mean rate of each group (rows) while each shape is shown (columns), Hz:')
        for g in range(5):
            print('  %-9s' % (NAMES[g] if g < 4 else 'none') + ' '.join('%6.1f' % r[p, groups[g]].mean() if len(groups[g]) else '   nan' for p in range(4)))
        d = np.load(os.path.join(OUT, '%s_%s_weights.npz' % (tag, name))); pre, post, w = d['ee_pre'], d['ee_post'], d['ee_ws'][:, 0]
        lab = np.full(Z.NE, 4); only = np.sum(r > 10, axis=0) == 1
        for p in range(4): lab[M[p][only[M[p]]]] = p
        print('mean E-E weight from group (row) to group (column), cells coding for exactly one shape; start value 0.100:')
        for a in range(5):
            row = []
            for b in range(5):
                m = (lab[pre] == a) & (lab[post] == b); row.append('%6.3f' % w[m].mean() if m.any() else '   nan')
            print('  %-9s' % (NAMES[a] if a < 4 else 'none') + ' '.join(row))
    # pauses: activity of every group from 1 s after the shape is switched off until the next one
    npat = int(ev[:, 2].max()) + 1; res = {}
    for a, b, p in off:
        if (b - a) * Z.DT < 2.0: continue
        c = counts(t, i, a + 10000, b); dur = (b - a - 10000) * Z.DT
        res.setdefault(p, []).append([c[g].mean() / dur if len(g) else 0.0 for g in groups])
    if name in ('init', 'learn'):
        print('pauses longer than 2 s; activity (Hz) of each group from 1 s after switch-off to the next shape; rows = shape shown before:')
        print('  %-9s' % '' + ' '.join('%8s' % n for n in NAMES + ['none']) + '   pauses   own group > 10 Hz')
        for p in sorted(res):
            x = np.array(res[p]); print('  %-9s' % NAMES[p] + ' '.join('%8.1f' % v for v in x.mean(0)) + '   %5d   %5.0f %%' % (len(x), 100 * (x[:, p] > 10).mean()))
    else:
        kind = ['circle'] * 4 + ['square'] * 4 + ['cross'] * 4 + ['triangle'] * 4 + ['other'] * 4
        print('partial / distorted cues (4 per shape, 4 unrelated); which group is most active in the pause after the cue (> 10 Hz), number of pauses:')
        print('  %-9s' % 'cue of' + ' '.join('%8s' % n for n in NAMES + ['no group']))
        tab = {}
        for p in sorted(res):
            for x in res[p]:
                k = int(np.argmax(x[:4])) if max(x[:4]) > 10 else 4
                tab.setdefault(kind[p], np.zeros(5, int))[k] += 1
        for kd in ['circle', 'square', 'cross', 'triangle', 'other']:
            if kd in tab: print('  %-9s' % kd + ' '.join('%8d' % v for v in tab[kd]))
    c = counts(t, i, t[0], t_end) / ((t_end - t[0]) * Z.DT)
    print('whole recorded part: mean rate of coding cells %.1f Hz, of the others %.2f Hz' % (c[np.concatenate(M)].mean() if len(np.concatenate(M)) else 0, c[rest].mean()), flush=True)


if __name__ == '__main__':
    for nm in (sys.argv[2:] or ['init', 'learn', 'cue']):
        if os.path.exists(os.path.join(OUT, '%s_%s_spikes.npz' % (sys.argv[1], nm))): report(sys.argv[1], nm)
