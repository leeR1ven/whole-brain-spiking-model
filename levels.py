"""Networks of several LEVELS built from the reproduced learning network (zenke.py: same cells, same synapses, same
plasticity rules, same compiled kernel). Only the wiring is different:
  input units (S)  ->  level 0  ->  level 1 -> ...   each level = excitatory cells + its own inhibitory cells;
  which level sends to which (forward links, loops inside a level, returns from a higher level) is given as a table.
Used to build the designer's picture: sense / concept level below, prefrontal loop above, returns from the loop.

Mine (not from the paper): all of the wiring choices here. The activity signal of the inhibitory plasticity is still one
number for all excitatory cells of all levels (as in the kernel)."""
import numpy as np
import zenke as Z

NS, DELAY = Z.NS, Z.DELAY


def build(seed, sizes, ni, s_in, ee, wie=0.2, p_i=0.05, alpha=4.0):
    """sizes / ni: excitatory / inhibitory cells per level.
    s_in: {level: (inputs per cell from the S units, start weight)}  - drawn at random over all S units.
    ee:   {(from level, to level): (connection probability, start weight)}"""
    rng = np.random.default_rng(seed); S = {}; ne, nI = int(sum(sizes)), int(sum(ni))
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int); ioff = np.concatenate([[0], np.cumsum(ni)]).astype(int)
    pre, post, w = [], [], []
    for lv, (k, w0) in s_in.items():
        m = rng.random((NS, sizes[lv])) < k / NS; i, j = np.nonzero(m); pre.append(i); post.append(j + off[lv]); w.append(np.full(len(i), w0))
    pre, post, w = np.concatenate(pre), np.concatenate(post), np.concatenate(w); o = np.argsort(pre, kind='stable')
    S['se'] = Z.csr(pre, post, NS, ne); S['se_ws'] = np.zeros((len(pre), 2)); S['se_ws'][:, 0] = w[o]
    pre, post, w = [], [], []
    for (a, b), (p, w0) in ee.items():
        m = rng.random((sizes[a], sizes[b])) < p
        if a == b: np.fill_diagonal(m, False)
        i, j = np.nonzero(m); pre.append(i + off[a]); post.append(j + off[b]); w.append(np.full(len(i), w0))
    pre, post, w = np.concatenate(pre), np.concatenate(post), np.concatenate(w); o = np.argsort(pre, kind='stable')
    S['ee'] = Z.csr(pre, post, ne, ne); S['ee_ws'] = np.zeros((len(pre), 2)); S['ee_ws'][:, 0] = w[o]
    def within(na_list, nb_list, oa, ob, no_self=False):
        pre, post = [], []
        for lv in range(len(sizes)):
            m = rng.random((na_list[lv], nb_list[lv])) < p_i
            if no_self: np.fill_diagonal(m, False)
            i, j = np.nonzero(m); pre.append(i + oa[lv]); post.append(j + ob[lv])
        return np.concatenate(pre), np.concatenate(post)
    i, j = within(ni, sizes, ioff, off); S['ie'] = Z.csr(i, j, nI, ne); S['ie_w'] = np.full(len(i), wie)
    i, j = within(sizes, ni, off, ioff); S['ei'] = Z.csr(i, j, ne, nI)
    i, j = within(ni, ni, ioff, ioff, True); S['ii'] = Z.csr(i, j, nI, nI)
    z = lambda n, v=0.0: np.full(n, v)
    S['E'] = [z(ne, -70e-3), z(ne), z(ne), z(ne), z(ne), z(ne), z(ne)]; S['I'] = [z(nI, -70e-3), z(nI), z(nI), z(nI), z(nI)]
    S['tr'] = [z(ne), z(ne), z(ne), z(NS), z(nI), z(ne, 1200.0)]
    S['stp'] = [z(ne, 1.0), z(ne, 0.2), z(ne, 1.0), z(ne, 0.2), z(NS, 1.0), z(NS, 0.2), np.zeros(ne, np.int64), np.zeros(NS, np.int64)]
    S['dl'] = [np.zeros((DELAY, ne), np.int64), np.zeros((DELAY, ne)), np.zeros((DELAY, ne)), np.zeros(DELAY, np.int64),
               np.zeros((DELAY, nI), np.int64), np.zeros(DELAY, np.int64), np.zeros((DELAY, NS), np.int64), np.zeros((DELAY, NS)), np.zeros(DELAY, np.int64)]
    S['sched'] = np.array([0.0, 0.0, 1e18, 0.0]); S['pf'] = np.array([alpha * ne]); S['step'] = 0
    S['lev'] = np.repeat(np.arange(len(sizes)), sizes); S['off'] = off
    return S


class Net:
    def __init__(self, seed, sizes, ni, s_in, ee, **kw):
        self.S = build(seed, sizes, ni, s_in, ee, **kw); Z.seed(seed + 1000); self.ne = int(sum(sizes)); self.ni = int(sum(ni)); self.sizes = sizes
        self.cnt = np.zeros(self.ne, np.int64); self.cnti = np.zeros(self.ni, np.int64)
        self._t = np.zeros(1, np.int64); self._i = np.zeros(1, np.int32); self._n = np.array([1], np.int64); self._ev = np.zeros((16, 3))

    def par(self, **kw):
        return Z.params(ne=self.ne, **kw)

    def show(self, units, seconds, P):
        """the S units in `units` fire at the pattern rate for `seconds` (None: only the background). Returns the rate
        (Hz) of every excitatory cell during it."""
        S = self.S; n = int(round(seconds / Z.DT))
        if units is None or len(units) == 0:
            S['sched'][:] = (0.0, 0.0, 1e18, 0.0); pats = [np.zeros(1, np.int64)]
        else:
            S['sched'][:] = (1.0, 0.0, 1e18, 0.0); pats = [np.asarray(units, dtype=np.int64)]
        ptr = np.array([0, len(pats[0])], np.int64); self.cnt[:] = 0; self.cnti[:] = 0; self._ev[0, 0] = 0
        Z.run(n, S['step'], P, *S['E'], *S['I'], *S['tr'], *S['stp'], *S['ee'], S['ee_ws'], *S['se'], S['se_ws'], *S['ie'], S['ie_w'],
              S['ei'][0], S['ei'][1], S['ii'][0], S['ii'][1], *S['dl'], S['sched'], S['pf'], ptr, pats[0], self.cnt, self.cnti, self._t, self._i, self._n, self._ev)
        S['step'] += n
        return self.cnt / seconds

    def reset(self):
        """all cells back to rest, nothing travelling (weights and short-term states are kept) - for tests"""
        S = self.S; S['E'][0][:] = -70e-3; S['I'][0][:] = -70e-3
        for x in S['E'][1:] + S['I'][1:]: x[:] = 0.0
        for k in (3, 5, 8): S['dl'][k][:] = 0
        for x in S['tr'][:5]: x[:] = 0.0

    def level(self, lv):
        return slice(int(self.S['off'][lv]), int(self.S['off'][lv + 1]))

    def block(self, a, b):
        """E-E synapses from level a to level b: indices into S['ee_ws'] (rows), sending cells, receiving cells"""
        S = self.S; pre, post = S['ee'][5], S['ee'][1]; m = (S['lev'][pre] == a) & (S['lev'][post] == b); k = np.nonzero(m)[0]
        return k, pre[k], post[k]
