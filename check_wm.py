"""Reproduce the working-memory network (Loh, Rolls & Deco 2007): spontaneous rates, and persistent firing after a cue."""
import numpy as np, time
from sim import Net
import wm
t0 = time.time(); net = Net(dt=0.1, seed=2); pool = wm.add_wm(net)
cue = net.add('cue', int((pool == 0).sum()), 'poisson', rate=0.0)
net.connect_pairs('cue', 'wm.E', np.arange(cue['n']), np.nonzero(pool == 0)[0], wm.G['E']['ext'], 0, 0.1, 0, ch=0)
net.finalize(); cnt = net.pops['wm.E']['count']
def seg(T, rate, label):
    net.set_rate('cue', rate); r = net.run(T)
    print('  %-34s cued pool %5.1f  other pool %5.1f  the rest %5.1f  inhibitory %5.1f Hz' % (label, cnt[pool == 0].mean() / T * 1e3, cnt[pool == 1].mean() / T * 1e3, cnt[pool == -1].mean() / T * 1e3, r['wm.I']))
print('paper: ' + wm.PUBLISHED)
seg(500.0, 0.0, 'start-up'); seg(1000.0, 0.0, 'no cue (spontaneous)'); seg(500.0, 120.0, 'cue on, 500 ms')
for k in range(4): seg(500.0, 0.0, 'cue off, %d-%d ms later' % (500 * k, 500 * (k + 1)))
print('total %.0f s' % (time.time() - t0))
