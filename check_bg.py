"""Basal ganglia alone: resting firing rates against the ranges given by Lindahl & Hellgren Kotaleski 2016."""
import sys, time
from sim import Net
import bg
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
t0 = time.time(); net = Net(dt=0.1, seed=1); n = bg.add_bg(net, 'bg', scale=scale); net.finalize()
print('built: %d cells, %.2f million synapses, %.0f s' % (net.N, net.nsyn / 1e6, time.time() - t0), flush=True)
net.run(500.0); r = net.run(T)
for k in bg.N_FULL:
    print('  %-3s n=%6d %7.2f Hz' % (k, n[k], r['bg.' + k]) + ('   paper: %s' % bg.PUBLISHED[k] if k in bg.PUBLISHED else ''))
print('total %.0f s' % (time.time() - t0))
