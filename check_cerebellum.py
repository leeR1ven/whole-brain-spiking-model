"""Reproduce Casali et al. 2019, Table 4: firing rates of the cerebellar network with 1 Hz background on the glomeruli."""
import sys, time
from sim import Net
import cerebellum as cb
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
t0 = time.time(); net = Net(dt=0.1, seed=1); n = cb.add_cerebellum(net, 'cb', scale=scale); net.finalize()
print('built: %d cells, %.2f million synapses, %.0f s' % (net.N, net.nsyn / 1e6, time.time() - t0), flush=True)
net.run(300.0, quiet=True); r = net.run(T, quiet=True)
for k in ['glom', 'grc', 'goc', 'sc', 'bc', 'pc', 'dcn']:
    print('  %-5s n=%6d %7.2f Hz' % (k, n[k], r['cb.' + k]) + ('   published %.1f' % cb.PUBLISHED_RATES[k] if k in cb.PUBLISHED_RATES else ''))
print('total %.0f s' % (time.time() - t0))
