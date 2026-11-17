"""Reproduce Potjans & Diesmann 2014: spontaneous firing rates of the full-scale layered circuit (their Table 6)."""
import sys, time, numpy as np, json
from sim import Net
import cortex
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
kscale = float(sys.argv[3]) if len(sys.argv) > 3 else scale
rates_full = json.load(open('data/column_rates_full.json')) if scale < 1 else None
t0 = time.time(); net = Net(dt=0.1, seed=1)
n = cortex.add_area(net, 'A', scale=scale, rates_full=rates_full, k_scale=kscale)
net.finalize(); print('built: %d neurons, %.1f million synapses, %.0f s' % (net.N, net.nsyn / 1e6, time.time() - t0), flush=True)
net.run(500.0, quiet=True)                                        # let the start-up transient pass
r = net.run(T)
print('scale %.3f, synapses per neuron x %.2f, %.0f ms:' % (scale, kscale, T))
for p in cortex.POPS:
    print('  %-5s %6.2f Hz' % (p, r['A.' + p]) + ('   published %.2f' % cortex.PUBLISHED_RATES[p] if p in cortex.PUBLISHED_RATES else ''))
if scale == 1.0:
    json.dump({p: r['A.' + p] for p in cortex.POPS}, open('data/column_rates_full.json', 'w'))
print('total %.0f s' % (time.time() - t0))
