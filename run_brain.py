"""Build the small whole brain, let it idle, print firing rates next to the published ones, then give the ear a short sound."""
import time, json, numpy as np
import brain, cortex, cerebellum, bg
t0 = time.time(); net, info = brain.build(seed=1)
c = info['counts']; tot = sum(c.values())
print('built in %.0f s: %d neurons, %.1f million synapses' % (time.time() - t0, tot, net.nsyn / 1e6))
print('share of neurons: cortex+hippocampus %.1f %% (human 19), cerebellum %.1f %% (human 80), the rest %.1f %% (human ~1)' % (100.0 * c['cortex'] / tot, 100.0 * c['cerebellum'] / tot, 100.0 * c['rest'] / tot))
net.run(500.0); T = 1000.0; t1 = time.time(); r = net.run(T); ar = brain.area_rates(net, info, T)
print('idle, %.0f ms (%.0f s of computing)' % (T, time.time() - t1))
rf = json.load(open('data/column_rates_full.json'))
print('cortex, Hz by layer (full-size circuit: ' + ' '.join('%s %.1f' % (p, rf[p]) for p in cortex.POPS) + ')')
for a in brain.AREAS:
    print('  %-4s ' % a + ' '.join('%5.1f' % ar[a][p] for p in cortex.POPS))
tc = net.pops['th.TC']['count']
print('thalamus relay cells: ' + ' '.join('%s %.1f' % (k, tc[info['tspan'](k)[0]:info['tspan'](k)[1]].mean() / (T * 1e-3)) for k in info['NUC']) + ' ; reticular %.1f Hz' % r['th.RT'])
print('hippocampus: ' + ' '.join('%s %.1f' % (k[3:], r[k]) for k in r if k.startswith('hp.')))
print('basal ganglia: ' + ' '.join('%s %.1f' % (k[3:], r[k]) for k in r if k.startswith('bg.')) + '   (paper: M1, M2 0.01-2; FS 10-20; SN 20-35)')
print('cerebellum: ' + ' '.join('%s %.1f' % (k[3:], r[k]) for k in r if k.startswith('cb.')) + '   (paper: ' + ' '.join('%s %.1f' % kv for kv in cerebellum.PUBLISHED_RATES.items()) + ')')
json.dump(dict(rates=r, areas=ar), open('data/brain_v2_idle.json', 'w'))
# a sound: the 100 ear fibres fire at 100 Hz for 50 ms
fib = np.arange(100)
def sound(step):
    return {'ear': fib[np.random.default_rng(step).random(100) < 100.0 * net.dt * 1e-3]}
show = [('A1', 'L4e'), ('A1', 'L23e'), ('AB', 'L4e'), ('AB', 'L23e'), ('PB', 'L4e'), ('PB', 'L23e'), ('PFi', 'L23e'), ('V1', 'L4e')]
print('a 50 ms sound to the ear. Hz in ' + ', '.join('%s.%s' % s_ for s_ in show))
for label, T, drv in (('before', 50.0, None), ('during', 50.0, sound), ('0-50 ms after', 50.0, None), ('50-100 ms after', 50.0, None)):
    net.run(T, drive=drv); ar = brain.area_rates(net, info, T)
    print('  %-16s' % label + ' '.join('%6.1f' % ar[a][p] for a, p in show) + '   MGB relay %.1f' % (net.pops['th.TC']['count'][info['tspan']('MGB')[0]:info['tspan']('MGB')[1]].mean() / (T * 1e-3)))
print('total %.0f s' % (time.time() - t0))
