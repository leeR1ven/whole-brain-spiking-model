"""Basal ganglia, copied from the spiking model of Lindahl & Hellgren Kotaleski 2016 (eNeuro), control state (normal
dopamine: all dopamine terms of the paper are zero there). Numbers from its Tables 1-8 (saved in data/).
Populations: striatal projection neurons with D1 / D2 receptors (M1, M2), striatal fast-spiking interneurons (FS),
subthalamic nucleus (ST), external pallidum (GA "arkypallidal", GI "prototypic"), substantia nigra pars reticulata (SN).

NOT from the paper (flagged):
  - equations in the standard form of the named neuron models (the equations were lost when the text was fetched);
  - the voltage dependence of the NMDA synapses is the standard magnesium-block formula, not read in the paper;
  - cells are wired at random with the published number of inputs per cell (the paper's striatum has spatial structure);
  - short-term plasticity: the listed conductance is taken as the size of the first response.
"""
N_FULL = dict(M1=37971, M2=37971, FS=1599, ST=388, GA=329, GI=988, SN=754)
QIF = dict(C_m=15.2, k=1.0, v_t=-29.7, a=0.01, b=-20.0, c=-60.0, v_peak=40.0)
CELL = dict(
    M1=('qif', dict(QIF, v_r=-78.2, d=66.9), [(12, 0), (160, 0, 'nmda'), (8, -74), (11, -74), (87, -74)]),
    M2=('qif', dict(QIF, v_r=-80.0, d=91.0), [(12, 0), (160, 0, 'nmda'), (8, -74), (11, -74), (76, -74)]),
    FS=('fsn', dict(C_m=80.0, k=1.0, v_r=-64.4, v_t=-50.0, v_b=-55.0, a=0.2, b=0.025, c=-60.0, d=0.0, v_peak=25.0), [(12, 0), (6, -74), (17, -74), (66, -74)]),
    GA=('adex', dict(C_m=60.0, g_L=1.0, E_L=-55.1, V_T=-54.7, D_T=2.55, a=2.5, b=105.0, tau_w=20.0, I_e=1.0, V_peak=15.0, V_reset=-60.0), [(5, 0), (12, 0), (5, -65), (6, -65)]),
    GI=('adex', dict(C_m=40.0, g_L=1.0, E_L=-55.1, V_T=-54.7, D_T=1.7, a=2.5, b=70.0, tau_w=20.0, I_e=12.0, V_peak=15.0, V_reset=-60.0), [(5, 0), (12, 0), (5, -65), (6, -65)]),
    SN=('adex', dict(C_m=80.0, g_L=3.0, E_L=-55.8, V_T=-55.2, D_T=1.8, a=3.0, b=200.0, tau_w=20.0, I_e=15.0, V_peak=20.0, V_reset=-65.0), [(5, 0), (12, 0), (5.2, -80), (2.1, -72)]),
    ST=('adex', dict(C_m=60.0, g_L=10.0, E_L=-80.2, V_T=-64.0, D_T=16.2, a=0.3, a_below=-70.0, b=0.05, tau_w=333.0, I_e=5.0, V_peak=15.0, V_reset=-70.0), [(4, 0), (160, 0, 'nmda'), (8, -84)]))
#        pre   post  inputs/cell  g nS   delay  slot  stp(U, tau_rec, tau_fac)
CONN = [('M1', 'M1', 364, 0.15,  1.7, 2, None), ('M1', 'M2', 84,  0.375, 1.7, 2, None),
        ('M2', 'M1', 392, 0.45,  1.7, 2, None), ('M2', 'M2', 504, 0.35,  1.7, 2, None),
        ('FS', 'M1', 16,  6.0,   1.7, 3, (0.29, 902.0, 53.0)), ('FS', 'M2', 11, 6.0, 1.7, 3, (0.29, 902.0, 53.0)),
        ('GA', 'M1', 10,  0.04,  7.0, 4, None), ('GA', 'M2', 10,  0.08,  7.0, 4, None),
        ('FS', 'FS', 10,  1.0,   1.7, 1, (0.29, 902.0, 53.0)),
        ('GI', 'FS', 10,  2.0,   7.0, 2, (0.29, 902.0, 53.0)), ('GA', 'FS', 10, 0.51, 7.0, 3, (0.29, 902.0, 53.0)),
        ('GA', 'GA', 5,   0.33,  1.0, 2, None), ('GI', 'GA', 25,  0.33,  1.0, 2, None),
        ('GA', 'GI', 5,   1.3,   1.0, 2, None), ('GI', 'GI', 25,  1.3,   1.0, 2, None),
        ('M2', 'GI', 500, 2.0,   7.0, 3, None),
        ('ST', 'GA', 30,  0.11,  2.0, 1, None), ('ST', 'GI', 30,  0.35,  2.0, 1, None),
        ('GI', 'ST', 30,  0.08,  1.0, 2, None),
        ('M1', 'SN', 500, 2.0,   7.0, 2, (0.0192, 623.0, 559.0)),
        ('GI', 'SN', 32,  76.0,  3.0, 3, (0.196, 969.0, 0.0)),
        ('ST', 'SN', 30,  0.91,  4.5, 1, (0.35, 800.0, 0.0))]
#        pop  rate Hz (cortex awake / "activation" state)  [(slot, g nS)]
EXT = dict(M1=(546.0, [(0, 0.5), (1, 0.11)]), M2=(722.0, [(0, 0.5), (1, 0.019)]), FS=(787.0, [(0, 0.5)]),
           ST=(250.0, [(0, 0.25), (1, 0.00625)]), GA=(200.0, [(0, 0.5)]), GI=(1530.0, [(0, 0.5)]), SN=(1800.0, [(0, 0.5)]))
PUBLISHED = dict(M1='0.01-2', M2='0.01-2', FS='10-20', SN='20-35')        # Hz, ranges the paper says its model is kept in


def add_bg(net, name='bg', scale=1.0, floor=1, ext=True):
    n = {k: max(floor, int(round(v * scale))) for k, v in N_FULL.items()}
    for k, (model, p, slots) in CELL.items():
        net.add(name + '.' + k, n[k], model, slots=slots, **p)
        if ext:
            rate, lst = EXT[k]
            for ch, g in lst:
                net.poisson_in(name + '.' + k, rate, g, ch)
    for pre, post, conv, g, d, ch, stp in CONN:
        k = min(conv, n[pre])
        net.connect(name + '.' + pre, name + '.' + post, k, g * conv / k, 0.0, d, 0.0, ch=ch, mode='indeg', stp=stp)
    return n
