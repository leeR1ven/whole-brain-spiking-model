"""Compiled inner loops of the simulator (numba). They do exactly what the numpy code in sim.py does, cell by cell;
only the random numbers differ. sim.py uses them when numba can be imported (Net(fast=False) forces the numpy path)."""
import numpy as np
from math import exp
from numba import njit


@njit(cache=True)
def seed(x):
    np.random.seed(x)


@njit(cache=True)
def bg(arr, stride, lo, hi, tot, w, ch):
    """background input: for every piece (cells lo..hi with one rate) draw the number of events, then where they fall"""
    for e in range(lo.shape[0]):
        k = np.random.poisson(tot[e])
        if k > 0:
            span = hi[e] - lo[e]
            for _ in range(k):
                arr[(lo[e] + int(np.random.random() * span)) * stride + ch[e]] += w[e]


@njit(cache=True)
def source(lo, hi, tot, sp):
    ns = 0
    for e in range(lo.shape[0]):
        k = np.random.poisson(tot[e]); span = hi[e] - lo[e]
        for _ in range(k):
            if ns < sp.shape[0]:
                sp[ns] = lo[e] + int(np.random.random() * span); ns += 1
    return ns


@njit(cache=True)
def count(cnt, sp, ns):
    for a in range(ns):
        cnt[sp[a]] += 1


@njit(cache=True)
def lif_cur(V, I, EL, P22, P21, P11, dc, vth, vrs, ref, tref, inp, sp):
    ns = 0
    for i in range(V.shape[0]):
        v = EL[i] + (V[i] - EL[i]) * P22[i] + I[i] * P21[i] + dc[i]
        I[i] = I[i] * P11[i] + inp[i]
        inp[i] = 0.0
        if ref[i] > 0:
            v = vrs[i]; ref[i] -= 1
        if v >= vth[i]:
            v = vrs[i]; ref[i] = tref[i]; sp[ns] = i; ns += 1
        V[i] = v
    return ns


@njit(cache=True)
def cond(model, V, g, wv, dec, Erev, nmda, anynmda, P, jump, vpk, vrs, ref, tref, inp, sp, dt):
    """model 0 lif_cond  P = gL, C_m, E_L, I_e
             1 adex      P = C_m, g_L, E_L, V_T, D_T, a, a_below, tau_w, I_e
             2 qif, 3 fsn  P = C_m, k, v_r, v_t, a, b, I_e, v_b"""
    n, C = g.shape; ns = 0
    for i in range(n):
        v = V[i]; gs = 0.0; gE = 0.0
        blk = 1.0 / (1.0 + exp(-0.062 * v) / 3.57) if anynmda else 1.0
        for c in range(C):
            x = g[i, c] * dec[i, c] + inp[i, c]; g[i, c] = x; inp[i, c] = 0.0
            if nmda[i, c]: x = x * blk
            gs += x; gE += x * Erev[i, c]
        if model == 0:
            gt = P[i, 0] + gs; vinf = (P[i, 0] * P[i, 2] + P[i, 3] + gE) / gt
            v = vinf + (v - vinf) * exp(-dt * gt / P[i, 1])
        elif model == 1:
            ex = exp(min((v - P[i, 3]) / P[i, 4], 20.0)); a = P[i, 5] if v < P[i, 6] else 0.0
            dv = (dt / P[i, 0]) * (-P[i, 1] * (v - P[i, 2]) + P[i, 1] * P[i, 4] * ex - wv[i] + P[i, 8] + gE - gs * v)
            wv[i] += (dt / P[i, 7]) * (a * (v - P[i, 2]) - wv[i]); v += dv
        else:
            dv = (dt / P[i, 0]) * (P[i, 1] * (v - P[i, 2]) * (v - P[i, 3]) - wv[i] + P[i, 6] + gE - gs * v)
            if model == 2:
                wv[i] += dt * P[i, 4] * (P[i, 5] * (v - P[i, 2]) - wv[i])
            else:
                U = P[i, 5] * (v - P[i, 7]) ** 3 if v >= P[i, 7] else 0.0
                wv[i] += dt * P[i, 4] * (U - wv[i])
            v += dv
        if ref[i] > 0:
            v = vrs[i]; ref[i] -= 1
        if v >= vpk[i]:
            v = vrs[i]; ref[i] = tref[i]; wv[i] += jump[i]; sp[ns] = i; ns += 1
        V[i] = v
    return ns


@njit(cache=True)
def lazy(V, g, aw, tup, act, nact, ref, inp, sp, t, dt, wsteps, gL, Cm, EL, Ie, Ee, Ei, dece, deci, ti, vth, vrs, tref):
    """cells that sleep unless an excitatory input has just reached them (see Net._lazy in sim.py). act[:nact] = awake cells"""
    n = V.shape[0]
    for i in range(n):
        e = inp[i, 0]; h = inp[i, 1]
        if e != 0.0 or h != 0.0:
            if aw[i] <= t:                                   # asleep
                gi = g[i, 1] * exp(-(t - tup[i]) * dt / ti)
                if e != 0.0:                                 # wake up
                    g[i, 1] = gi; g[i, 0] = 0.0; V[i] = (gL * EL + gi * Ei + Ie) / (gL + gi); tup[i] = t
                    aw[i] = t + wsteps; act[nact] = i; nact += 1
                else:
                    g[i, 1] = gi + h; tup[i] = t; inp[i, 1] = 0.0
            elif e != 0.0:
                aw[i] = t + wsteps
    ns = 0; m = 0
    for a in range(nact):
        i = act[a]
        if aw[i] <= t:
            continue                                         # went back to sleep
        ge = g[i, 0] * dece + inp[i, 0]; gi = g[i, 1] * deci + inp[i, 1]; inp[i, 0] = 0.0; inp[i, 1] = 0.0
        gt = gL + ge + gi; vinf = (gL * EL + Ie + ge * Ee + gi * Ei) / gt
        v = vinf + (V[i] - vinf) * exp(-dt * gt / Cm)
        if ref[i] > 0:
            v = vrs; ref[i] -= 1
        if v >= vth:
            v = vrs; ref[i] = tref; aw[i] = t + wsteps; sp[ns] = i; ns += 1
        V[i] = v; g[i, 0] = ge; g[i, 1] = gi; tup[i] = t + 1
        act[m] = i; m += 1
    return ns, m


@njit(cache=True)
def deliver(sp, ns, ip, slot, w, d, buf, t, L, S):
    for a in range(ns):
        i = sp[a]
        for k in range(ip[i], ip[i + 1]):
            buf[((t + d[k]) % L) * S + slot[k]] += w[k]


@njit(cache=True)
def deliver_tm(sp, ns, ip, slot, w, d, buf, t, L, S, now, U, trec, tfac, u, x, last):
    for a in range(ns):
        i = sp[a]
        if ip[i + 1] == ip[i]: continue
        el = now - last[i]; xx = 1.0 - (1.0 - x[i]) * exp(-el / trec)
        uu = U + (u[i] - U) * exp(-el / tfac) if tfac > 0 else U
        m = uu * xx / U; x[i] = xx - uu * xx; u[i] = uu + U * (1.0 - uu); last[i] = now
        for k in range(ip[i], ip[i + 1]):
            buf[((t + d[k]) % L) * S + slot[k]] += w[k] * m


@njit(cache=True)
def deliver_nmda(sp, ns, ip, slot, w, d, buf, t, L, S, now, tau, a_, s_, last):
    for a in range(ns):
        i = sp[a]
        if ip[i + 1] == ip[i]: continue
        sj = s_[i] * exp(-(now - last[i]) / tau); m = a_ * (1.0 - sj); s_[i] = sj + m; last[i] = now
        for k in range(ip[i], ip[i + 1]):
            buf[((t + d[k]) % L) * S + slot[k]] += w[k] * m


# ---------------------------------------------------------------- synapses that change (plasticity)
# Rule of Zenke, Agnes & Gerstner 2015 as written in the authors' code (see zenke.py, which reproduces their network).
# Weights of such synapses are kept as plain numbers w (their start value is 0.1 in the paper); `unit` turns them into
# the current the cell receives (one value per synapse). z20 / z100 are each cell's spike traces (decay 20 ms / 100 ms).

@njit(cache=True)
def lif_cur2(V, I, I2, EL, P22, P21, P11, P21b, P11b, dc, vth, vrs, ref, tref, inp, sp):
    """lif_cur with a second, slow synaptic current (inp[:, 1])"""
    ns = 0
    for i in range(V.shape[0]):
        v = EL[i] + (V[i] - EL[i]) * P22[i] + I[i] * P21[i] + I2[i] * P21b[i] + dc[i]
        I[i] = I[i] * P11[i] + inp[i, 0]; I2[i] = I2[i] * P11b[i] + inp[i, 1]
        inp[i, 0] = 0.0; inp[i, 1] = 0.0
        if ref[i] > 0:
            v = vrs[i]; ref[i] -= 1
        if v >= vth[i]:
            v = vrs[i]; ref[i] = tref[i]; sp[ns] = i; ns += 1
        V[i] = v
    return ns


@njit(cache=True)
def traces(sp, ns, z20, z100, zh, d20, d100, dh):
    """zh: very slow trace (the cell's long-run rate = zh / its decay time); an empty array if not used"""
    for a in range(ns):
        z20[sp[a]] += 1.0; z100[sp[a]] += 1.0
    for i in range(z20.shape[0]): z20[i] *= d20
    for i in range(z100.shape[0]): z100[i] *= d100
    if zh.shape[0] > 0:
        for a in range(ns): zh[sp[a]] += 1.0
        for i in range(zh.shape[0]): zh[i] *= dh


@njit(cache=True)
def pl_pre(sp, ns, ip, post, slot, ws, d, buf, t, L, S, zpost, unit, a_fast, slow_gain, A, delta, stp, U, taud, tauf, dt, x, u, last, on, hom, zh, tauh_s, kappa):
    """spikes of sending cells: deliver (through short-term depression / facilitation; a share into the slow channel of
    the receiving cell), then  w += -A z20(receiver) + delta.   hom: the depression is multiplied by
    (long-run rate of the receiver / kappa)^2 - the 'sliding threshold' of the authors' sim_rc_p11 / the paper's eq. 17-18"""
    for a in range(ns):
        i = sp[a]
        if ip[i + 1] == ip[i]: continue
        m = 1.0
        if stp:
            el = (t - last[i]) * dt; last[i] = t
            xx = 1.0 - (1.0 - x[i]) * exp(-el / taud); uu = U + (u[i] - U) * exp(-el / tauf)
            m = xx * uu / U; x[i] = xx - uu * xx; u[i] = uu + U * (1.0 - uu)
        for k in range(ip[i], ip[i + 1]):
            val = ws[k, 0] * unit[k] * m; b = ((t + d[k]) % L) * S + slot[k]
            buf[b] += a_fast * val
            if slow_gain != 0.0: buf[b + 1] += slow_gain * val
            if on:
                j = post[k]
                if hom:
                    r = zh[j] / tauh_s / kappa; w = ws[k, 0] - A * zpost[j] * r * r + delta
                else:
                    w = ws[k, 0] - A * zpost[j] + delta
                ws[k, 0] = w if w > 0.0 else 0.0


@njit(cache=True)
def pl_post(sp, ns, bptr, bpos, bpre, ws, zpre, z20, z100, A, beta, wmax):
    """spikes of receiving cells:  w += A z100(receiver) z20(sender) - beta z20(receiver)^3 (w - reference)"""
    for a in range(ns):
        i = sp[a]
        if bptr[i + 1] == bptr[i]: continue
        zs = A * z100[i]; z3 = beta * z20[i] ** 3
        for q in range(bptr[i], bptr[i + 1]):
            k = bpos[q]; w = ws[k, 0] + zs * zpre[bpre[q]] - z3 * (ws[k, 0] - ws[k, 1])
            ws[k, 0] = wmax if w > wmax else (0.0 if w < 0.0 else w)


@njit(cache=True)
def consolidate(ws, pot, wa, wb, wc, rate):
    """reference weight: follows the weight, in a double well with stable values wa and wc"""
    for k in range(ws.shape[0]):
        s_ = ws[k, 1]; ws[k, 1] = s_ + ((ws[k, 0] - s_) + pot * (wa - s_) * (wb - s_) * (wc - s_)) * rate


@njit(cache=True)
def inh_pre(sp, ns, ip, post, slot, ws, d, buf, t, L, S, zpost, lab, gm, unit, on):
    """inhibitory synapses onto excitatory cells; gm[label] = eta (population rate / target - 1) of the receiving cells"""
    for a in range(ns):
        i = sp[a]
        for k in range(ip[i], ip[i + 1]):
            buf[((t + d[k]) % L) * S + slot[k]] -= ws[k, 0] * unit[k]
            if on:
                j = post[k]; w = ws[k, 0] + gm[lab[j]] * (zpost[j] + 1.0)
                ws[k, 0] = w if w > 0.0 else 0.0


@njit(cache=True)
def inh_post(sp, ns, bptr, bpos, bpre, ws, zpre, lab, gm, wmax, cnt):
    for a in range(ns):
        i = sp[a]
        if bptr[i + 1] == bptr[i]: continue
        g = gm[lab[i]]; cnt[lab[i]] += 1.0
        for q in range(bptr[i], bptr[i + 1]):
            k = bpos[q]; w = ws[k, 0] + g * zpre[bpre[q]]
            ws[k, 0] = wmax if w > wmax else w
