#!/opt/anaconda3/envs/Python311/bin/python
"""
CONSTANT-wind runs of the causal ζ-ANN on the FLY 90°-turn vee flight
(straight → 90° turn → straight, no upwind crossing) — the same trajectory
as the earlier fly demos — fed via the 1:1 measurement mapping
fly [phi, psi, gamma] = [heading, course, airflow] = the net's inputs.
FORWARD window only (causal ANN uses past-only context by design).
Also plots the trajectory coloured by ζ observability.

  fig 1  ann_papernet_constant_fwd.png   ζ̂(t) forward, a few constant ζ
  fig 2  ann_papernet_traj_minev.png      x-y vee path coloured by √(min-EV(ζ))
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import twe.observability as OG
from twe.paths import output_dir
import turn_control_sweep as TC
from twe.zeta_ann_estimator import CausalZetaANN

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
ZETAS = [0.0, 1.0, -2.0]                       # off the flight's upwind axis
NET_INPUTS = ('phi', 'psi', 'gamma')            # fly meas = [heading, course, airflow]
ANG = NET_INPUTS
LAM = 1e-6

OUT = output_dir('ann_observability_probe_output')
os.makedirs(OUT, exist_ok=True)


def main():
    spec = OG.SYSTEMS['fly']()
    engine = OG.ObservabilityEngine(spec)
    ann = CausalZetaANN(spec)                   # loads zeta_ann_causal/network.h5
    W = ann.time_steps                          # replaces the imported T constant
    sidx = [spec.measurement_names.index(m) for m in NET_INPUTS]
    DT = spec.dt
    ix, iy = spec.state_names.index('x'), spec.state_names.index('y')
    q_diag = {s: 1e-14 for s in spec.state_names}; q_diag['zeta'] = 1e-2
    r_diag = {m: 0.01 for m in spec.measurement_names}

    speed, heading, k0, nt, meta = TC.vee_setpoints(DT, 90.0, 10.0, TC.V_CRUISE)
    W_WIND = TC.WIND
    gs = TC.V_CRUISE
    t_turn0, t_turn1 = meta['t_leg'], meta['t_leg'] + meta['t_turn']

    traj, evmap = {}, {}
    for zc in ZETAS:
        setpoint = spec.make_setpoint(speed, heading, W_WIND, zc)
        t, X, U = engine.simulate_mpc(setpoint)
        Y = np.array([np.asarray(spec.h(X[k], U[k]), float) for k in range(len(t))])
        zf = ann.predict_series(Y)              # takes full Y, does its own column selection
        traj[zc] = dict(t=t, X=X.copy(), zf=zf)
        df = engine.linearized_ev(W, q_diag, r_diag, LAM, sensors=ANG, states=('zeta',))
        idx = np.clip(np.round(df['time'].values / DT).astype(int), 0, len(t) - 1)
        evmap[zc] = (idx, df['zeta'].values)
        e = wrap(zf - zc); fin = np.isfinite(e)
        tt, vv = df['time'].values, df['zeta'].values
        seg = lambda lo, hi: np.nanmedian(vv[(tt >= lo) & (tt < hi)])
        print(f'ζ={zc:+.1f}: forward median |err| {np.degrees(np.nanmedian(np.abs(e))):.1f}°, '
              f'RMSE {np.sqrt(np.nanmean(e[fin]**2)):.3f};  '
              f'min-EV(ζ) pre-turn {seg(0,t_turn0):.2g} / turn {seg(t_turn0,t_turn1):.2g} '
              f'/ post {seg(t_turn1,1e9):.2g}')

    # ---------- fig 1: constant-wind traces, forward only ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    for zc in ZETAS:
        d = traj[zc]
        ln, = ax.plot(d['t'], wrap(d['zf'] - zc) + zc, '-', lw=2.2,
                      label=f'ζ̂ forward  (true ζ={zc:+.1f})')
        ax.axhline(zc, color=ln.get_color(), ls='--', lw=1)
    ax.axvspan(t_turn0, t_turn1, color='0.5', alpha=0.2, label='the turn')
    ax.set_xlabel('time [s]'); ax.set_ylabel('ζ [rad]')
    ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'Causal ζ-ANN on the fly 90°-turn flight (ground speed = {gs:.1f}m/s) '
                 'forward window; dashed = true ζ')
    fig.tight_layout()
    p = os.path.join(OUT, 'ann_mynet_constant.png')
    fig.savefig(p, dpi=150); plt.close(fig); print('→', p)

    # ---------- fig 2: trajectory coloured by ζ observability ----------
    allev = np.concatenate([np.sqrt(evmap[zc][1]) for zc in ZETAS])
    allev = allev[np.isfinite(allev) & (allev > 0)]

    fig, axs = plt.subplots(1, len(ZETAS), figsize=(4.6 * len(ZETAS), 4.4), squeeze=False)
    for j, zc in enumerate(ZETAS):
        ax = axs[0, j]
        d = traj[zc]; idx, ev = evmap[zc]
        ax.plot(d['X'][:, ix], d['X'][:, iy], '-', color='0.85', lw=0.8, zorder=0)
        sc = ax.scatter(d['X'][idx, ix], d['X'][idx, iy], c=ev,
                        cmap='viridis_r',
                        norm=LogNorm(vmin=1e-1, vmax=1e6),
                        s=26, zorder=3)
        ax.plot(d['X'][0, ix], d['X'][0, iy], 'o', color='#1a9850', ms=9,
                mec='k', zorder=5)
        ax.set_aspect('equal'); ax.grid(alpha=.25)
        ax.set_title(f'ζ = {zc:+.1f} rad'); ax.set_xlabel('x [m]')
        if j == 0:
            ax.set_ylabel('y [m]')
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$\sigma_\zeta=\min\,EV(\zeta)$', fontsize=8)
    fig.suptitle(f'Fly 90°-turn flight (ground speed = {gs:.1f}m/s) coloured by ζ observability (min error '
                'variance of ζ from φ, ψ, γ)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, 'ann_mynet_traj_minev.png')
    fig.savefig(p, dpi=150); plt.close(fig); print('→', p)


if __name__ == '__main__':
    main()
