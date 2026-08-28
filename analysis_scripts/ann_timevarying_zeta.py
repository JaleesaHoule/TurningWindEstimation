#!/opt/anaconda3/envs/Python311/bin/python
"""
ANN estimate of a constantly changing wind direction, on a one-turn flight.

ζ is a constant-parameter state in the fly model, so to make the wind direction
sweep we impose a time-varying ζ(t) in the TRUTH and regenerate the angular
measurements from it. The ambient wind holds a fixed speed w but its direction
follows a downward parabola

        ζ(t) = −π · (2t/T − 1)²      peak 0 at mid-flight, −π at both ends
                                     (−π ≡ +π on the circle: "−π → 0 → π")

Two measurement-noise scenarios are shown: clean (no noise) inputs, and high noise
(σ = 0.1 rad ≈ 5.7°, 10× the net's training augmentation) added to (φ, ψ, γ).

Output → ann_observability_probe_output/.
"""
import os
import sys
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import turn_control_sweep as TC
from twe.observability import ObservabilityEngine, SYSTEMS
from twe.paths import output_dir
from twe.zeta_ann_estimator import CausalZetaANN

OUT = output_dir('ann_observability_probe_output')
os.makedirs(OUT, exist_ok=True)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
HIGH_NOISE = 0.1                                   # rad, on the angular sensors


def main():
    spec = SYSTEMS['fly']()
    engine = ObservabilityEngine(spec)
    ann = CausalZetaANN(spec)
    DT = spec.dt
    zi = spec.state_names.index('zeta')
    xi, yi = spec.state_names.index('x'), spec.state_names.index('y')
    sidx = [spec.measurement_names.index(m) for m in ('phi', 'psi', 'gamma')]

    # SAME trajectory as ann_traces_by_true_zeta: straight → 90° turn → straight
    speed, heading, k0, nt, meta = TC.vee_setpoints(DT, 90.0, 10.0, TC.V_CRUISE)
    N = len(speed)
    W_WIND = TC.WIND                               # 0.4 m/s, as in that figure
    sim = engine._get_mpc_sim()
    sim.update_dict(spec.make_setpoint(speed, heading, W_WIND, 0.0), name='setpoint')
    with contextlib.redirect_stdout(io.StringIO()):
        tt, x, u, _ = sim.simulate(x0=None, mpc=True, return_full_output=True)
    X = np.vstack(list(x.values())).T
    U = np.vstack(list(u.values())).T
    N = len(tt); t = np.arange(N) * DT

    # time-varying TRUE wind direction: downward parabola −π → 0 → −π(≡π)
    uu = 2 * t / t[-1] - 1
    zeta_true = -np.pi * uu ** 2
    Xz = X.copy(); Xz[:, zi] = zeta_true
    Y = np.array([np.asarray(spec.h(Xz[k], U[k]), float) for k in range(N)])

    rng = np.random.default_rng(0)
    def ann_est(sigma):
        Yn = Y.copy()
        if sigma > 0:
            Yn[:, sidx] += rng.normal(0.0, sigma, Yn[:, sidx].shape)
        return ann.predict_series(Yn)

    scen = [('clean', 0.0, '#1a9850', 14),
            (f'σ={HIGH_NOISE} rad noise', HIGH_NOISE, '#d62728', 12)]
    print(f'ONE-TURN flight: N={N} steps, T={t[-1]:.2f}s, w={W_WIND} m/s, '
          f'turn steps {k0}..{k0+nt}')
    zhats = {}
    for lbl, sg, _, _ in scen:
        zh = ann_est(sg)
        zhats[lbl] = zh
        e = wrap(zh - zeta_true); fin = np.isfinite(e)
        print(f'  {lbl:20s}: median |err| = {np.nanmedian(np.abs(e)):.3f} rad '
              f'({np.degrees(np.nanmedian(np.abs(e))):.1f}°), '
              f'RMSE = {np.sqrt(np.nanmean(e[fin]**2)):.3f} rad')

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5),
                           gridspec_kw={'width_ratios': [1, 1.3]})
    sc = ax[0].scatter(Xz[:, xi], Xz[:, yi], c=zeta_true, cmap='twilight',
                       s=16, vmin=-np.pi, vmax=np.pi, zorder=3)
    ax[0].plot(Xz[:, xi], Xz[:, yi], '-', color='0.85', lw=0.6, zorder=0)
    ax[0].plot(Xz[0, xi], Xz[0, yi], 'o', color='#1a9850', ms=9, mec='k', zorder=5)
    ax[0].set_aspect('equal'); ax[0].grid(alpha=.3)
    ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]')
    ax[0].set_title('one-turn flight; colour = TRUE wind direction ζ')
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label('true ζ [rad]')

    ax[1].plot(t, zeta_true, 'k-', lw=2.6, label='TRUE ζ (downward parabola)',
               zorder=2)
    for lbl, sg, col, ms in scen:
        zp = zeta_true + wrap(zhats[lbl] - zeta_true)     # hug the true curve
        ax[1].scatter(t, zp, s=ms, color=col, alpha=0.85, edgecolors='none',
                      zorder=4, label=f'ANN ζ̂ — {lbl}')
    for yv in (-np.pi, np.pi):
        ax[1].axhline(yv, color='0.6', ls=':', lw=1)
    ax[1].text(t[-1] * 0.99, -np.pi + 0.06, '−π ≡ +π', ha='right', va='bottom',
               fontsize=8, color='0.4')
    ax[1].axvspan(t[k0], t[min(k0 + nt, N - 1)], color='0.5', alpha=0.15,
                  label='the turn')
    ax[1].set_ylim(-np.pi - 0.35, np.pi + 0.35)
    ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_yticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
    ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('wind direction ζ [rad]')
    ax[1].grid(alpha=.3); ax[1].legend(loc='upper right', fontsize=8)
    ax[1].set_title(f'ANN estimate vs time-varying ζ  (one turn, w={W_WIND} m/s)')

    fig.suptitle('ζ-ANN tracking a CONSTANTLY-CHANGING wind direction on the '
                 'one-turn flight (ζ swept −π → 0 → π)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, 'ann_timevarying_zeta.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print('→', p)


if __name__ == '__main__':
    main()
