#!/opt/anaconda3/envs/Python311/bin/python
"""
Does the ζ-ANN actually estimate wind direction, or does it fill in a prior on the
sections the observability analysis marks unobservable?

Test: fix ONE commanded flight — straight → 90° turn → straight at the fly cruise
speed — and sweep the TRUE constant wind direction ζ. For each ζ we
fly it (MPC), read the 3 angular measurements (φ, ψ, γ) off the achieved state,
and run the trained causal ζ-ANN.

Outputs → ann_observability_probe_output/.
"""
import os
import sys

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
def circmean(a):
    a = a[np.isfinite(a)]
    return np.nan if not len(a) else np.arctan2(np.sin(a).mean(), np.cos(a).mean())


def main():
    spec = SYSTEMS['fly']()
    engine = ObservabilityEngine(spec)
    ann = CausalZetaANN(spec)                 # loads zeta_ann_causal/network.h5
    W = ann.time_steps
    DT = spec.dt

    # ONE fixed commanded flight (family-b geometry): straight → 90° turn → straight
    AMP, OMEGA_TURN, V = 90.0, 10.0, TC.V_CRUISE
    speed, heading, k0, nt, meta = TC.vee_setpoints(DT, AMP, OMEGA_TURN, V)
    N = len(speed)
    t = np.arange(N) * DT
    t_turn0, t_turn1 = meta['t_leg'], meta['t_leg'] + meta['t_turn']
    pre_steps = np.where(t < t_turn0)[0]
    post_steps = np.where(t > t_turn1)[0]

    ANG = ('phi', 'psi', 'gamma')
    sidx = [spec.measurement_names.index(m) for m in ANG]
    q_diag = dict(spec.q_tiny); q_diag['zeta'] = 1e-2      # matches the coloring
    r_diag = dict(spec.r_default)
    HIGH_NOISE = 0.1                                        # rad, on the sensors
    rng = np.random.default_rng(0)

    def ann_est(Y, sigma):
        Yn = Y.copy()
        if sigma > 0:
            Yn[:, sidx] += rng.normal(0.0, sigma, Yn[:, sidx].shape)
        return ann.predict_series(Yn)

    print(f'flight: N={N} steps, turn on [{t_turn0:.3f},{t_turn1:.3f}] s '
          f'(steps {k0}..{k0+nt}); ANN window W={W}')
    print(f'pre-turn steps {pre_steps[0]}..{pre_steps[-1]}, '
          f'post-turn {post_steps[0]}..{post_steps[-1]}')

    zetas = [1,0,-2]#np.linspace(-np.pi, np.pi, 29)
    zhat_pre, zhat_post = [], []
    rmse_pre, rmse_post = [], []
    ev_pre, ev_post = [], []
    per_flight = {}
    for zt in zetas:
        try:
            tt, X, U = TC._mpc(engine, spec, speed, heading, float(zt))
            if X.shape[0] != N or not np.all(np.isfinite(X)):
                raise ValueError('infeasible')
        except Exception:
            for L in (zhat_pre, zhat_post, rmse_pre, rmse_post, ev_pre, ev_post):
                L.append(np.nan)
            continue
        engine.set_trajectory(tt, X, U)                    # so linearized_ev uses it
        Y = np.array([np.asarray(spec.h(X[k], U[k]), float) for k in range(N)])
        zhat = ann_est(Y, 0.0)                             # clean ζ̂ per step
        zhat_n = ann_est(Y, HIGH_NOISE)                    # high-noise ζ̂
        err = np.abs(wrap(zhat - zt))

        # observability profile: marginal min-EV(ζ) from the 3 angular sensors
        df = engine.linearized_ev(W, q_diag, r_diag, TC.LAM,
                                  sensors=ANG, states=('zeta',))
        evv, evt = df['zeta'].values, df['time'].values
        ev_pre.append(np.nanmedian(evv[evt < t_turn0]))
        ev_post.append(np.nanmedian(evv[evt > t_turn1]))

        zhat_pre.append(circmean(zhat[pre_steps]))
        zhat_post.append(circmean(zhat[post_steps]))
        rmse_pre.append(np.sqrt(np.nanmean(err[pre_steps] ** 2)))
        rmse_post.append(np.sqrt(np.nanmean(err[post_steps] ** 2)))
        # keep per-flight traces for a few representative ζ (all of them when the
        # sweep is short, else the near-0 / a low / a high value)
        keep = (len(zetas) <= 6
                or np.isclose(zt, zetas[np.argmin(np.abs(zetas))])
                or np.isclose(zt, zetas[3]) or np.isclose(zt, zetas[-4]))
        if keep:
            per_flight[round(float(zt), 2)] = (t.copy(), zhat.copy(),
                                               zhat_n.copy(), float(zt))

    zhat_pre = np.array(zhat_pre); zhat_post = np.array(zhat_post)
    rmse_pre = np.array(rmse_pre); rmse_post = np.array(rmse_post)
    ev_pre = np.array(ev_pre); ev_post = np.array(ev_post)

    # best-fit "collapsed prior": the constant the pre-turn estimate looks like
    c_pre = circmean(zhat_pre)
    print(f'\npre-turn  median min-EV(ζ)  = {np.nanmedian(ev_pre):.3g}  (high ⇒ unobservable)')
    print(f'post-turn median min-EV(ζ)  = {np.nanmedian(ev_post):.3g}  (low  ⇒ observable)')
    print(f'pre-turn  mean |error|      = {np.nanmean(rmse_pre):.3f} rad')
    print(f'post-turn mean |error|      = {np.nanmean(rmse_post):.3f} rad')
    print(f'pre-turn estimate clusters near ζ̂≈{c_pre:+.2f} rad regardless of truth'
          if np.nanstd(zhat_pre) < 0.5 else
          'pre-turn estimate tracks truth (std %.2f)' % np.nanstd(zhat_pre))

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ideal = np.linspace(-np.pi, np.pi, 100)

    ax[0].plot(ideal, ideal, 'k--', lw=1, label='perfect tracking (ζ̂=ζ)')
    ax[0].plot(zetas, zhat_pre, 'o-', color='#d62728',
               label='ANN mean ζ̂ — PRE-turn (unobservable)')
    ax[0].plot(zetas, zhat_post, 's-', color='#1f77b4',
               label='ANN mean ζ̂ — POST-turn (observable)')
    ax[0].axhline(c_pre, color='#d62728', ls=':', lw=1,
                  label=f'pre-turn cluster ζ̂≈{c_pre:+.2f}')
    ax[0].set_xlabel('TRUE wind direction ζ [rad]')
    ax[0].set_ylabel('ANN estimate ζ̂ [rad]')
    ax[0].set_title('Does the ANN track the true ζ?')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_aspect('equal')

    ax[1].plot(zetas, rmse_pre, 'o-', color='#d62728', label='PRE-turn (unobservable)')
    ax[1].plot(zetas, rmse_post, 's-', color='#1f77b4', label='POST-turn (observable)')
    ax[1].plot(ideal, np.abs(wrap(ideal - c_pre)), 'r:', lw=1,
               label='|ζ − prior| (pure guessing)')
    ax[1].set_xlabel('TRUE wind direction ζ [rad]')
    ax[1].set_ylabel('ANN |ζ̂ − ζ| [rad]')
    ax[1].set_title('ANN error vs true ζ')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    ax[2].semilogy(zetas, ev_pre, 'o-', color='#d62728', label='PRE-turn')
    ax[2].semilogy(zetas, ev_post, 's-', color='#1f77b4', label='POST-turn')
    ax[2].axhline(0.5 / TC.LAM, color='0.5', ls=':', label='unobservable ceiling')
    ax[2].set_xlabel('TRUE wind direction ζ [rad]')
    ax[2].set_ylabel(r'marginal min-EV(ζ)  [rad$^2$]')
    ax[2].set_title('Observability from (φ, ψ, γ)')
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, which='both')

    fig.suptitle('ζ-ANN on observable vs unobservable trajectory sections — '
                 'sweep the TRUE wind direction on ONE fixed 90° turn flight',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, 'ann_tracks_true_zeta.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print('→', p)

    # per-flight traces for a few ζ: CLEAN (line) vs HIGH NOISE (scatter)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    for zt, (tt, zh, zhn, z0) in sorted(per_flight.items()):
        ln, = ax.plot(tt, zh, '-', lw=2.0, label=f'ζ̂ clean  (true ζ={z0:+.2f})')
        c = ln.get_color()
        ax.scatter(tt, wrap(zhn - z0) + z0, s=12, color=c, alpha=0.4,
                   edgecolors='none', label=f'ζ̂ σ={HIGH_NOISE}rad')
        ax.axhline(z0, color=c, ls='--', lw=1)
    ax.axvspan(t_turn0, t_turn1, color='0.5', alpha=0.2, label='the turn')
    ax.set_xlabel('time [s]'); ax.set_ylabel('ζ [rad]')
    ax.set_title('ANN ζ̂ per step: line=clean, dots=high noise, dashed=true ζ')
    ax.legend(fontsize=7, ncol=3, loc='upper center'); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'ann_traces_by_true_zeta.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print('→', p)


if __name__ == '__main__':
    main()
