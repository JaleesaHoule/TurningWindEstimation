#!/opt/anaconda3/envs/Python311/bin/python
"""
Wind-direction-independent turn-angle analysis for ζ (wind-direction) observability.

The fly's ζ-dependent measurements (apparent-airflow angle γ, magnitude a)
depend only on the heading RELATIVE to the wind, β = φ − ζ (β = 0 downwind,
β = ±π upwind), so the observability of ζ for a maneuver is invariant to a
joint rotation of φ and ζ. We therefore fix ζ = 0 (WLOG) and sweep the two
RELATIVE angles:

  * β0  = initial heading relative to the wind (= φ0 here, since ζ = 0)
  * Δφ  = signed turn angle

For every (β0, Δφ) we fly straight → turn(Δφ) → straight (MPC, maintains
course) and record the best (minimum-over-trajectory) min error variance of ζ,
for the pybounds empirical pipeline (Q = 0) and for the linearized Eq.(33)
Gramian at each Q. No observability threshold — the output is the raw min-EV.

Outputs (to Qanalysis/turn_angle_analysis/):
  * overview_all_Q.png     — grid of β0×Δφ heatmaps, one per Q (shared scale)
  * heatmap_Q<...>.png     — one full-size β0×Δφ heatmap per Q value
  * turn_angle_analysis.npz — raw min-EV array + axes

Run with the Python311 env:
    /opt/anaconda3/envs/Python311/bin/python turn_angle_analysis.py
"""

import os
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twe.observability import ObservabilityEngine, SYSTEMS, build_setpoints
from twe.paths import output_dir


# ─────────────────────────────── configuration ──────────────────────────────
SYSTEM = 'fly'
ZETA0 = 0.0                                 # wind direction fixed WLOG
BETA0_DEG = list(range(-180, 180, 30))      # initial heading rel. to wind (rows)
# signed turn angle. (The map is symmetric under (β0,Δφ)→(−β0,−Δφ) — a right
# turn from β0 equals a left turn from −β0 — but we keep the full ±180 range.)
DPHI_DEG = list(range(-180, 181, 30))       # turn angle −180..180° (cols)
# Values ARE the displayed Q_d (dt=0.01). The last four
# (200,300,500,1000) give displayed Q = 2, 3, 5, 10.
# PER-STEP Q_d [rad²] — the Q of the discrete Eq.30/33 recursion, passed to the
# engine verbatim. (Was Q_ZETA_PSD, a continuous PSD that the column labels then
# multiplied by dt; the values below are those old labels, so every published
# column keeps its number and only the variable's meaning changed.)
Q_ZETA_D = np.array([1e-5, 1e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3,
                     5e-3, 1e-2, 1e-1, 1.0, 2.0, 3.0, 5.0, 10.0])
LAM = 1e-6
R_SENSOR = 0.1                              # per-sensor noise covariance (matches
#                                             fly_wind_example.ipynb cell 33)
WINDOW = 4                                  # matches fly_wind_example
STRAIGHT_DUR = 0.18
TURN_DUR = 0.04
EV_VMIN, EV_VMAX = 1e-4, 1e6                # fixed color range for all maps
CMAP = 'inferno_r'                          # low EV (observable) = light
OUTDIR = output_dir('turn_angle_analysis')
NORM = mpl.colors.LogNorm(EV_VMIN, EV_VMAX)


# ─────────────────────────────── sweep ──────────────────────────────────────
def trajectory_min_ev(engine, spec, beta0_rad, dphi_rad, w, eps, r_diag,
                      sensors, states, q_floor):
    """ Fly straight → turn(Δφ) → straight at initial heading β0 in wind ζ0,
    and return best (min-over-trajectory) min-EV(ζ) for [pybounds, *Q]. """
    segs = [dict(motif='straight', duration=STRAIGHT_DUR),
            dict(motif='turn', duration=TURN_DUR, angle=dphi_rad),
            dict(motif='straight', duration=STRAIGHT_DUR)]
    speed_sp, heading_sp = build_setpoints(segs, spec.dt, v0=spec.v0_default,
                                           heading0=beta0_rad)
    sp = spec.make_setpoint(speed_sp, heading_sp, spec.wind_default, ZETA0)
    out = np.full(1 + len(Q_ZETA_D), np.nan)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.simulate_mpc(sp)
        ev_pb, _ = engine.empirical_ev(w, eps, r_diag, LAM,
                                       sensors=sensors, states=states)
        out[0] = np.nanmin(ev_pb['zeta'].values)
        for k, q in enumerate(Q_ZETA_D):
            q_diag = dict(q_floor)
            q_diag['zeta'] = float(q)
            try:
                ev = engine.linearized_ev(w, q_diag, r_diag, LAM,
                                          sensors=sensors, states=states)
                out[1 + k] = np.nanmin(ev['zeta'].values)
            except np.linalg.LinAlgError:
                out[1 + k] = np.nan
    return out


# ─────────────────────────────── plotting ───────────────────────────────────
def _wrap_beta(res):
    """ β₀ is periodic: β₀=+180° is the identical heading as β₀=−180° (straight
    upwind). The sweep samples −180 but not +180, so append a +180 row that
    mirrors the −180 row → a full, symmetric −180…+180° map. """
    beta = list(BETA0_DEG) + [180]
    row = res[BETA0_DEG.index(-180)][None]     # (1, nd, nQ)
    return np.concatenate([res, row], axis=0), beta


def _extent(beta=None):
    beta = list(BETA0_DEG) if beta is None else beta
    ds = DPHI_DEG[1] - DPHI_DEG[0]
    bs = beta[1] - beta[0]
    return [DPHI_DEG[0] - ds / 2, DPHI_DEG[-1] + ds / 2,
            beta[0] - bs / 2, beta[-1] + bs / 2]


def _decorate(ax):
    ax.set_xlabel('turn angle  Δφ  (deg)')
    ax.set_ylabel('initial heading rel. wind  β₀ = φ₀−ζ  (deg)')
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))


def qlabel(k, dt):
    return 'pybounds (Q=0)' if k == 0 else f'Q = {Q_ZETA_D[k - 1] * dt:g}'


def _cellfmt(v):
    """ Compact 1-sig-fig scientific label, e.g. 2.7e-4 → '3e-4', 1e6 → '1e6'. """
    if not np.isfinite(v):
        return ''
    m, e = f'{v:.0e}'.split('e')
    return f'{m}e{int(e)}'


def _annotate_cells(ax, mat, beta, fontsize=5.2):
    """ Write each cell's min-EV value, colored for contrast with the cell. """
    cmap = mpl.colormaps[CMAP]
    for i, b in enumerate(beta):
        for j, d in enumerate(DPHI_DEG):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            cr, cg, cb, _ = cmap(NORM(v))
            tc = 'white' if (0.299 * cr + 0.587 * cg + 0.114 * cb) < 0.5 else 'black'
            ax.text(d, b, _cellfmt(v), ha='center', va='center',
                    fontsize=fontsize, color=tc)


def draw_motif(gax, dphi_deg):
    """ Small canonical straight → turn(Δφ) → straight motif (start heading 0,
    same step proportions as the real trajectories); green start dot, arrowhead
    at the trajectory end pointing along the final heading. """
    n1 = max(int(round(STRAIGHT_DUR * 100)), 2)
    nt = max(int(round(TURN_DUR * 100)), 1)
    a = np.radians(dphi_deg)
    hd = np.concatenate([np.zeros(n1), np.linspace(a / nt, a, nt),
                         np.full(n1, a)])
    x = np.concatenate([[0.0], np.cumsum(np.cos(hd))])
    y = np.concatenate([[0.0], np.cumsum(np.sin(hd))])
    gax.plot(x, y, '-', color='0.12', lw=1.6, solid_capstyle='butt', zorder=2)
    gax.plot(x[0], y[0], 'o', color='green', ms=3, zorder=3)
    # filled arrowhead exactly at the trajectory end, along the final heading,
    # sized proportionally to the motif so it never overshoots or dominates
    span = max(np.ptp(x), np.ptp(y), 1.0)
    hl, hw = 0.16 * span, 0.11 * span
    hx, hy = np.cos(hd[-1]), np.sin(hd[-1])
    gax.add_patch(mpatches.FancyArrow(
        x[-1] - hx * hl, y[-1] - hy * hl, hx * hl, hy * hl,
        length_includes_head=True, width=0.0, head_width=hw, head_length=hl,
        color='0.12', zorder=4, clip_on=False))
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(np.ptp(x), np.ptp(y), 4.0) / 2 + 0.15 * span + 2.0
    gax.set_xlim(cx - half, cx + half)
    gax.set_ylim(cy - half, cy + half)
    gax.set_aspect('equal'); gax.axis('off')


def draw_heading_wind_glyph(gax, beta0_deg):
    """ Dark arrow = animal's initial heading β0; blue arrow = wind (ζ = 0);
    a small gray arc between them annotates the β0 angle. Arrows are single
    FancyArrow patches (head exactly at the tip). """
    gax.set_xlim(-1.18, 1.18); gax.set_ylim(-1.18, 1.18)
    gax.set_aspect('equal'); gax.axis('off')
    # arc from the wind direction (0°) to the heading (β0) — spans |β0|
    if abs(beta0_deg) > 1:
        t1, t2 = (0.0, beta0_deg) if beta0_deg >= 0 else (beta0_deg, 0.0)
        gax.add_patch(mpatches.Arc((0, 0), 0.82, 0.82, angle=0.0,
                      theta1=t1, theta2=t2, color='0.45', lw=1.1, zorder=2))
    for ang, color in ((0.0, 'steelblue'), (np.radians(beta0_deg), '0.12')):
        ex, ey = np.cos(ang), np.sin(ang)
        gax.add_patch(mpatches.FancyArrow(
            0.0, 0.0, ex, ey, length_includes_head=True, width=0.05,
            head_width=0.26, head_length=0.32, color=color, zorder=3,
            clip_on=False))


def save_individual_heatmaps(res, dt):
    resw, beta = _wrap_beta(res)
    nb, nd = len(beta), len(DPHI_DEG)
    for k in range(resw.shape[2]):
        fig = plt.figure(figsize=(10.0, 9.2))
        outer = fig.add_gridspec(2, 3, width_ratios=[1.3, 8.5, 0.35],
                                 height_ratios=[8.5, 1.0], wspace=0.04,
                                 hspace=0.06)
        ax = fig.add_subplot(outer[0, 1])
        im = ax.imshow(resw[:, :, k], origin='lower', aspect='auto', cmap=CMAP,
                       norm=NORM, extent=_extent(beta))
        _annotate_cells(ax, resw[:, :, k], beta)
        ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
        ax.tick_params(labelsize=7)
        ax.set_title(f'ζ min error variance vs (β₀, Δφ) — {qlabel(k, dt)}\n'
                     '(ζ fixed = 0; lower/lighter = more observable)', fontsize=10)
        cax = fig.add_subplot(outer[0, 2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('min error variance of ζ  (best over trajectory)')
        # y-axis strip: heading (dark) vs wind ζ=0 (blue) per β0 row
        lgs = outer[0, 0].subgridspec(nb, 1, hspace=0.1)
        for i, bb in enumerate(beta):
            draw_heading_wind_glyph(fig.add_subplot(lgs[nb - 1 - i]), bb)
        # x-axis strip: turn motif per Δφ column
        bgs = outer[1, 1].subgridspec(1, nd, wspace=0.1)
        for j, dpv in enumerate(DPHI_DEG):
            draw_motif(fig.add_subplot(bgs[j]), dpv)
        fig.text(0.012, 0.55, 'initial heading (dark arrow) vs wind ζ=0 '
                 '(blue arrow),  β₀ (deg)', rotation=90, va='center', fontsize=9)
        fig.text(0.52, 0.012, 'turn motif  Δφ  (straight → turn → straight, '
                 '−180→180°)', ha='center', fontsize=9)
        tag = 'Q0_pybounds' if k == 0 else f'Q_{Q_ZETA_D[k-1]*dt:g}'
        fig.savefig(os.path.join(OUTDIR, f'heatmap_{tag}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)


def save_overview(res, dt):
    resw, beta = _wrap_beta(res)
    nQ = resw.shape[2]
    ncols = 4
    nrows = int(np.ceil(nQ / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows),
                            squeeze=False)
    im = None
    for k in range(nQ):
        ax = axs.ravel()[k]
        im = ax.imshow(resw[:, :, k], origin='lower', aspect='auto', cmap=CMAP,
                       norm=NORM, extent=_extent(beta))
        _annotate_cells(ax, resw[:, :, k], beta, fontsize=3.0)
        ax.set_title(qlabel(k, dt), fontsize=9)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-180, 181, 90))
        ax.tick_params(labelsize=7)
        if k % ncols == 0:
            ax.set_ylabel('β₀ (deg)', fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel('Δφ (deg)', fontsize=8)
    for ax in axs.ravel()[nQ:]:
        ax.set_visible(False)
    # axis-legend icons on the bottom-left panel (reference for all panels):
    # y-axis = heading(dark)+wind ζ=0(blue) vectors per β0; x-axis = turn motifs
    bl = axs.ravel()[(nrows - 1) * ncols]
    bl.set_xlabel(''); bl.set_ylabel('')   # drop deg labels so icons sit close
    nb, nd = len(beta), len(DPHI_DEG)
    for i, bb in enumerate(beta):
        draw_heading_wind_glyph(
            bl.inset_axes([-0.185, i / nb, 0.095, 1 / nb]), bb)
    for j, dpv in enumerate(DPHI_DEG):
        draw_motif(bl.inset_axes([j / nd, -0.175, 1 / nd, 0.115]), dpv)
    bl.text(-0.26, 0.5, 'β₀: heading (dark) vs wind ζ=0 (blue)',
            transform=bl.transAxes, rotation=90, va='center', ha='center',
            fontsize=7.5)
    bl.text(0.5, -0.27, 'Δφ turn motif', transform=bl.transAxes, ha='center',
            va='center', fontsize=7.5)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label('min error variance of ζ  (best over trajectory)')
    fig.suptitle('ζ observability vs initial heading-rel-wind β₀ and turn Δφ  '
                 '(ζ fixed = 0; wind-direction-independent)\n'
                 'lower/lighter = more observable',
                 fontsize=11)
    fig.savefig(os.path.join(OUTDIR, 'overview_all_Q.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    spec = SYSTEMS[SYSTEM]()
    if 'zeta' not in spec.state_names:
        raise SystemExit(f"system '{SYSTEM}' has no ζ state; use fly/fly7/drone")
    engine = ObservabilityEngine(spec)
    w, eps = WINDOW, spec.eps_default
    r_diag = {k: R_SENSOR for k in spec.r_default}
    sensors = tuple(spec.fim_sensors)
    states = tuple(spec.fim_states)
    if 'zeta' not in states:
        states = states + ('zeta',)
    q_floor = dict(spec.q_tiny)
    dt = spec.dt

    nb, nd = len(BETA0_DEG), len(DPHI_DEG)
    res = np.full((nb, nd, 1 + len(Q_ZETA_D)), np.nan)
    print(f'system={SYSTEM}  ζ fixed={ZETA0}  w/v={spec.wind_default/spec.v0_default:.2f}\n'
          f'β0: {nb} values {BETA0_DEG[0]}..{BETA0_DEG[-1]}°   '
          f'Δφ: {nd} values {DPHI_DEG[0]}..{DPHI_DEG[-1]}°   '
          f'→ {nb*nd} trajectories\n'
          f'Q (rad²): 0(pybounds) + {list(Q_ZETA_D*dt)}\n', flush=True)

    for i, b in enumerate(BETA0_DEG):
        print(f'β0 = {b:4d}° ...', flush=True)
        for j, dp in enumerate(DPHI_DEG):
            res[i, j] = trajectory_min_ev(engine, spec, np.radians(b),
                                          np.radians(dp), w, eps, r_diag,
                                          sensors, states, q_floor)

    np.savez(os.path.join(OUTDIR, 'turn_angle_analysis.npz'),
             min_ev=res, beta0_deg=np.array(BETA0_DEG),
             dphi_deg=np.array(DPHI_DEG),
             Q=np.concatenate([[0.0], Q_ZETA_D * dt]))
    save_individual_heatmaps(res, dt)
    save_overview(res, dt)
    print(f'\nsaved heatmaps + overview + npz under: {OUTDIR}')


if __name__ == '__main__':
    main()
