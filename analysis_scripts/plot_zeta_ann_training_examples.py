#!/opt/anaconda3/envs/Python311/bin/python
"""
Figures written to zeta_ann_causal/:
  1. training_examples.png   Fig-3b-style tile of arrow-chain flights

Run:  python plot_zeta_ann_training_examples.py
      (N_TRAJ=48 controls how many flights are generated for the tile)
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import train_zeta_ann as TZ

OUT = TZ.OUT
N_TILE = int(os.environ.get('N_TILE', 15))       # flights in the Fig-3b tile
N_TS = 4                                          # flights with full time series
EV_LO, EV_HI = 1e-3, 1e6                          # F_ζ⁻¹ colour range (paper)
ARR_SCALE = 16                                    # heading arrows per panel width


def track(K, i):
    """ Ground track (x, y) from the achieved body-frame velocity + heading. """
    phi, vp, vq = K['phi'][i], K['v_para'][i], K['v_perp'][i]
    xd = vp * np.cos(phi) - vq * np.sin(phi)
    yd = vp * np.sin(phi) + vq * np.cos(phi)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (xd[1:] + xd[:-1]) * TZ.DT)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (yd[1:] + yd[:-1]) * TZ.DT)])
    return x, y


def fig_tracks(K):
    """ Fig-3b tile: ONE PANEL PER FLIGHT (big), each drawn as one HEADING ARROW
    PER TIME STEP tiling the path, coloured by the REAL F_ζ⁻¹ (min error variance
    of ζ from the 3 angular sensors), with the wind arrow. """
    n = min(N_TILE, len(K['phi']))
    minev = K['minev_step']
    # ACHIEVED peak turn rate per flight, then pick tiles that EVENLY SPAN that
    # range (gentle → sharp saccade) so the sharpest turns are always shown, not
    # an arbitrary first-N subset.
    peak = np.array([np.abs(np.gradient(np.unwrap(p)) / TZ.DT).max()
                     for p in K['phi']])
    order = np.argsort(peak)
    sel = order[np.linspace(0, len(order) - 1, n).round().astype(int)]
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, AX = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.8 * nrow),
                           squeeze=False)
    q = None
    for j in range(nrow * ncol):
        ax = AX[j // ncol][j % ncol]
        if j >= n:
            ax.axis('off'); continue
        i = int(sel[j])
        x, y = track(K, i)
        phi = K['phi'][i]
        # faint path line for SHAPE, then one heading arrow per time step at a
        # FIXED visual size (scale_units='width' → length = 1/ARR_SCALE of the
        # panel width, identical in every subplot regardless of speed/extent);
        # colour = real F_ζ⁻¹
        ax.plot(x, y, '-', color='0.78', lw=1.0, zorder=1)
        q = ax.quiver(x, y, np.cos(phi), np.sin(phi),
                      np.clip(minev[i], EV_LO, EV_HI), cmap='inferno_r',
                      norm=LogNorm(EV_LO, EV_HI), angles='xy',
                      scale_units='width', scale=ARR_SCALE, width=0.011,
                      headwidth=3.5, headlength=4.5, headaxislength=4,
                      pivot='tail', zorder=3)
        ax.plot(x[0], y[0], 'o', color='#1a9850', ms=7, mec='k', zorder=5)
        zt = float(K['zeta'][i][-1])
        ax.annotate('', xy=(0.86 + 0.12 * np.cos(zt), 0.87 + 0.12 * np.sin(zt)),
                    xytext=(0.86, 0.87), xycoords='axes fraction',
                    textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=2.2))
        ax.text(0.86, 0.70, 'wind', transform=ax.transAxes, ha='center',
                va='top', fontsize=7, color='tab:blue')
        # UNIFORM FRAMING: a SQUARE data window sized to THIS flight's own major
        # axis, so every trajectory fills the same fraction of its (square) panel
        # regardless of absolute size or bbox aspect ratio. Equal aspect at the
        # default adjustable='box' otherwise shrinks each axes box to the data
        # aspect, so high-aspect (near-straight) flights render tiny within their
        # cell — the cause of the mismatched sizes.
        cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        R = max(0.5 * max(x.max() - x.min(), y.max() - y.min()), 1e-3) * 1.30
        ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
        ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_title('ζ=%+.0f°   w=%.2f m/s\npeak %.0f rad/s   Δφ=%+.0f°'
                     % (np.degrees(zt), float(K['w'][i][0]), peak[i],
                        np.degrees(K['turn_amp'][i, 0])), fontsize=8)
    # colourbar in its OWN axes on the far right, clear of the trajectories
    fig.subplots_adjust(left=0.015, right=0.90, top=0.92, bottom=0.02,
                        wspace=0.05, hspace=0.28)
    cax = fig.add_axes([0.915, 0.30, 0.013, 0.40])
    cb = fig.colorbar(q, cax=cax, extend='both')
    cb.set_label(r'$\mathcal{F}_\zeta^{-1}$  [rad$^2$] — min error variance of ζ '
                 'from (φ, ψ, γ)  (yellow = observable)')
    fig.suptitle('ζ-ANN training trajectories — %d MPC-feasible flights (%.1f s '
                 'each)\none HEADING arrow per time step; colour = real ζ '
                 'observability from the 3 ANGULAR sensors only; blue = wind ζ'
                 % (n, TZ.N_STEPS * TZ.DT), fontsize=12)
    p = os.path.join(OUT, 'training_examples.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('→', p)



def main():
    os.makedirs(OUT, exist_ok=True)
    n = max(N_TILE, TZ.N_TRAJ) if os.environ.get('N_TRAJ') else N_TILE
    print('generating %d MPC flights for the plots...' % n)
    M, zeta, K = TZ.generate(n_traj=n, seed=TZ.SEED, return_states=True,
                             progress=50)
    print('  feasible %d/%d' % (M.shape[0], n))
    fig_tracks(K)
    fig_timeseries(M, zeta, K)
    fig_coverage(M, zeta, K)
    fig_curation(K)


if __name__ == '__main__':
    main()
