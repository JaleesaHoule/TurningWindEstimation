#!/opt/anaconda3/envs/Python311/bin/python
"""
Follow-up: Fix ζ, sweep the TRUE wind speed w across/through the training range, split each
flight into straight windows vs turn-straddling windows, report ANN |ζ̂−ζ|.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import turn_control_sweep as TC
import train_zeta_ann as TZ
from twe.observability import ObservabilityEngine, SYSTEMS
from twe.paths import output_dir
from twe.zeta_ann_estimator import CausalZetaANN

OUT = output_dir('ann_observability_probe_output')
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def mpc_at_w(engine, spec, sp, hd, w, zeta):
    import contextlib, io
    sim = engine._get_mpc_sim()
    sim.update_dict(spec.make_setpoint(sp, hd, float(w), float(zeta)), name='setpoint')
    with contextlib.redirect_stdout(io.StringIO()):
        t, x, u, _ = sim.simulate(x0=None, mpc=True, return_full_output=True)
    return t, np.vstack(list(x.values())).T, np.vstack(list(u.values())).T


def main():
    spec = SYSTEMS['fly']()
    engine = ObservabilityEngine(spec)
    ann = CausalZetaANN(spec)
    W = ann.time_steps
    DT = spec.dt

    speed, heading, k0, nt, meta = TC.vee_setpoints(DT, 90.0, 10.0, TC.V_CRUISE)
    N = len(speed)
    ZT = 0.6                                   # fixed true ζ, off the wrap & upwind axis
    turn = np.zeros(N, bool)
    turn[k0:min(k0 + nt + W, N)] = True        # windows straddling the turn
    straight = np.ones(N, bool); straight[:W - 1] = False; straight[turn] = False

    ws = np.linspace(0.0, 3.0, 25)             # training w ∈ [0, 1.2]
    e_straight, e_turn = [], []
    for w in ws:
        try:
            t, X, U = mpc_at_w(engine, spec, speed, heading, w, ZT)
            if X.shape[0] != N or not np.all(np.isfinite(X)):
                raise ValueError
        except Exception:
            e_straight.append(np.nan); e_turn.append(np.nan); continue
        Y = np.array([np.asarray(spec.h(X[k], U[k]), float) for k in range(N)])
        err = np.abs(wrap(ann.predict_series(Y) - ZT))
        e_straight.append(np.sqrt(np.nanmean(err[straight] ** 2)))
        e_turn.append(np.sqrt(np.nanmean(err[turn] ** 2)))
    e_straight, e_turn = np.array(e_straight), np.array(e_turn)

    inb = ws <= TZ.R_W[1]
    print(f'fixed true ζ = {ZT} rad; training w ∈ {TZ.R_W}')
    print(f'straight-window ANN error:  in-range mean {np.nanmean(e_straight[inb]):.3f} '
          f'/ out-of-range mean {np.nanmean(e_straight[~inb]):.3f} rad')
    print(f'turn-window ANN error:      in-range mean {np.nanmean(e_turn[inb]):.3f} '
          f'/ out-of-range mean {np.nanmean(e_turn[~inb]):.3f} rad')



if __name__ == '__main__':
    main()
