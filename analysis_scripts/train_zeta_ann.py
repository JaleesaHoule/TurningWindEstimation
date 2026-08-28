#!/opt/anaconda3/envs/Python311/bin/python
"""
Train a wind-direction ANN for the fly-in-wind model.

ESTIMATOR CONFIG — the paper's wind-direction ANN (arXiv 2511.08766v2, Methods
p. 15), reproduced:
    * inputs = heading φ, optic-flow angle β(=ψ), airflow angle γ over ω = 4  →
      12 input neurons                                             -> OMEGA
    * three hidden layers of 64 ReLU, one linear output for ζ      -> LAYERS
    * a Gaussian-noise input layer (µ=0, σ=0.01 rad), train-only  -> NOISE_STD
    * the circular loss (sinζ−sinζ̌)² + (cosζ−cosζ̌)²             -> build_model
    * ω=4 PRIOR steps (causal/backward window)                    -> windows()
    * observability-curated to the 40–70% band (Fig. 3e)         -> OBS_PCT_LO/HI
    * 80–20 split, 10,000 epochs, batch 4096                     -> TEST_SIZE etc.

TRAINING TRAJECTORIES — generated the paper's way (Fig. 3b): For each flight a set-point is sampled and the
fly's MPC recovers the control inputs that track it through the real dynamics. The set-points mix a rich motif
repertoire so it is not just smooth arcs:
    heading   a SMOOTH turn-rate (sinusoids + a constant bias → gentle curves,
              loops, spirals) PLUS 0..SACC_MAX SHARP SACCADES — brief Gaussian
              rate pulses of varying sign, angle and angular velocity
    speed     a linear ACCEL / DECEL ramp (ratio ≷ 1) plus a smooth wobble
    sideslip  a commanded v_perp (crabbing), zero for some flights
Wind (w, ζ) is constant per flight; ζ spans the full circle.

Naming: paper ψ→φ, paper β(optic flow)→ψ, paper γ(airflow)→γ (fly_h order).

COST: one MPC solve per trajectory (~0.5–1 s), so N_TRAJ is smaller than the
paper's 40 k — each flight is long (1.5 s = 151 steps) and yields ~147 windows,
so a few thousand feasible flights is ample training data.

Run:  /opt/anaconda3/envs/Python311/bin/python train_zeta_ann.py
      (preview only: N_TRAJ=24 python plot_zeta_ann_training_examples.py)
Out:  zeta_ann_causal/network.h5  (+ a training-curve figure)
"""

import os
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twe.paths import output_dir, CAUSAL_NET
OUT = output_dir('zeta_ann_training')            # training-curve figure(s)

# ── trajectory + estimator geometry ─────────────────────────────────────────
DT = 0.01
N_STEPS = int(os.environ.get('N_STEPS', 21))       # (paper) 0.2 s / 21 points
OMEGA = 4                       # (paper) ANN window; 3 signals × 4 = 12 inputs
N_TRAJ = int(os.environ.get('N_TRAJ', 3000))       # MPC rollouts (one solve each)
N_JOBS = int(os.environ.get('N_JOBS', 1))          # parallel MPC workers (1 = serial)
SEED = 0

# ── ANN architecture + training (the paper's) ───────────────────────────────
LAYERS = (64, 64, 64)
NOISE_STD = 0.01
BATCH = 4096
EPOCHS = int(os.environ.get('EPOCHS', 10_000))
TEST_SIZE = 0.2

# ── OBSERVABILITY CURATION (paper Fig. 3e) — see window_observability ────────
OBS_CURATE = os.environ.get('OBS_CURATE', '1') != '0'
OBS_PCT_LO = float(os.environ.get('OBS_PCT_LO', 40.0))
OBS_PCT_HI = float(os.environ.get('OBS_PCT_HI', 70.0))

# ζ-observability is computed the SAME way the sweeps and the paper's Fig-3
# colour do: the sliding-window min error variance F_ζ⁻¹ from the stochastic
# Gramian, driven by ONLY THE THREE ANGULAR SENSORS (φ, ψ, γ) — the exact inputs
# the ANN sees — with the FULL 18-state FIM inverted and ζ's diagonal read off
# (the marginal bound, all other states unknown too). This is the real thing, not
# a ∂γ/∂ζ proxy.
ANG_SENSORS = ('phi', 'psi', 'gamma')
Q_ZETA = 1e-3                   # per-step process noise on ζ [rad²]
LAM = 1e-6                      # Tikhonov floor (1/λ = 1e6 = the unobservable cap)
R_MEAS = NOISE_STD ** 2         # measurement variance = the net's training σ²

# ── set-point sampling (fly-calibrated) ─────────────────────────────────────
# ONE turn per 0.2 s flight, with turn AMPLITUDE and turn RATE sampled
# INDEPENDENTLY — the rate/amplitude decoupling this project is built on. The
# turn is a constant-rate heading ramp of duration |Δφ|/ω placed within the
# flight (straight before and after); so the set spans slow-shallow → fast-sharp
# and everything between, plus near-straight when Δφ≈0.
# capped at the REAL fly saccade p95 (153°); bigger Δφ only makes unrealistic
# loops. Real saccades: amp median 66° (p25-p75 22-118°), peak rate median 22,
# fast mode 30-38 rad/s, duration 0.06-0.12 s — so a fast turn here is a brief
# CRISP dogleg (straight, ~40-70 ms flick, straight), not a spiral.
R_AMP    = (0.0, 2.7)           # turn amplitude |Δφ| [rad]  (0 .. ~155°, real p95)
R_RATE   = (2.0, 42.0)          # turn rate ω [rad/s]  (gentle arc .. sharp saccade)
R_V0     = (0.15, 0.55)         # cruise speed [m/s] (floored to avoid degenerate
                                # low-speed spin-in-place; fly cruises ~0.29)
R_V_RATIO = (0.35, 2.2)         # final/initial speed — clear ACCEL & DECEL ramps
V_FLOOR  = 0.05                 # keep forward flight forward [m/s]
R_W      = (0.0, 1.2)           # wind magnitude [m/s]  (sweep uses 0.4)
N_COMP   = 2                    # sinusoidal components in a smooth speed/slip profile
F_RANGE  = (0.5, 4.0)           # their frequencies [Hz] — visible over 0.2 s
SLIP_MAX = 0.18                 # max commanded sideslip |v_perp| [m/s] (crabbing)


def _smooth(rng, n, nt, amp, bias=1.0):
    """ n smooth continuous profiles over nt steps: a mean-zero sum of N_COMP
    sinusoids plus a constant bias (bias>0 ⇒ a sustained drift, so headings can
    spiral), scaled per row by amp. """
    tt = np.linspace(0.0, 1.0, nt)
    out = np.zeros((n, nt))
    for _ in range(N_COMP):
        f = rng.uniform(*F_RANGE, (n, 1))
        ph = rng.uniform(-np.pi, np.pi, (n, 1))
        out += np.sin(2 * np.pi * f * tt + ph)
    out = out / N_COMP + bias * rng.uniform(-1.0, 1.0, (n, 1))
    return out * amp


def _saccades(rng, n, t):
    """ SHARP turn-rate pulses to superimpose on the smooth turning: 0..SACC_MAX
    per flight, each a brief Gaussian spike of random SIGN, peak rate (SACC_RATE)
    and width (SACC_WIDTH), so the heading takes sharp steps of varying angle and
    angular velocity — the fly's saccades, not just smooth arcs. """
    T = t[-1]
    out = np.zeros((n, len(t)))
    n_sacc = rng.integers(0, SACC_MAX + 1, n)
    for s in range(SACC_MAX):
        active = (s < n_sacc)[:, None]
        sign = rng.choice([-1.0, 1.0], (n, 1))
        A = sign * rng.uniform(*SACC_RATE, (n, 1))
        t0 = rng.uniform(0.1 * T, 0.9 * T, (n, 1))
        wd = rng.uniform(*SACC_WIDTH, (n, 1))
        out += np.where(active, A * np.exp(-0.5 * ((t - t0) / wd) ** 2), 0.0)
    return out


def generate(n_traj=N_TRAJ, seed=SEED, return_states=False, engine=None,
             progress=None):
    """ DYNAMICALLY FEASIBLE trajectories via the fly's MPC. Samples a smooth
    (heading, speed) set-point per flight, has the MPC track it through the real
    dynamics, and reads the measurements (φ, ψ, γ) off the ACHIEVED state. Returns
    (M, zeta) with M (n_kept, N_STEPS, 3); with return_states=True also K.
    Infeasible/non-finite rollouts are dropped, so n_kept ≤ n_traj. """
    from twe.observability import ObservabilityEngine, SYSTEMS
    spec = SYSTEMS['fly']()
    if engine is None:
        engine = ObservabilityEngine(spec)
    si = {nm: spec.state_names.index(nm)
          for nm in ('phi', 'v_para', 'v_perp', 'w', 'zeta')}
    midx = [spec.measurement_names.index(m) for m in ('phi', 'psi', 'gamma')]
    rng = np.random.default_rng(seed)
    t = np.arange(N_STEPS) * DT

    # ONE turn per flight: amplitude Δφ and rate ω sampled INDEPENDENTLY, so the
    # set spans slow-shallow → fast-sharp turns (and near-straight when Δφ≈0). The
    # turn is a constant-rate heading ramp of duration |Δφ|/ω, placed within the
    # 0.2 s with straight flight before and after.
    T = t[-1]
    dphi = rng.choice([-1.0, 1.0], (n_traj, 1)) * rng.uniform(*R_AMP, (n_traj, 1))
    omega = rng.uniform(*R_RATE, (n_traj, 1))
    dur = np.minimum(np.abs(dphi) / np.maximum(omega, 1e-6), 0.9 * T)
    t0 = rng.uniform(0, 1, (n_traj, 1)) * (T - dur)
    frac = np.clip((t[None, :] - t0) / np.maximum(dur, 1e-6), 0.0, 1.0)
    phi0 = rng.uniform(-np.pi, np.pi, (n_traj, 1))
    phi_sp = phi0 + dphi * frac
    omega_sp = np.where((frac > 0) & (frac < 1),
                        dphi / np.maximum(dur, 1e-6), 0.0)
    # speed: a linear ACCEL/DECEL ramp (ratio ≷ 1) plus a smooth wobble
    v0 = rng.uniform(*R_V0, (n_traj, 1))
    ramp = 1.0 + (rng.uniform(*R_V_RATIO, (n_traj, 1)) - 1.0) * (t / t[-1])
    v_sp = np.clip(v0 * ramp * (1.0 + _smooth(rng, n_traj, N_STEPS,
                   rng.uniform(0, 0.25, (n_traj, 1)), bias=0.0)), V_FLOOR, None)
    # commanded SIDESLIP (crabbing): a smooth v_perp, zero for some flights
    slip_amp = SLIP_MAX * rng.uniform(0, 1, (n_traj, 1)) ** 1.5
    v_perp_sp = np.clip(_smooth(rng, n_traj, N_STEPS, slip_amp, bias=0.6),
                        -3 * SLIP_MAX, 3 * SLIP_MAX)
    w0 = rng.uniform(*R_W, (n_traj, 1))
    z0 = rng.uniform(-np.pi, np.pi, (n_traj, 1))

    ns = len(spec.state_names)
    n_win = N_STEPS - OMEGA + 1
    q_diag = dict(spec.q_tiny); q_diag['zeta'] = Q_ZETA
    r_diag = {m: R_MEAS for m in spec.measurement_names}
    M = np.full((n_traj, N_STEPS, 3), np.nan)
    Xs = np.full((n_traj, N_STEPS, ns), np.nan)
    EVw = np.full((n_traj, n_win), np.nan)             # per-window min-EV(ζ)
    ok = np.zeros(n_traj, bool)
    for i in range(n_traj):
        sp = spec.make_setpoint(v_sp[i], phi_sp[i], float(w0[i, 0]),
                                float(z0[i, 0]))
        sp['v_perp'] = v_perp_sp[i]                    # commanded sideslip
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, X, U = engine.simulate_mpc(sp)     # sets engine's trajectory
                if X.shape[0] != N_STEPS or not np.all(np.isfinite(X)):
                    raise ValueError('infeasible rollout')
                # REAL F_ζ⁻¹ from the 3 angular sensors, on THIS trajectory
                ev = engine.linearized_ev(OMEGA, q_diag, r_diag, LAM,
                                          sensors=ANG_SENSORS, states=('zeta',))
        except Exception:
            continue
        Y = np.array([np.asarray(spec.h(X[k], U[k]), float)
                      for k in range(N_STEPS)])
        if not np.all(np.isfinite(Y[:, midx])):
            continue
        M[i], Xs[i], ok[i] = Y[:, midx], X, True
        EVw[i] = ev['zeta'].values
        if progress and (i + 1) % progress == 0:
            print('  MPC %d/%d (%d feasible)' % (i + 1, n_traj, ok.sum()),
                  flush=True)

    M, Xs, EVw = M[ok], Xs[ok], EVw[ok]
    phi = Xs[:, :, si['phi']]
    v_para, v_perp = Xs[:, :, si['v_para']], Xs[:, :, si['v_perp']]
    w, zeta = Xs[:, :, si['w']], Xs[:, :, si['zeta']]
    if not return_states:
        return M, zeta
    # per-STEP min-EV for plotting: the causal window ending at step k gets
    # window index k-(ω-1); the first ω-1 steps inherit the first window's value
    minev_step = np.repeat(EVw[:, :1], N_STEPS, axis=1)
    minev_step[:, OMEGA - 1:] = EVw
    K = dict(t=t, w=w, zeta=zeta, phi=phi, v_para=v_para, v_perp=v_perp,
             omega_cmd=omega_sp[ok], turn_amp=dphi[ok], turn_rate=omega[ok],
             ev_window=EVw, minev_step=minev_step, n_infeasible=int((~ok).sum()))
    return M, zeta, K


def _gen_shard(job):
    """ One worker: generate a seed-sharded chunk of flights. Reuses generate()
    verbatim (its own engine), so it cannot drift from the serial path. """
    _i, n, seed = job
    return generate(n_traj=n, seed=seed, return_states=True)


def generate_parallel(n_traj=N_TRAJ, seed=SEED, n_jobs=None):
    """ Same output as generate(return_states=True) but the independent MPC solves
    are split across processes. Each worker samples a DIFFERENT seed shard of the
    SAME distributions, so the union spans the full motif set; results are merged
    row-for-row. (MPC rollouts are not bit-reproducible anyway, so serial vs
    parallel differ by solver noise, not by design — see the memory note.) """
    import multiprocessing as mp
    n_jobs = n_jobs or N_JOBS
    n_jobs = max(1, min(n_jobs, mp.cpu_count() - 1))
    if n_jobs == 1:
        return generate(n_traj=n_traj, seed=seed, return_states=True, progress=250)
    # keep each worker single-threaded so the solves run truly in parallel and do
    # not oversubscribe the cores (children inherit this env under 'spawn')
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'
    base, rem = divmod(n_traj, n_jobs)
    counts = [base + (1 if i < rem else 0) for i in range(n_jobs)]
    jobs = [(i, counts[i], seed + 1 + i) for i in range(n_jobs) if counts[i] > 0]
    print('  parallel MPC: %d workers, chunks %s (%d flights)'
          % (len(jobs), counts, n_traj), flush=True)
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        res = pool.map(_gen_shard, jobs)
    Ms, zs, Ks = zip(*res)
    M = np.concatenate(Ms, 0)
    zeta = np.concatenate(zs, 0)
    cat = ('w', 'zeta', 'phi', 'v_para', 'v_perp', 'omega_cmd', 'turn_amp',
           'turn_rate', 'ev_window', 'minev_step')
    K = {k: np.concatenate([kk[k] for kk in Ks], 0) for k in cat}
    K['t'] = Ks[0]['t']
    K['n_infeasible'] = int(sum(kk['n_infeasible'] for kk in Ks))
    return M, zeta, K


def windows(M, zeta, omega=OMEGA):
    """ BACKWARD (causal) windows: per signal, columns x_+0..x_-(omega-1) (most
    recent FIRST), label at the window's LAST step. First omega-1 steps of each
    trajectory have no history and are dropped. Returns EVERY window; the
    observability curation (Fig. 3e) is applied afterward in main(). """
    n_traj, n_k, n_sig = M.shape
    keep = n_k - omega + 1
    cols = [M[:, omega - 1 - j: omega - 1 - j + keep, s]
            for s in range(n_sig) for j in range(omega)]
    X = np.stack(cols, axis=-1).reshape(n_traj * keep, n_sig * omega)
    y = zeta[:, omega - 1:].reshape(-1)
    return X.astype('float32'), y.astype('float32')


def dgamma_dzeta(K):
    """ ∂γ/∂ζ per step (n_traj, N_STEPS) — the sensitivity of the ONLY
    ζ-dependent measurement (airflow angle γ) to the wind direction, i.e. the
    window's ζ-observability signal. γ = atan2(a_perp, a_para),
    a_para = v_para − w cos(φ−ζ), a_perp = v_perp + w sin(φ−ζ). """
    phi, vp, vq = K['phi'], K['v_para'], K['v_perp']
    w, zeta = K['w'], K['zeta']
    bhw = phi - zeta
    apar = vp - w * np.cos(bhw)
    aperp = vq + w * np.sin(bhw)
    denom = np.maximum(apar ** 2 + aperp ** 2, 1e-12)
    return -w * (apar * np.cos(bhw) - aperp * np.sin(bhw)) / denom


def window_observability(K):
    """ Per-window ζ-observability aligned ROW-FOR-ROW with windows(): the Fisher
    information 1/F_ζ⁻¹ from the REAL 3-angular-sensor sliding Gramian stored in
    K['ev_window'] (min error variance of ζ), flattened in windows()'s order.
    Larger = more observable (small min-EV). """
    return (1.0 / np.maximum(K['ev_window'], 1e-30)).reshape(-1)


def build_model(n_in):
    """ 12 → GaussianNoise(σ) → [64 ReLU]×3 → 1 linear, the paper's circular
    loss, Adam at its Keras default lr = 1e-3. """
    import keras
    from keras import layers
    m = keras.Sequential([keras.Input(shape=(n_in,)),
                          layers.GaussianNoise(NOISE_STD)]
                         + [layers.Dense(u, activation='relu') for u in LAYERS]
                         + [layers.Dense(1, activation='linear')])

    def circular_loss(y_true, y_pred):
        import keras.ops as K
        yt, yp = K.reshape(y_true, (-1,)), K.reshape(y_pred, (-1,))
        return K.mean(K.square(K.sin(yt) - K.sin(yp))
                      + K.square(K.cos(yt) - K.cos(yp)))

    m.compile(optimizer='adam', loss=circular_loss)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    print('generating %d MPC rollouts (%d steps = %.2gs, dt=%g, ω=%d)...'
          % (N_TRAJ, N_STEPS, N_STEPS * DT, DT, OMEGA))
    M, zeta, K = (generate_parallel() if N_JOBS > 1
                  else generate(return_states=True, progress=250))
    X, y = windows(M, zeta)
    print('  feasible: %d of %d flights (%d dropped)'
          % (M.shape[0], N_TRAJ, K['n_infeasible']))
    amp = np.degrees(np.abs(K['turn_amp'].ravel()))
    rate = K['turn_rate'].ravel()
    print('  turn amplitude |Δφ|: %.0f..%.0f° (median %.0f°)'
          % (amp.min(), amp.max(), np.median(amp)))
    print('  turn rate ω: %.0f..%.0f rad/s (median %.0f)'
          % (rate.min(), rate.max(), np.median(rate)))
    print('  v_para %.2f..%.2f m/s;  w %.2f..%.2f m/s'
          % (K['v_para'].min(), K['v_para'].max(), K['w'].min(), K['w'].max()))
    print('  measurement tensor %s -> %d windows x %d inputs'
          % (M.shape, X.shape[0], X.shape[1]))

    if OBS_CURATE and (OBS_PCT_LO > 0 or OBS_PCT_HI < 100):
        obs = window_observability(K)                  # real Fisher info 1/F_ζ⁻¹
        lo, hi = np.nanpercentile(obs, [OBS_PCT_LO, OBS_PCT_HI])
        mask = np.isfinite(obs) & (obs >= lo) & (obs <= hi)
        print('  curation: keep ζ-observability %.0f–%.0f%% (Fisher 1/F_ζ⁻¹ '
              '%.3g…%.3g): %d of %d windows (%.0f%%)' % (OBS_PCT_LO, OBS_PCT_HI,
              lo, hi, mask.sum(), len(X), 100 * mask.mean()))
        X, y = X[mask], y[mask]
    else:
        print('  curation: OFF — all windows')

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_test = int(TEST_SIZE * len(X))
    te, tr = idx[:n_test], idx[n_test:]
    print('  train %d / test %d  (%.0f-%.0f split)'
          % (len(tr), len(te), 100 * (1 - TEST_SIZE), 100 * TEST_SIZE))

    model = build_model(X.shape[1])
    print('  training %d epochs, batch %d ...' % (EPOCHS, BATCH))
    hist = model.fit(X[tr], y[tr], validation_data=(X[te], y[te]),
                     epochs=EPOCHS, batch_size=BATCH, verbose=2)

    p = np.asarray(model.predict(X[te], batch_size=8192, verbose=0)).ravel()
    err = np.arctan2(np.sin(p - y[te]), np.cos(p - y[te]))
    print('\nheld-out circular error: MAE %.4f rad (%.2f°), RMSE %.4f rad (%.2f°)'
          % (np.abs(err).mean(), np.degrees(np.abs(err).mean()),
             np.sqrt((err ** 2).mean()), np.degrees(np.sqrt((err ** 2).mean()))))

    path = CAUSAL_NET                              # ANNmodels/zeta_ann_causal/network.h5
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print('→', path)

    fig, AX = plt.subplots(1, 3, figsize=(14, 4))
    AX[0].semilogy(hist.history['loss'], label='train')
    AX[0].semilogy(hist.history['val_loss'], label='test')
    AX[0].set_xlabel('epoch'); AX[0].set_ylabel('circular loss')
    AX[0].legend(); AX[0].set_title('training', fontsize=10)
    AX[1].hist(np.degrees(err), bins=120, color='tab:blue')
    AX[1].set_xlabel('circular error [deg]'); AX[1].set_ylabel('count')
    AX[1].set_title('held-out error', fontsize=10)
    s = rng.choice(len(te), size=min(4000, len(te)), replace=False)
    AX[2].scatter(np.degrees(y[te][s]),
                  np.degrees(np.arctan2(np.sin(p[s]), np.cos(p[s]))),
                  s=2, alpha=.3)
    AX[2].plot([-180, 180], [-180, 180], 'k--', lw=1)
    AX[2].set_xlabel('true ζ [deg]'); AX[2].set_ylabel('predicted ζ [deg]')
    AX[2].set_title('held-out predictions', fontsize=10)
    for a in AX:
        a.grid(alpha=.3)
    fig.tight_layout()
    fp = os.path.join(OUT, 'training.png')
    fig.savefig(fp, dpi=130, bbox_inches='tight')
    print('→', fp)


if __name__ == '__main__':
    main()
