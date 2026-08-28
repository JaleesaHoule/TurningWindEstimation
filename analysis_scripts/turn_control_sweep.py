#!/opt/anaconda3/envs/Python311/bin/python
"""
System (as specified): the 18-state ``fly`` model, driven by the ANGULAR sensors
only — φ (heading), ψ (course/slip) and γ (apparent-airflow direction), i.e.
``spec.fim_sensors`` — and every number reported here is the min error variance of
the WIND DIRECTION ζ alone (the marginal bound: the full Gramian is inverted and
ζ's diagonal entry read off, so all 17 other states are unknown too).

THE CONTROL RULE, USED BY ALL FOUR FAMILIES
-----------------------------------------------
Straight leg → constant-rate turn of Δ → straight leg, commanded as a per-step
(ground speed, heading) set-point pair that the MPC inverts. Nothing else: no
deceleration ramp, no actuator caps, no endpoint iteration. Three numbers define a
trajectory — the net turn Δ, the turn rate ω, and the ground speed v:

    heading(t) = +Δ/2  →  −Δ/2   at rate ω, centred in time;   speed(t) = v

Because the heading sweep is SYMMETRIC about 0, the net displacement is exactly
along +x, so the family shares a start AND an end point by construction once the
straight legs absorb the difference:

    x_end = 2·ℓ·cos(Δ/2) + 2·R·sin(Δ/2),    R = v/ω
    ⇒ ℓ = (x_end − 2·R·sin(Δ/2)) / (2·cos(Δ/2))

Every row starts at (0, 0), and — before rotation — heads symmetric about 0 to end at
(X_END, 0), measured endpoint spread a fraction of a millimetre, every turn in the SAME
direction (TURN_SIGN, clockwise: heading falls from +Δ/2 to −Δ/2). The wind is upwind
of the turn (the fly heads INTO it at midturn), so ∂γ/∂ζ = +w/(v+w) has no pole. The
whole geometry is then RIGIDLY ROTATED by HEADING_ROT (see below) so ζ never sits on
the ±π wrap boundary: heading and ζ both gain HEADING_ROT, β = φ − ζ is unchanged, and
the endpoint moves to (X_END·cos HEADING_ROT, X_END·sin HEADING_ROT). What is NOT fixed
is the total DURATION: with the endpoint pinned in SPACE
the leg lengths, and hence T = 2ℓ/v + Δ/ω, follow from Δ, ω and v. That is expected
and is reported per row.

THE FOUR FAMILIES
-----------------
a) ``a_amplitude``   — Δ swept 30…120° at FIXED ω and v. Amplitude alone: the fly
                       turns at one rate and simply holds the turn longer.
b) ``b_turn_rate``   — ω swept 2.5…20 rad/s at FIXED Δ = 90° and v. Rate alone,
                       the amplitude/rate decoupling the study is built around.
                       (a) and (b) are the two orthogonal axes of the (Δ, ω) plane.
c) ``c_speed``       — v swept 0.145…0.580 m/s at three FIXED turn angles. The turn
                       RADIUS is held fixed per angle, so within one angle every
                       speed traces the IDENTICAL ground track and only ω = v/R and
                       T = L/v change. This is the only knob that separates "sharper
                       corner" from "faster flight". Heading upwind, ∂γ/∂ζ falls
                       smoothly with speed (+0.73 → +0.41) with no pole; a fast row is
                       observed for fewer samples, which is intrinsic.
d) ``d_wind_shift``  — exactly family (b), but the wind DIRECTION really moves: ζ
                       steps by +90° at step 25 while the fly keeps flying the same
                       commanded trajectory, so the WIND is the only thing that differs
                       from (b). A two-stage MPC rollout with a state handoff at the
                       step (see ``replan_through_shift``), not an edit of ζ into a
                       finished trajectory, which would be dynamically inconsistent.
e) ``e_invariant_``    — the same shift with the trajectory held EXACTLY invariant: the
   ``course_wind_``      fly cancels the wind's changed drag feed-forward, so no motion
   ``shift``             transient can contribute to the bound at all. See
                       ``hold_course_through_shift``. It moves (d) by only 1–3.5%, so it
                       is a control confirming (d) is clean, not a different answer.
f) ``f_late_wind_``    — (d) again, with the shift arriving at t = 0.60 s (step 60)
   ``shift``             instead of step 25. In (d) the shift lands BEFORE the turn in
                       every row but the slowest; at 0.60 s it lands DURING or just
                       AFTER it, with only 21–36 steps left to recover.

FIGURES (per family, in ``plots/``; data in ``data/``)
------------------------------------------------------
  family.png                     every trajectory of the family, arrowhead style,
                                 one arrowhead per data point, wind vector shown
                                 (two arrows + the turn arc in family d).
  traj_<row>.png                 one per trajectory: the path drawn twice, coloured
                                 by min-EV(ζ) from the stochastic OBSERVABILITY
                                 Gramian (left) and the CONSTRUCTABILITY Gramian
                                 (right), over min-EV(ζ) vs time at every Q_ζ —
                                 observability solid, constructability dotted in the
                                 matching colour.
  matrix_best_w4.png             trajectory × Q_ζ heat map, observability and
  matrix_mean_w4.png             constructability as the two subplots, on ONE fixed
                                 colour scale (1e-4…1e6 rad²) shared by every figure
                                 this script writes.
  ekf_zeta.png                   EKF and UKF estimates of ζ: path on the left coloured
  ukf_zeta.png                   by constructability min-EV(ζ), ζ̂ vs time on the right
                                 with one colour per Q_ζ. Both filters see one shared
                                 initial guess and one shared measurement record per
                                 trajectory (recorded in filter_zeta_error.csv). ζ̂ is
                                 plotted on a branch centred on ζ, since ζ = −π sits on
                                 the usual ±π cut.

Q_ζ is the per-step process covariance on the wind direction [rad²], passed to the
engine verbatim (no dt conversion). In families a–c the simulated truth holds ζ CONSTANT
and Q_ζ is the assumed volatility entering the bound; in d–f the wind genuinely moves.
The filters start ZETA_OFFSET rad wrong in a–c (there is otherwise nothing to watch,
since the truth never moves) and CORRECT in d–f (there the step itself is the signal).

    /opt/anaconda3/envs/Python311/bin/python turn_control_sweep.py
"""

import os
import io
import csv
import datetime
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twe.observability import ObservabilityEngine, SYSTEMS
from twe.plotting import heading_of, heading_wedges
from twe.paths import output_dir


# ─────────────────────────────── configuration ──────────────────────────────
SYSTEM = 'fly'                 # 18-state fly-in-wind (m, I, C_*, km1-4 as
                                # CONSTANTS). The paper's own models are n = 4
                                # (altitude) and n = 11 (drone); the 18-state
                                # 'fly' carries 9 parameters as states, which is
                                # far outside that range.
WIND = 0.4                      # [m/s]
# NO-UPWIND MODE. Set NOUPWIND=1 in the environment to rotate the wind by 90° and write
# to turn_control_sweep_noupwind_output/ instead.
#
# The heading profile runs from +Δ/2 to −Δ/2, symmetric about φ = 0 (see
# vee_setpoints), and the upwind heading — the one that puts the fly nose-on to the
# wind, β = φ − ζ = π — is φ = ζ + π. At the default ζ = −π that is φ = 0 exactly, so
# EVERY trajectory in the sweep crosses upwind at its turn midpoint, by construction.
#
# Avoiding that needs no change to the paths: rotating the wind by δ moves the upwind
# heading to δ, and the condition for a turn of amplitude Δ never to reach it is simply
# |δ| > Δ/2. δ = 90° clears the largest turn here (Δ = 135°, headings ±67.5°) by 22.5°,
# and clears the DOWNWIND heading φ = ζ by the same margin, so the trajectories stay
# strictly on one side of the along-wind axis. The paths are geometrically identical to
# the upwind run — only their orientation relative to the wind differs — so the two
# output directories are directly comparable row for row.
#
# The whole geometry is RIGIDLY ROTATED by HEADING_ROT so the wind direction never
# sits on the ±π wrap boundary (ζ = −π, where a wrapped ζ estimate flips between −π
# and +π−ε). Adding HEADING_ROT to every commanded heading AND to ζ leaves
# β = φ − ζ — and hence every angular measurement, ∂γ/∂ζ, and the ζ observability —
# UNCHANGED: it is the same physics, relabelled. With HEADING_ROT = π/2 the upwind
# run moves to ζ = −π/2 and the off-axis run (NOUPWIND) to ζ = 0, both clear of ±π.
NOUPWIND = os.environ.get('NOUPWIND') == '1'
HEADING_ROT = np.pi / 2         # [rad] rigid rotation off the ±π wrap boundary
_ZETA_BASE = -np.pi / 2 if NOUPWIND else -np.pi
ZETA = float(np.arctan2(np.sin(_ZETA_BASE + HEADING_ROT),
                        np.cos(_ZETA_BASE + HEADING_ROT)))   # [rad] wind direction
X_END = 0.20                    # [m] rotated chord length (the run is (0,0) → the
                                # point X_END away along heading HEADING_ROT)
V_CRUISE = 0.29               # [m/s] fly between-saccade median ground speed
TURN_SIGN = -1                  # same turn direction in every row
R_TERM = 1e-12                  # do-mpc rterm: the penalty on the input RATE Δu.
#   The inherited value from ``make_fly_spec``'s build_mpc (the pybounds
#   fly_wind_example default) is 1e-6, and it does not merely cap the reachable turn
#   rate — it makes tracking fidelity depend on ω, which is the variable family (b)
#   sweeps. Measured heading error of the commanded turn, Δ = 90°:
#       ω        2.5    10     20      30      40   rad/s
#       1e-6     0.8°   3.3°   6.6°   11.3°   17.4°     ← error grows with ω
#       1e-12    0.0°   0.0°   0.0°    0.1°    0.1°
#   So the fast rows were flying a rounded-off version of their own commanded turn while
#   the slow rows flew theirs exactly — an artifact correlated with the swept variable.
#   It is NOT a horizon limit (10, 30 and 60 steps give identical results) and NOT a
#   physical limit (the rotational time constant is I/C_phi = 19 ms, well inside even a
#   39 ms turn); it is purely this regularizer. Applied by overriding build_mpc on THIS
#   script's own spec instance, so the shared `fly` spec and every other script that
#   uses it are untouched.

W = 4                           # estimation window [steps] — the only one used
LAM = 1e-6
# Q_ζ = 0 is NOT in this grid, and cannot be: the stochastic observability and
# constructability Gramians (Eq. 30/33) invert Q, so Q_ζ = 0 makes Qd singular
# and the recursion is undefined. The Q = 0 limit is already covered here by the
# EMPIRICAL Gramian basis, which is the deterministic (Q = 0) one by definition.
# The Q_ζ = 0 vs 1 ESTIMATOR comparison lives in q0_estimator_comparison.py,
# where only the filters run and no Gramian is inverted.
Q_ZETA = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])   # per-step Q_ζ [rad²]
# the full sweep is COMPUTED (CSVs, matrices), but the ζ̂-vs-time line
# plots draw only these two — the near-zero floor (ζ a fixed parameter) and 1.0
# (a fast random walk) — so each filter figure contrasts the two extremes cleanly.
PLOT_Q = (Q_ZETA[0], Q_ZETA[-1])                         # = (1e-5, 1.0)
Q_REF = 1e-2                    # the Q_ζ whose min-EV colours the trajectory panels
#   (the largest volatility that is still a physically plausible wind over a run of
#   this length — see tempo_drifting_wind.py's PHYSICAL_Q_MAX)
BASES = ('observability', 'constructability')
EV_VMIN, EV_VMAX = 1e-4, 1e6    # ONE colour/axis range on every min-EV panel here
EV_CMAP = 'inferno_r'
Q_CMAP = 'copper_r'             # ordered colormap for Q_ζ on the EV figures
Q_CMAP_FILT = 'rainbow'         # kept as-is on the filter figure

# family (a): amplitude alone
A_AMPS = (30.0, 60.0, 90.0, 120.0, 135.0)
A_OMEGA = 10.0
# family (b) and (d): turn rate alone
B_AMP = 90.0
B_OMEGAS = (2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
#   The top of this range is only reachable with R_TERM below; with the inherited 1e-6
#   the commanded turn is simply not flown above ~20 rad/s. It now brackets the real
#   fly, whose measured median peak rate is ~24 rad/s at 90° and ~39 rad/s at 135°,
#   instead of stopping short of it.
# family (c): overall speed × turn angle, geometry fixed per angle
C_AMPS = (30.0, 120.0)
C_RADIUS = V_CRUISE / A_OMEGA               # 29 mm — fixed path per turn angle
C_SPEEDS = (0.145, 0.290, 0.435, 0.580)     # ⇒ ω = v/R = 5, 10, 15, 20 rad/s
# family (d): the wind direction really moves
D_SHIFT_STEP = 25               # families (d) and (e): the shift lands BEFORE the
#                                 turn in every row but the slowest
F_SHIFT_STEP = 60               # family (f): t = 0.60 s, which lands DURING or just
#                                 AFTER the turn instead — the complementary case
D_SHIFT_RAD = np.pi / 2
SHIFT_STEP = D_SHIFT_STEP       # the step in force for the family being built now

# filters
FILTERS = ('EKF', 'UKF', 'AIKF', 'ANN')
# ── the AI-KF and its ANN ───────────────────────────────────────────────────
# The ANN is the CAUSAL wind-direction net of train_zeta_ann.py (the repo's own
# ζ networks are forward-window and cannot run inside a filter). It reads the
# same three angular sensors the filters do, over a backward ω window.
#
# The motif is |u_phi|: TURNING is what makes ζ observable here, exactly as
# |u_x| (forward acceleration) is the motif for altitude in the paper. The gate
# bound is measured, not guessed -- the backward-window(20) mean of |u_phi|
# peaks at 77 on the ω=2.5 row and 664 on the ω=40 row, so an upper bound near
# the median peak (372) opens the channel during fast saccades and leaves it
# shut through slow turns.
#
# R_ann_lo is NOT the paper's 1e-3. That value asserts σ ≈ 0.03 rad, and this
# net's error in its most-observable decile is an order of magnitude larger;
# claiming 1e-3 would make the filter over-trust a noisy pseudo-measurement.
# It is set from the measured held-out variance instead (see
# eval_zeta_ann_vs_observability.py).
AIKF_MOTIF = 'u_phi'
AIKF_WINDOW = 20
AIKF_UPPER = 372.0
AIKF_R_HI = 1e12
AIKF_R_LO = 7.936127e-02                # measured, not the paper's 1e-3
SEED = 7                        # one seed, so both filters see identical data
ZETA_OFFSET = 1.0               # [rad] the filters' initial ζ error in families a–c,
#   where the truth never moves and a correctly-initialized filter would have nothing
#   to show. In d–f it is 0: there the wind step IS the signal, and a wrong prior would
#   confound "recovering from a bad guess" with "tracking a real change". Set
#   ZETA_OFFSET_SHIFT below if the shift families should carry the same prior error.
ZETA_OFFSET_SHIFT = 1.0         # [rad] the same quantity for families d–f
# Either way the value is EXACT and identical in every row: pybounds draws the initial
# estimate from default_rng(seed + 1000*trajectory_version), so each rollout otherwise
# gets a different random prior (measured spread 0.44–1.49 rad in a–c and 0.04–0.56 rad
# in d–f, the latter supposedly "correct"). σ₀ = 0.1·max(|x₀|,0.1) also means ζ = −π
# inflates the ζ prior's σ from 0.01 to 0.314 rad. row_filters cancels that draw.
# UKF tuning: reused verbatim from tempo_filter_sweep.py, which is where it was
# validated. alpha=1, kappa=0 is the only choice with a non-negative centre weight for
# this 18-state model, and wrap_states (treating phi and zeta as circular in the state
# MEAN) is the change that actually mattered there.
# UNSCENTED TRANSFORM TUNING — the paper's, not tempo_filter_sweep's.
# Cellini et al. (2511.08766) Methods: "The Unscented transform parameters were
# set to alpha = 10^-3, beta = 1, kappa = 0", which is also what their
# util/StateEstimator.m sets. tempo_filter_sweep used alpha = 1, beta = 2, and
# on this model that is not a cosmetic difference:
#
#   alpha = 1,    n = 18  ->  n + lambda = 18,      sigma spread 4.243 sigma
#   alpha = 1e-3, n = 18  ->  n + lambda = 1.8e-5,  sigma spread 0.0042 sigma
#
# At 4.24 sigma the sigma points of an ANGLE span a large arc, so wrapping their
# residuals (wrap_states) folds distant points onto each other and the covariance
# stops describing the actual spread. That -- not the wind step, and not the
# AI-KF -- is what made the UKF diverge at small Q_zeta and run 3-4x worse than
# the EKF. At alpha = 1e-3 the spread is 0.004 sigma, wrapping never triggers,
# Q_zeta = 0 runs clean, and the UKF beats the EKF (RMSE 0.022 vs 0.073).
UKF_KW = dict(alpha=1e-3, beta=1.0, kappa=0.0)

# ── the NO-PRIOR variant (ekf_zeta_noprior.png / ukf_zeta_noprior.png) ──
# A second pass of the same filters on the same data, differing in exactly ONE way:
#
#   P₀(ζ) is DIFFUSE. The default rule inflates σ₀ by the offset itself
#   ("keep P0 honest", observability_gui._estimation_problem), giving
#   σ₀(ζ) = hypot(0.1·π, 1) = 1.048 rad — a prior that says "1 rad wrong, and I
#   know it to within 1 rad". Here σ₀(ζ) = π/√3 = 1.814 rad instead, the standard
#   deviation of a UNIFORM distribution on the circle: the filter is told it knows
#   nothing whatever about the wind direction. ζ̂(0) is unchanged, so the guess is
#   the same arbitrary value in both figures; only its weight changes.
#
# CONVENTION: every pass here uses start_at='posterior' — the app's convention, and
# what this sweep is asked to match. The initial estimate IS x̂₀|₀: the k = 0
# correction is skipped, y₀ is never consumed, and X_hat[0] is the guess itself, so
# every curve starts ON the dotted 'initial guess' line. (The alternative,
# start_at='prior', makes X_hat[0] one update old and aligns the filter with the
# constructability Gramian's recursion F₀ = C₀ᵀR⁻¹C₀; set the passes' start_at back
# to 'prior' to restore that alignment.) Everything else — seed, measurement record,
# Q_ζ grid, sensors, UKF tuning — is identical across passes.
NOPRIOR_SIGMA_ZETA = float(np.pi / np.sqrt(3))    # [rad] uniform-on-the-circle σ

# ── the BAD-GUESS variants (*_zeta_bad_initial_guesses_10.png / _50.png) ──
# The converse question to the no-prior one: there the filter knew nothing about ζ and
# everything else roughly right; here the REST of the state vector is wrong, and we ask
# whether ζ — which the geometry makes almost instantly observable — survives that.
#
# WHICH STATES. Of the six channels in h() only φ is a state read out verbatim; ψ =
# atan2(v_perp, v_para) and γ = atan2(a_perp, a_para) are functions of several states
# each, and the sweep drives the filters from (φ, ψ, γ) alone. So BAD_STATES is every
# state that is neither directly measured nor constant:
#
#   perturbed  x, y, z, v_para, v_perp, phi_dot, w
#   φ          excluded — the one directly measured state, corrected at once anyway
#   ζ          excluded FROM THE SCALE ONLY. Its guess stays pinned to ZETA + offset
#              and σ₀(ζ) stays at 1.048 rad, identical to every other figure here, so a
#              changed ζ prior cannot confound what is being tested. ζ is of course
#              still 1 rad wrong.
#   m, I, C_para, C_perp, C_phi, km1–km4
#              excluded because they are CONSTANT. They are carried as states with only
#              the q_tiny = 1e-10 floor, so an initial error in them never decays — it
#              is a permanently wrong plant model rather than a bad guess. Measured at
#              5×: km1 drew to −0.05 from a true 1.0, i.e. thrust acting backwards, and
#              the UKF then diverged at EVERY Q_ζ (also at 3×). Deliberately untouched.
#
# NB z is unobservable from this sensor set — it reaches h() only through r = g/z, which
# is not among (φ, ψ, γ) — so its bad guess is inert and simply persists. Included so
# the set is honestly "everything unmeasured and non-constant", not because it acts.
#
# HOW MUCH. Two passes, drawn as σ₀ = scale × 0.1 × max(|x₀|, 0.1) with P₀ scaled to
# match, so the filter is told how wrong it is. '10' is the script's own long-standing
# default and therefore the BASELINE; '50' is the stressed case.
#   Note the max(|x₀|, 0.1) floor: only z, v_para and w are near enough to 0.1 in
#   magnitude for the percentage to be literal. x, y, v_perp and phi_dot start at or
#   near zero, so for them the floor dominates and the perturbation is an ABSOLUTE
#   0.01/0.05 rather than a relative 10%/50%. The per-state σ₀ is written to the CSV.
#
# NOTE the draw changes with the scale, so the 50% pass is NOT the same guess
# realization as the others; the 10% pass, being the default scale, IS. Either way ζ̂(0)
# is held identical by re-pinning against each pass's own draw.
BAD_STATES = ('x', 'y', 'z', 'v_para', 'v_perp', 'phi_dot', 'w')
BAD_X0_SCALES = (('10', 1.0), ('50', 5.0))   # (filename tag, × the default 10%)

# Each bad-guess level is run TWICE: once with P₀ honest (scaled to the draw, as
# everywhere else in this script) and once with P₀ deliberately INFLATED on those same
# states — σ₀ handed to the filter is BAD_P0_INFLATE × the honest value, so the filter
# is UNDERconfident: it believes the initial guess even less than it deserves and
# therefore leans harder on the measurements. The guess itself is bit-identical between
# the two passes (p0_sigma is applied AFTER the draw), so the pair isolates the effect
# of the covariance alone. NB the opposite error — P₀ too SMALL for the guess, the
# classic divergence-inducer — is deliberately NOT what this does.
BAD_P0_INFLATE = 10.0           # × σ₀ (so ×100 in variance) on BAD_STATES

# arrowheads — a FIXED size in DATA units, shared by every panel of every figure, so
# arrow size never encodes anything and panels stay comparable (the per-panel
# `span`-scaled sizing used previously made small paths sprout huge wedges)
WEDGE_R = 0.0038                # [m]
WEDGE_ANGLE = 33.0              # [deg]
AXES_PAD = 0.010                # [m]

RUN_DATE = os.environ.get('RUN_DATE') or datetime.date.today().isoformat()
OUTDIR = os.path.join(output_dir('turn_control_sweep_noupwind_output' if NOUPWIND
                                 else 'turn_control_sweep_output'), RUN_DATE)

# the 6 families are independent; N_JOBS>1 runs them in parallel (one per worker)
N_JOBS = int(os.environ.get('N_JOBS', 1))

EV_NORM = mpl.colors.LogNorm(EV_VMIN, EV_VMAX)


# ─────────────────────────── the one control rule ───────────────────────────
def leg_length(amp_deg, radius):
    """ Straight lead-in / lead-out that lands the symmetric turn on (X_END, 0). """
    d = np.radians(amp_deg)
    return (X_END - 2.0 * radius * np.sin(d / 2.0)) / (2.0 * np.cos(d / 2.0))


def vee_setpoints(dt, amp_deg, omega, v):
    """ Per-step (speed, heading) set-points for one row: straight → constant-rate
    turn of amp_deg → straight, symmetric about the along-wind axis. """
    d = np.radians(amp_deg)
    radius = v / omega if omega > 0 else np.inf
    if d <= 0:
        # the FLAT control row: no turn at all, a straight (0,0) → (X_END, 0) run at the
        # same speed. ω is meaningless here and the closed-form leg/t_turn both
        # degenerate (t_turn = 0 would divide by zero in the heading ramp below), so
        # this is built directly. n_turn = 1 keeps every downstream consumer of the
        # turn interval well defined; the shaded "turn" band is then one step wide and
        # effectively invisible, which is the honest picture.
        n = int(round((X_END / v) / dt)) + 1
        if n < W + 10:
            raise ValueError(f'flat row: only {n} steps for a {W}-step window')
        return (np.full(n, float(v)), np.full(n, HEADING_ROT), n // 2, 1,
                dict(radius=np.inf, leg=0.5 * X_END, t_leg=0.5 * X_END / v,
                     t_turn=0.0, T=(n - 1) * dt))
    leg = leg_length(amp_deg, radius)
    if leg <= 0:
        raise ValueError(f'Δ={amp_deg:g}°, ω={omega:g} rad/s, v={v:g} m/s: the turn '
                         f'alone already overshoots x_end={X_END:g} m')
    t_leg, t_turn = leg / v, d / omega
    n = int(round((2 * t_leg + t_turn) / dt)) + 1
    if n < W + 10:
        raise ValueError(f'only {n} steps — too few for a {W}-step window')
    t = np.arange(n) * dt
    dphi = TURN_SIGN * d
    heading = (HEADING_ROT - 0.5 * dphi
               + dphi * np.clip((t - t_leg) / t_turn, 0.0, 1.0))
    speed = np.full(n, float(v))
    k0 = int(np.searchsorted(t, t_leg))
    nt = max(int(round(t_turn / dt)), 1)
    return speed, heading, k0, nt, dict(radius=radius, leg=leg, t_leg=t_leg,
                                        t_turn=t_turn, T=(n - 1) * dt)


def build_mpc_tracking(sim):
    """ The spec's own objective with R_TERM in place of the inherited 1e-6, so the
    commanded trajectory is actually flown at every turn rate. See R_TERM. """
    cost = ((sim.model.x['v_para'] - sim.model.tvp['v_para_set']) ** 2 +
            (sim.model.x['v_perp'] - sim.model.tvp['v_perp_set']) ** 2 +
            (sim.model.x['phi'] - sim.model.tvp['phi_set']) ** 2)
    sim.mpc.set_objective(mterm=cost, lterm=cost)
    sim.mpc.set_rterm(u_para=R_TERM, u_perp=R_TERM, u_phi=R_TERM)


def _mpc(engine, spec, sp, hd, zeta, x0=None):
    """ One MPC inverse-dynamics rollout of a commanded (speed, heading) profile.
    Thin wrapper on the engine's cached simulator that also accepts an explicit
    initial state, which ``ObservabilityEngine.simulate_mpc`` does not expose. """
    sim = engine._get_mpc_sim()
    sim.update_dict(spec.make_setpoint(sp, hd, WIND, zeta), name='setpoint')
    with contextlib.redirect_stdout(io.StringIO()):
        t, x_sim, u_sim, _ = sim.simulate(x0=x0, mpc=True, return_full_output=True)
    return (t, np.vstack(list(x_sim.values())).T, np.vstack(list(u_sim.values())).T)


def replan_through_shift(engine, spec, X, U, sp, hd, k_s, dz):
    """ Family (d): the wind direction steps by `dz` at step `k_s` and the MPC RE-PLANS
    from there, tracking the same commanded (speed, heading) profile.

    A two-stage rollout with a state handoff, not a splice of ζ into a finished
    trajectory: ζ enters both h() and the drag terms, so an after-the-fact edit would
    be dynamically inconsistent (the objection raised in tempo_drifting_wind.py).
    Stage 1 flies the whole profile at ζ = ZETA; the state at `k_s` is taken verbatim,
    ζ is advanced by `dz`, and stage 2 flies the REMAINDER from there. Within each
    stage x_{k+1} = f(x_k, u_k) holds, and at the junction the only difference is the
    ζ increment — exactly the discrete model x_{k+1} = f(x_k,u_k) + w_k with w_k on ζ
    that both the Gramian and the filter assume.

    Re-integrating OPEN LOOP from the stage-1 controls instead (the random-walk method
    of tempo_drifting_wind.py) is wrong here: a 90° step of a 0.4 m/s wind is
    comparable to the fly's own 0.29 m/s ground speed, so the uncorrected path leaves
    the family-(b) track entirely and the comparison with (b) is destroyed.

    The re-plan holds the ground track to ~0.2 mm, but it is a CLOSED-LOOP recovery:
    the fly notices the error and corrects it, so φ̇ still wobbles by ~0.05 rad/s at
    the junction. Family (e) removes even that. """
    zi = spec.state_names.index('zeta')
    x0 = {n: float(X[k_s, i]) for i, n in enumerate(spec.state_names)}
    x0['zeta'] = float(X[k_s, zi] + dz)
    _, X2, U2 = _mpc(engine, spec, sp[k_s:], hd[k_s:], x0['zeta'], x0=x0)
    return np.vstack([X[:k_s], X2]), np.vstack([U[:k_s], U2])


# the three force-level states: fixing these fixes the whole trajectory, because
# x, y and φ are their exact integrals
HOLD_STATES = ('v_para', 'v_perp', 'phi_dot')


def hold_course_through_shift(engine, spec, Xn, Un, k_s, dz, iters=6, tol=1e-14):
    """ Family (e): the wind direction steps by `dz`, and the trajectory does NOT
    change at all — the fly re-aims its force output to cancel the wind's changed
    drag exactly, feed-forward, as though it already knew.

    Why this is possible in closed form. In ``fly_f`` the wind reaches the dynamics
    ONLY through the apparent-airflow terms

        a_para = v_para − w·cos(φ − ζ),      a_perp = v_perp + w·sin(φ − ζ)

    which appear as the drag forces −C_para·a_para and −C_perp·a_perp, i.e. additively
    alongside the controls km1·u_para and km3·u_perp. The control therefore has full
    authority over exactly the channel the wind direction perturbs, so a change in ζ
    can be absorbed by a change in u with the state derivative left untouched. ζ does
    NOT enter x_dot, y_dot or φ_dot at all, so no correction is even needed there.

    Rather than write the algebraic correction out (which is exact only in continuous
    time — over one RK4 step φ turns, so a control held constant across the step
    cannot cancel a drag term that varies within it), this solves for the control that
    reproduces the nominal NEXT STATE under the engine's own RK4 step:

        find u_k  such that  [f_rk4(x_k, u_k)]_{v_para, v_perp, φ̇} = x_nom_{k+1}

    Three unknowns, three equations, and the dynamics are affine in u, so Newton with
    a finite-difference Jacobian converges to machine precision in 2–4 iterations. The
    remaining states are exact integrals of these three, so matching them pins the
    whole trajectory: measured agreement is 1e-16 in the matched states, ~2e-4 rad in
    φ and ~40 µm in position over a 200 mm path.

    The trajectory is a genuine rollout throughout — x_{k+1} = f_rk4(x_k, u_k) holds
    at every step by construction — with the single ζ increment at `k_s` as the only
    departure, so it is the same discrete model the Gramian and filters assume. What
    changes across the shift is the CONTROL (by ~0.83 in u_para and u_perp) and the
    SENSORY signal; the motion does not. """
    sim = engine._rk4_sim()
    zi = spec.state_names.index('zeta')
    mi = [spec.state_names.index(n) for n in HOLD_STATES]
    X, U = Xn.copy(), Un.copy()
    X[k_s:, zi] = Xn[k_s:, zi] + dz
    for k in range(k_s, len(X) - 1):
        u = U[k].copy()
        for _ in range(iters):
            r = sim._rk4_step(X[k], u)[mi] - Xn[k + 1][mi]
            if np.max(np.abs(r)) < tol:
                break
            J = np.empty((3, 3))
            for j in range(3):
                h = 1e-6 * max(abs(u[j]), 1.0)
                du = np.zeros(3)
                du[j] = h
                J[:, j] = (sim._rk4_step(X[k], u + du)[mi]
                           - sim._rk4_step(X[k], u - du)[mi]) / (2 * h)
            u = u - np.linalg.solve(J, r)
        U[k] = u
        X[k + 1] = sim._rk4_step(X[k], u)
    return X, U


def rollout(engine, spec, amp_deg, omega, v, shift=None, hold=False):
    """ MPC inverse-dynamics rollout of one commanded profile.

    `shift` = (step, dzeta) makes the WIND DIRECTION really step at that time step.
    `hold` picks how the fly deals with it: False re-plans (family d), True cancels
    the wind change feed-forward so the trajectory is invariant (family e). """
    dt = spec.dt
    sp, hd, k0, nt, geo = vee_setpoints(dt, amp_deg, omega, v)
    zi = spec.state_names.index('zeta')
    t, X, U = _mpc(engine, spec, sp, hd, ZETA)
    zeta_true = np.full(len(t), ZETA)
    dev = 0.0
    if shift is not None:
        Xn = X.copy()
        X, U = (hold_course_through_shift(engine, spec, Xn, U, *shift) if hold else
                replan_through_shift(engine, spec, X, U, sp, hd, *shift))
        dev = float(np.max(np.linalg.norm(X[:, :2] - Xn[:, :2], axis=1)))
        row_dev = {n: float(np.max(np.abs(X[:, i] - Xn[:, i])))
                   for i, n in enumerate(spec.state_names) if n != 'zeta'}
        geo = dict(geo, dev_phi_dot=row_dev['phi_dot'], dev_phi=row_dev['phi'],
                   dev_v_para=row_dev['v_para'])
        zeta_true = X[:, zi].copy()
    else:
        geo = dict(geo, dev_phi_dot=0.0, dev_phi=0.0, dev_v_para=0.0)
    engine.set_trajectory(t, X, U)

    X = engine.X
    pd_ = X[:, spec.state_names.index('phi_dot')]
    g = np.hypot(X[:, spec.state_names.index('v_para')],
                 X[:, spec.state_names.index('v_perp')])
    hdg_err = float(np.max(np.abs(X[:, spec.state_names.index('phi')] - hd)))
    return dict(amp=amp_deg, omega=omega, v=v, k_turn0=k0, n_turn=nt, N=engine.N,
                X=X.copy(), U=engine.U.copy(), t=engine.t.copy(),
                heading=heading_of(engine, spec).copy(), zeta_true=zeta_true,
                peak_phidot=float(np.max(np.abs(pd_))),
                # "did the fly actually fly what it was told" — measured on the
                # HEADING it was commanded, not on peak |φ̇| (which the MPC rounds off
                # at the corners of a short ramp even when tracking is fine)
                tracked=bool(np.degrees(hdg_err) < 10.0),
                hdg_err_deg=float(np.degrees(hdg_err)),
                g_mean=float(np.mean(g)), g_cross=float(g[min(k0 + nt // 2,
                                                              len(g) - 1)]),
                path_len=float(np.sum(np.hypot(*np.diff(X[:, :2], axis=0).T))),
                end=X[-1, :2].copy(), dev=dev, **geo)


def dgamma_dzeta(v, beta=None):
    """ ∂γ/∂ζ — how strongly the apparent-airflow direction reports the wind
    direction — at apparent-airflow angle β = φ − ζ. From

        a_para = v − w·cos β,   a_perp = w·sin β,   γ = atan2(a_perp, a_para)
        ⇒ ∂γ/∂ζ = −w·(v·cos β − w) / (a_para² + a_perp²)

    This reduces to the familiar −w/(v−w) when the fly heads WITH the wind (β = 0),
    where it has a pole at v = w, and to +w/(v+w) heading INTO the wind (β = π), where
    there is NO pole and the sensitivity simply falls off with speed. The default β is
    the mid-turn value: the heading sweep is symmetric about 0, so φ = 0 there and
    β = −ζ. """
    if beta is None:
        beta = -ZETA
    a_para = v - WIND * np.cos(beta)
    a_perp = WIND * np.sin(beta)
    d = a_para ** 2 + a_perp ** 2
    return np.inf if d < 1e-18 else -WIND * (v * np.cos(beta) - WIND) / d


# ──────────────────────────────── the bounds ────────────────────────────────
def row_bounds(engine, spec, row):
    """ Sliding min-EV(ζ) for every Q_ζ and both Gramians, plus the per-sample
    mapping used to colour the path (each sample takes its nearest window's value,
    windows attributed to their CENTRE). """
    r_diag = dict(spec.r_default)
    sensors = tuple(spec.fim_sensors)            # ('phi', 'psi', 'gamma')
    q_floor = dict(spec.q_tiny)
    t_s = np.arange(row['N']) * spec.dt
    ev, evp, metrics = {}, {}, {}
    for iq, qd in enumerate(Q_ZETA):
        q_diag = dict(q_floor)
        q_diag['zeta'] = float(qd)
        for b in BASES:
            with contextlib.redirect_stdout(io.StringIO()):
                df = engine.linearized_ev(W, q_diag, r_diag, LAM, sensors=sensors,
                                          states=('zeta',), basis=b)
            ev[(b, iq)] = df
            val = df['zeta'].values
            evp[(b, iq)] = val[np.clip(np.searchsorted(df['time'].values, t_s),
                                       0, len(val) - 1)]
            fin = val[np.isfinite(val)]
            # `mean` is NOT ceiling-free: windows in which zeta is unobservable pin
            # at the regularizer ceiling 1/lam, so the mean partly measures the
            # FRACTION of informative windows on a scale lam sets. satfrac records
            # how much of each mean is just that ceiling.
            metrics[(b, iq)] = ((float(np.min(fin)), float(np.mean(fin)),
                                 float(np.mean(fin >= 0.5 / LAM)))
                                if len(fin) else (np.nan, np.nan, np.nan))
    row['ev'], row['evp'], row['metrics'] = ev, evp, metrics
    return row


_ANN_ATTACHED = {}
ANN_KW = {}


def _attach_ann(engine, spec):
    """ Put the causal ζ net into the engine's ANN cache so run_aikf finds it
    instead of training one, and record the kwargs that identify it. """
    global ANN_KW
    if id(engine) in _ANN_ATTACHED:
        return
    from twe.zeta_ann_estimator import attach
    ann, kw = attach(engine, spec, seed=SEED)
    ANN_KW = kw
    _ANN_ATTACHED[id(engine)] = ann
    print(f'  ANN: causal ζ net, ω={ann.time_steps} steps '
          f'({ann.time_steps * spec.dt:.2f} s), sensors {ann.sensors}', flush=True)


def row_filters(engine, spec, row, offset):
    """ EKF and UKF estimates of ζ at every Q_ζ, both driven by the same angular
    sensors and the same measurement realization (one seed, one trajectory version, so
    the two filters see identical data — see tempo_filter_sweep.py).

    A filter tuned to a nearly static wind (small Q_ζ) can fail outright on the +90°
    step of families (d) and (e). The mechanism is NOT "small Q means a small gain and
    therefore a lazy estimate", which would merely bias ζ̂: it is a MISATTRIBUTION.
    With Q_ζ tiny, the filter's ζ variance is tiny, so the sustained post-step
    innovation cannot be explained by ζ and the update dumps it into the 18-state
    model's PARAMETER states instead — which carry only the q_tiny = 1e-10 floor and
    so are also held nearly fixed. Traced step by step at Q_ζ = 1e-5, ω = 5 rad/s:
    C_para is driven to −2.4 (true +1.47), i.e. NEGATIVE drag, which makes the filter's
    own plant model unstable; φ̇ reaches −337 within two steps, then v_para 1e38, 1e87
    and NaN. Raising Q_ζ gives the error a legitimate channel to live in, so the
    parameters are left alone and the filter survives — which is why only the SMALL Q_ζ
    columns fail. The fly7 spec, which folds those parameters into constants, has no
    such channel to corrupt.

    A failed run is recorded as DIVERGED — no curve drawn, flagged in the CSV — rather
    than replaced by something plottable. """
    r_diag = dict(spec.r_default)
    q_floor = dict(spec.q_tiny)
    zi = spec.state_names.index('zeta')
    # The initial estimate and the noisy measurement record are drawn ONCE per
    # (trajectory, seed) inside _estimation_problem, so every filter and every Q_ζ in
    # this row starts from the identical guess and sees the identical data — that part
    # needs no help. What DOES need help is comparability ACROSS rows: the draw is keyed
    # on the trajectory version, so each row would otherwise start from a different
    # prior, which is not a controlled comparison of turn tempo.
    #
    # x0_offset is applied AFTER the draw, so replaying the same draw here and asking
    # for the offset that lands ζ̂(0) exactly on ZETA + `offset` pins it to the intended
    # value in every row, using only the public argument. The other 17 states keep their
    # random priors (unchanged from every other script here); the residual σ₀(ζ) that
    # results is recorded per row so it can be audited.
    target = ZETA + offset

    def pinned(scale):
        """ ζ̂(0) pinned to `target` for a given x0_scale. The pin has to be redone per
        scale: x0_scale changes the DRAW, so the offset that lands ζ̂(0) on target for
        the default scale lands somewhere else for another one. """
        _, raw, _, _, _ = engine._estimation_problem(r_diag, SEED, scale, None)
        off = {'zeta': float(target - raw[zi])}
        _, xh, P, _, _ = engine._estimation_problem(r_diag, SEED, scale, off)
        if abs(xh[zi] - target) > 1e-9:
            raise AssertionError(f'ζ̂(0) pinning failed: {xh[zi]} vs {target}')
        return off, xh, P

    _attach_ann(engine, spec)
    x0_off, xh0, P0 = pinned(1.0)
    # passes 1 (figure of record) and 2 (no-prior, NOPRIOR_SIGMA_ZETA) share one draw;
    # each bad-guess pass re-pins against its own. See BAD_STATES for the state choice.
    passes = [('', dict(x0_offset=x0_off, start_at='posterior')),
              ('_np', dict(x0_offset=x0_off, p0_sigma={'zeta': NOPRIOR_SIGMA_ZETA},
                           start_at='posterior'))]
    x0t = engine.X[0]
    bi = [spec.state_names.index(nm) for nm in BAD_STATES]
    for tag, sc in BAD_X0_SCALES:
        scale = {nm: sc for nm in BAD_STATES}
        off_b, xh_b, P_b = pinned(scale)
        honest = {nm: float(np.sqrt(P_b[i, i])) for nm, i in zip(BAD_STATES, bi)}
        relerr = float(np.max(np.abs(xh_b[bi] - x0t[bi])
                              / np.maximum(np.abs(x0t[bi]), 0.1)))
        # the same guess judged two ways: honest P₀, then P₀ inflated ×BAD_P0_INFLATE.
        # p0_sigma is applied after the draw, so xh_b is bit-identical in both.
        for isfx, p0s in (('', None),
                          ('i', {nm: BAD_P0_INFLATE * s for nm, s in honest.items()})):
            passes.append((f'_bad{tag}{isfx}',
                           dict(x0_offset=off_b, x0_scale=scale,
                                start_at='posterior',
                                **({'p0_sigma': p0s} if p0s else {}))))
            row[f'sig0_bad{tag}{isfx}'] = p0s if p0s else honest
            row[f'relerr_bad{tag}{isfx}'] = relerr
        row[f'xhat0_bad{tag}'] = xh_b.copy()
    for sfx, extra in passes:
        est, diverged = {}, {f: [] for f in FILTERS}
        # The ANN reads MEASUREMENTS only — no dynamics, no Q — so its estimate
        # is computed ONCE per pass, independently of any filter. It used to be
        # lifted out of run_aikf's return, which silently turned an AI-KF
        # failure into a fake "ANN diverged" at that Q_ζ.
        try:
            ep = {k: v for k, v in extra.items()
                  if k in ('x0_scale', 'x0_offset', 'p0_sigma')}
            Y_ann, _, _, _, U_ann = engine._estimation_problem(r_diag, SEED, **ep)
            ann_series = _ANN_ATTACHED[id(engine)].predict_series(Y_ann, U_ann)
        except Exception:
            ann_series = None
        for qd in Q_ZETA:
            q_diag = dict(q_floor)
            q_diag['zeta'] = float(qd)
            for name in FILTERS:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        if name == 'AIKF':
                            Xh, _, _, _ = engine.run_aikf(
                                'zeta', q_diag, r_diag, seed=SEED,
                                sensors=tuple(spec.fim_sensors),
                                motif_input=AIKF_MOTIF,
                                motif_window=AIKF_WINDOW,
                                motif_upper=AIKF_UPPER,
                                R_ann_lo=AIKF_R_LO, R_ann_hi=AIKF_R_HI,
                                **ANN_KW, **UKF_KW, **extra)
                            z = Xh[:, zi].copy()
                        elif name == 'ANN':
                            # the raw net output: no dynamics, no Q dependence,
                            # so the same series stands at every Q_ζ
                            z = (np.full(row['N'], np.nan) if ann_series is None
                                 else np.asarray(ann_series, dtype=float).copy())
                        else:
                            fn = (engine.run_ekf if name == 'EKF'
                                  else engine.run_ukf)
                            kw = dict(UKF_KW) if name == 'UKF' else {}
                            Xh, _ = fn(q_diag, r_diag, seed=SEED,
                                       sensors=tuple(spec.fim_sensors),
                                       **kw, **extra)
                            z = Xh[:, zi].copy()
                    # the ANN has no estimate until its window fills, so leading
                    # NaNs are EXPECTED there and are not divergence
                    chk = z[ANN_KW['time_steps'] - 1:] if name == 'ANN' else z
                    if not len(chk) or not np.all(np.isfinite(chk)):
                        raise np.linalg.LinAlgError('non-finite estimate')
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    z = np.full(row['N'], np.nan)
                    diverged[name].append(qd)
                est[(name, qd)] = z
        row['est' + sfx], row['diverged' + sfx] = est, diverged
    row['zhat0'] = float(xh0[zi])
    row['sig0_zeta'] = float(np.sqrt(P0[zi, zi]))
    row['sig0_zeta_np'] = NOPRIOR_SIGMA_ZETA
    row['xhat0'] = xh0.copy()
    return row


# ─────────────────────────────── the families ───────────────────────────────
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def build_family(engine, spec, key, rows_spec, shift=None, offset=ZETA_OFFSET,
                 hold=False):
    """ Run every row of one family: rollout → bounds → filters. """
    print(f'\n{"#" * 78}\n### {key}\n{"#" * 78}', flush=True)
    rows = []
    for rs in rows_spec:
        row = rollout(engine, spec, rs['amp'], rs['omega'], rs['v'], shift=shift,
                      hold=hold)
        row.update(rs)
        row_bounds(engine, spec, row)
        row_filters(engine, spec, row, offset)
        rows.append(row)
        print(f'  {row["label"]:<26s} Δ={row["amp"]:5.1f}° ω={row["omega"]:5.2f} '
              f'v={row["v"]:.3f} R={row["radius"]*1e3:6.1f}mm leg={row["leg"]*1e3:6.1f}mm '
              f'N={row["N"]:4d} T={row["T"]:.2f}s L={row["path_len"]:.4f}m '
              f'end=({row["end"][0]:.4f},{row["end"][1]:+.4f}) '
              f'peak|φ̇|={row["peak_phidot"]:5.2f} hdg_err={row["hdg_err_deg"]:4.1f}° '
              f'∂γ/∂ζ={dgamma_dzeta(row["g_cross"]):+6.2f}'
              + (f' dev={row["dev"]*1e6:.1f}um  φ̇dev={row["dev_phi_dot"]:.1e}'
                 if shift else '')
              + ''.join(f'  ** {f} diverged at Q_ζ='
                        f'{", ".join(f"{q:g}" for q in row["diverged"][f])} **'
                        for f in FILTERS if row['diverged'][f])
              + ('' if row['tracked'] else
                 '  ** MPC did not track the commanded heading (>10°) **'),
              flush=True)
    z0 = np.array([r['zhat0'] for r in rows])
    sg = np.array([r['sig0_zeta'] for r in rows])
    print(f'  filter init: ζ̂(0) pinned to {ZETA:+.4f}{offset:+.4f} = '
          f'{ZETA + offset:+.4f} rad in EVERY row (spread '
          f'{np.ptp(z0):.2e}), shared by EKF, UKF and every Q_ζ; '
          f'σ₀(ζ) {sg.min():.3f}–{sg.max():.3f}', flush=True)
    def _div(sfx):
        return ''.join(
            f'  ** {f} diverged at Q_ζ='
            f'{", ".join(f"{q:g}" for q in sorted({q for r in rows for q in r["diverged" + sfx][f]}))} **'
            for f in FILTERS if any(r['diverged' + sfx][f] for r in rows))

    print(f'  no-prior pass: same ζ̂(0), but σ₀(ζ) = {NOPRIOR_SIGMA_ZETA:.3f} rad '
          f'(uniform on the circle) and ζ̂(0) treated as x̂₀|₀ (y₀ dropped)'
          + _div('_np'), flush=True)
    for tag, _sc in BAD_X0_SCALES:
        rel = np.array([r[f'relerr_bad{tag}'] for r in rows])
        for isfx in ('', 'i'):
            print(f'  bad-guess pass {tag}%'
                  + (f' P₀×{BAD_P0_INFLATE:g}' if isfx else ' P₀ honest')
                  + f': σ₀ = {tag}% × max(|x₀|,0.1) on {"/".join(BAD_STATES)} '
                  f'(constants, φ and ζ left alone; worst realized error '
                  f'{100*rel.min():.0f}–{100*rel.max():.0f}%), ζ̂(0) as x̂₀|₀'
                  + _div(f'_bad{tag}{isfx}'), flush=True)
    ends = np.array([r['end'] for r in rows])
    print(f'  endpoint spread across the family: '
          f'{1e3*np.max(np.linalg.norm(ends - ends.mean(0), axis=1)):.2f} mm; '
          f'durations {min(r["T"] for r in rows):.2f}–{max(r["T"] for r in rows):.2f} s',
          flush=True)
    return rows


def flat_row(color):
    """ The Δ = 0 control row shared by families (a) and (b): a straight run over the
    same (0,0) → (X_END, 0) chord at the same speed, so it is the same DISPLACEMENT as
    every other row but the only one with no turn in it. Being shorter than any vee it
    is also the shortest in TIME (X_END/v = 0.69 s), which is worth remembering when
    reading a whole-trajectory 'mean' metric against it. """
    return dict(amp=0.0, omega=0.0, v=V_CRUISE, group='', color=color,
                label='flat', legend='Δ = 0°  (no turn)   '
                                     f'(straight {X_END:g} m, T = {X_END/V_CRUISE:.2f} s)')


def spec_a():
    cm = plt.get_cmap('viridis')
    n = len(A_AMPS)
    return [flat_row('#b03060')] + \
           [dict(amp=a, omega=A_OMEGA, v=V_CRUISE, group='',
                 color=cm(i / max(n - 1, 1)),
                 label=f'amp{int(a):03d}',
                 legend=f'Δ = {a:.0f}°   (T_turn = {np.radians(a)/A_OMEGA*1e3:.0f} ms, '
                        f'R = {V_CRUISE/A_OMEGA*1e3:.0f} mm)')
            for i, a in enumerate(A_AMPS)]


def spec_b():
    cm = plt.get_cmap('viridis')
    n = len(B_OMEGAS)
    return [flat_row('#b03060')] + \
           [dict(amp=B_AMP, omega=om, v=V_CRUISE, group='',
                 color=cm(i / max(n - 1, 1)),
                 label=f'w{om:04.1f}',
                 legend=f'ω = {om:.1f} rad/s   ({np.degrees(om):.0f}°/s, T_turn = '
                        f'{np.radians(B_AMP)/om*1e3:.0f} ms, R = '
                        f'{V_CRUISE/om*1e3:.0f} mm)')
            for i, om in enumerate(B_OMEGAS)]


def spec_c():
    cm = plt.get_cmap('plasma')
    n = len(C_SPEEDS)
    out = []
    for a in C_AMPS:
        for i, v in enumerate(C_SPEEDS):
            out.append(dict(amp=a, omega=v / C_RADIUS, v=v,
                            group=f'Δ = {a:.0f}°',
                            color=cm(0.08 + 0.78 * i / max(n - 1, 1)),
                            label=f'amp{int(a):03d}_v{int(round(v*1000)):03d}',
                            legend=f'v = {v:.3f} m/s   (ω = {v/C_RADIUS:.1f} rad/s, '
                                   f'∂γ/∂ζ = {dgamma_dzeta(v):+.2f})'))
    return out


def _turning_b(suffix):
    """ Family (b)'s rows WITHOUT its flat control, relabelled. The Δ = 0 row belongs
    to (a) and (b), where the question is what a turn buys over not turning. In (d)–(f)
    the subject is the wind STEP and the rows are turn rates, so a row with no turn has
    no ω to sweep; worse, its n_turn = 1 would collapse the common turn window that
    turn_windows derives from the fastest turn (4 steps at ω = 40) down to a single
    step, silently changing what every cell of those matrices measures. """
    return [dict(r, label=r['label'] + suffix) for r in spec_b() if r['amp'] > 0]


def spec_d():
    return _turning_b('_shift')


def spec_e():
    return _turning_b('_hold')


def spec_f():
    return _turning_b('_late')


# ────────────────────────────── shared drawing ──────────────────────────────
def short_label(r, sep='\n'):
    """ The row's swept quantity, prefixed by its group where a family has one. """
    s = r['legend'].split('   ')[0]
    return f'{r["group"]}{sep}{s}' if r['group'] else s


def family_limits(rows, pad=AXES_PAD):
    x = np.concatenate([r['X'][:, 0] for r in rows])
    y = np.concatenate([r['X'][:, 1] for r in rows])
    return (x.min() - pad, x.max() + pad), (y.min() - pad, y.max() + pad)


def draw_path(ax, row, color=None, values=None, cmap=EV_CMAP, norm=EV_NORM,
              lw_base=0.8, zorder=3, wedge_r=WEDGE_R):
    """ One arrowhead PER DATA POINT, at a fixed size in data units. Either a flat
    colour (`color`) or a per-sample scalar (`values`, e.g. min-EV(ζ)). """
    X, hdg = row['X'], row['heading']
    ax.plot(X[:, 0], X[:, 1], '-', color='0.86', lw=lw_base, zorder=zorder - 1)
    if values is None:
        cmap_use = mpl.colors.ListedColormap([color])
        norm_use = mpl.colors.Normalize(-1, 1)
        vals = np.zeros(len(X))
    else:
        cmap_use, norm_use, vals = plt.get_cmap(cmap), norm, np.asarray(values)
    pc = heading_wedges(X[:, 0], X[:, 1], vals, hdg, norm_use, cmap_use,
                        size_radius=wedge_r, size_angle=WEDGE_ANGLE, nskip=0)
    pc.set_zorder(zorder)
    ax.add_collection(pc)
    return pc


def mark_ends(ax, row, ms_start=7, ms_end=11):
    X = row['X']
    ax.plot(X[0, 0], X[0, 1], 'o', color='#1a9850', ms=ms_start, zorder=8,
            mec='k', mew=0.6)
    ax.plot(X[-1, 0], X[-1, 1], '*', color='#d73027', ms=ms_end, zorder=8,
            mec='k', mew=0.4)


def wind_glyph(ax, shift=False, fs=8.0):
    """ Wind reference arrow, drawn in AXES coordinates in the headroom `tidy`
    reserves at the top, so it can never land on a trajectory. With `shift`, two
    arrows — the wind before and after the step — plus the arc between them. """
    tf = ax.transAxes
    r = 0.055
    # anchored so the arrows point INTO the empty headroom whatever ζ is
    x, y = (0.16, 0.82) if shift else (0.16, 0.91)

    def arrow(ang, col):
        ax.add_patch(mpatches.FancyArrow(
            x, y, r * np.cos(ang), r * np.sin(ang), transform=tf, width=0.008,
            head_width=0.026, head_length=0.026, color=col,
            length_includes_head=True, zorder=10, clip_on=False))

    lx = x + r + 0.03
    if not shift:
        arrow(ZETA, 'steelblue')
        ax.text(lx, y, f'wind {WIND:g} m/s,  ζ = {np.degrees(ZETA):.0f}°',
                transform=tf, ha='left', va='center', fontsize=fs, color='steelblue',
                zorder=10, clip_on=False)
        return
    a1, a2 = ZETA, ZETA + D_SHIFT_RAD
    arrow(a1, 'steelblue')
    arrow(a2, '#c0392b')
    # an Arc from a1 to a2 sweeps counter-clockwise by construction, so it shows the
    # direction of the shift for ANY ζ without a hand-picked curvature sign
    ax.add_patch(mpatches.Arc((x, y), 1.5 * r, 1.5 * r, transform=tf,
                              theta1=np.degrees(a1), theta2=np.degrees(a2),
                              color='0.30', lw=1.5, zorder=11, clip_on=False,
                              linestyle=(0, (3, 2))))
    ax.text(lx, y + 0.035, f'wind {WIND:g} m/s,  ζ = {np.degrees(ZETA):.0f}°   '
            f'(t < {SHIFT_STEP * 0.01:.2f} s)', transform=tf, ha='left',
            va='center', fontsize=fs, color='steelblue', zorder=10, clip_on=False)
    ax.text(lx, y - 0.035, f'ζ = {np.degrees(ZETA + D_SHIFT_RAD):.0f}°   '
            f'(t ≥ {SHIFT_STEP * 0.01:.2f} s)   [+90°]', transform=tf,
            ha='left', va='center', fontsize=fs, color='#c0392b', zorder=10,
            clip_on=False)


def tidy(ax, xlim, ylim, shift=False):
    """ Equal aspect, plus empty headroom at the top for the wind glyph so no
    annotation can ever sit on a trajectory. """
    ax.set_xlim(*xlim)
    span = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + (0.75 if shift else 0.38) * span)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.28)


# ────────────────────────────────── figures ─────────────────────────────────
def fig_family_grid(rows, title, subtitle, outdir):
    """ Family (c) only. There the members of a group trace the IDENTICAL ground
    track by construction, so an overlay would hide every one but the last; what
    actually differs is how many samples the path is observed with. One panel per
    trajectory makes both readable: identical shape down a column, visibly different
    arrowhead spacing across a row. """
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r['group']:
            groups.append((r['group'], []))
        groups[-1][1].append(r)
    nr, nc = len(groups), max(len(g[1]) for g in groups)
    xlim, ylim = family_limits(rows)
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc + 0.6, 3.3 * nr + 1.0),
                             squeeze=False, sharex=True, sharey=True)
    for i, (gname, grows) in enumerate(groups):
        for j in range(nc):
            ax = axes[i, j]
            if j >= len(grows):
                ax.axis('off')
                continue
            r = grows[j]
            draw_path(ax, r, color=r['color'])
            mark_ends(ax, r, ms_start=5, ms_end=9)
            if j == 0:
                wind_glyph(ax, fs=7.5)
                ax.set_ylabel(f'{gname}\ny [m]', fontsize=9.5)
            tidy(ax, xlim, ylim)
            ax.tick_params(labelsize=8)
            ax.set_title(f'v = {r["v"]:.3f} m/s,  ω = {r["omega"]:.1f} rad/s\n'
                         f'T = {r["T"]:.2f} s,  N = {r["N"]} samples,  '
                         f'∂γ/∂ζ = {dgamma_dzeta(r["v"]):+.2f}', fontsize=8.5,
                         color=r['color'])
            if i == nr - 1:
                ax.set_xlabel('x [m]', fontsize=9)
    fig.suptitle(f'{title}\n{subtitle}\none arrowhead per data point — the shape is '
                 'identical down each column (the path is FIXED); only how densely '
                 'it is sampled changes across a row', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(outdir, 'family_by_speed.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_family(rows, title, subtitle, outdir, shift=False, leg_title=None,
               note=''):
    """ The whole family on ONE set of axes — arrowhead style, one arrowhead per data
    point, colour = the swept variable, legend outside the axes to the right. """
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r['group']:
            groups.append((r['group'], []))
        groups[-1][1].append(r)
    xlim, ylim = family_limits(rows)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for r in rows:                              # slow/large first, fast/small on top
        draw_path(ax, r, color=r['color'])
    for r in rows:
        mark_ends(ax, r)
    # where a family has groups (c: one per turn angle) the members of a group trace the
    # SAME path, so name each shape at its own apex rather than in the legend, and let
    # the legend carry only the swept variable
    for gname, grows in groups:
        if not gname:
            continue
        X = grows[0]['X']
        k = int(np.argmax(X[:, 1]))
        ax.annotate(gname, xy=(X[k, 0], X[k, 1]), xytext=(0, 7),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9.5, color='0.25')
    wind_glyph(ax, shift=shift)
    tidy(ax, xlim, ylim, shift=shift)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ONE legend for the whole figure, outside the axes on the right: the groups of
    # family (c) share their entries, so a per-panel legend would repeat itself and
    # collide with the neighbouring panel's y-axis
    if note:
        ax.set_title(note, fontsize=9.5, color='0.3')
    seen, h = set(), []
    for r in rows:
        if r['legend'] in seen:
            continue
        seen.add(r['legend'])
        h.append(Line2D([], [], color=r['color'], lw=4, label=r['legend']))
    h += [Line2D([], [], color='#1a9850', marker='o', ls='none', mec='k',
                 label='shared start (0, 0)'),
          Line2D([], [], color='#d73027', marker='*', ls='none', ms=11, mec='k',
                 label=f'shared end ({X_END*np.cos(HEADING_ROT):+.2f}, '
                       f'{X_END*np.sin(HEADING_ROT):+.2f})')]
    fig.legend(handles=h, fontsize=9, loc='center left',
               bbox_to_anchor=(1.005, 0.5), frameon=True, framealpha=0.95,
               title=leg_title, title_fontsize=9)
    fig.suptitle(f'{title}\n{subtitle}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(outdir, 'family.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_traj(row, rows, title, outdir, shift=False):
    """ One trajectory: min-EV(ζ) along the path from BOTH Gramians side by side,
    over min-EV(ζ) vs time at every Q_ζ (observability solid, constructability
    dotted in the matching colour). """
    xlim, ylim = family_limits(rows)
    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0],
                          width_ratios=[1.0, 1.0, 0.05], hspace=0.30, wspace=0.20)
    iq_ref = int(np.argmin(np.abs(Q_ZETA - Q_REF)))
    for col, b in enumerate(BASES):
        ax = fig.add_subplot(gs[0, col])
        draw_path(ax, row, values=row['evp'][(b, iq_ref)])
        mark_ends(ax, row)
        wind_glyph(ax, shift=shift, fs=7.5)
        tidy(ax, xlim, ylim, shift=shift)
        ax.set_xlabel('x [m]')
        if col == 0:
            ax.set_ylabel('y [m]')
        eq = {'observability': 'Eq. 33, initial-state / smoother',
              'constructability': 'Eq. 30, final-state / causal filter'}[b]
        ax.set_title(f'stochastic {b} Gramian\n({eq})   —   Q_ζ = {Q_ZETA[iq_ref]:g} rad²',
                     fontsize=10)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=EV_NORM,
                                            cmap=plt.get_cmap(EV_CMAP)), cax=cax)
    cb.set_label('min EV of ζ  [rad²]   (lighter = better estimated)', fontsize=9)

    ax = fig.add_subplot(gs[1, :2])
    cmq = plt.get_cmap(Q_CMAP)
    for iq, qd in enumerate(Q_ZETA):
        c = cmq(0.15 + 0.85 * iq / max(len(Q_ZETA) - 1, 1))
        for b, ls in (('observability', '-'), ('constructability', ':')):
            df = row['ev'][(b, iq)]
            ax.semilogy(df['time'].values, df['zeta'].values, ls, lw=1.9, color=c)
    ax.axvspan(row['k_turn0'] * 0.01, (row['k_turn0'] + row['n_turn']) * 0.01,
               color='0.5', alpha=0.13, lw=0)
    if shift:
        ax.axvline(SHIFT_STEP * 0.01, color='#c0392b', lw=1.6, ls='--')
    ax.set_ylim(EV_VMIN, EV_VMAX)
    ax.set_xlim(0, row['t'][-1])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('min EV of ζ  [rad²]\n(LOWER = better estimated)')
    ax.grid(alpha=0.3, which='both')
    h = [Line2D([], [], color=cmq(0.15 + 0.85 * iq / max(len(Q_ZETA) - 1, 1)),
                lw=2.6, label=f'{qd:g}') for iq, qd in enumerate(Q_ZETA)]
    h += [Line2D([], [], color='0.25', ls='-', lw=2, label='observability'),
          Line2D([], [], color='0.25', ls=':', lw=2, label='constructability'),
          mpatches.Patch(color='0.5', alpha=0.2, label='the turn')]
    if shift:
        h.append(Line2D([], [], color='#c0392b', ls='--', lw=1.8,
                        label=f'wind shifts +90°'))
    ax.legend(handles=h, fontsize=8.5, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, title='Q_ζ [rad²]', title_fontsize=8.5)
    ax.set_title(f'min-EV(ζ) vs time, window {W} steps ({W*0.01:g} s) — every Q_ζ, '
                 'both Gramians', fontsize=10)

    fig.suptitle(f'{title}\n'
                 f'Δ = {row["amp"]:.0f}°, ω = {row["omega"]:.1f} rad/s, '
                 f'v = {row["v"]:.3f} m/s, R = {row["radius"]*1e3:.0f} mm, '
                 f'T = {row["T"]:.2f} s, N = {row["N"]} steps, '
                 f'wind {WIND:g} m/s, ∂γ/∂ζ = {dgamma_dzeta(row["g_cross"]):+.2f}\n'
                 'sensors φ, ψ, γ (angular only) — one arrowhead per data point',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(outdir, f'traj_{row["label"]}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def turn_windows(rows):
    """ One COMMON window length — the fastest turn's, i.e. the shortest in the family —
    centred on each row's own turn midpoint.

    Families (d)–(f) sweep ω over a 16× range, so their turns last anywhere from ~4 to
    ~63 steps. Comparing "the min-EV over the turn" across those rows would compare
    windows of wildly different length, and min-EV falls with window length for trivial
    reasons, so the slow rows would win by duration alone. Fixing the length at the
    fastest turn's removes that: every row is judged on the same amount of time, taken
    from the middle of its turn.

    NB at ω = 40 rad/s a 90° turn lasts (π/2)/40 = 0.039 s ≈ 4 steps, which equals the
    Gramian window W itself — so the common window is about one W, and this metric is
    effectively "the constructability of a single window straddling the turn". """
    turning = [r for r in rows if r['amp'] > 0]
    if not turning:
        raise ValueError('turn_windows: no turning row to take a window length from')
    # length from the TURNING rows only — a Δ = 0 control row carries the placeholder
    # n_turn = 1 and would otherwise set the common window to a single step
    nmin = max(int(min(r['n_turn'] for r in turning)), 1)
    wins = []
    for r in rows:
        mid = r['k_turn0'] + 0.5 * r['n_turn']
        lo = int(round(mid - 0.5 * nmin))
        wins.append((max(0, min(lo, r['N'] - nmin)), ))
    return nmin, [(lo, lo + nmin) for (lo,) in wins]


def fig_matrix(rows, metric, title, subtitle, outdir, turn_window=False):
    """ Trajectory × Q_ζ, both Gramians as the two subplots, on the fixed colour
    scale shared by every figure here. Thumbnails are drawn in ONE constant colour —
    the colour information lives in the heat map.

    turn_window=True restricts the statistic to the common turn window (see
    turn_windows) instead of the whole trajectory. """
    mi = 0 if metric == 'best' else 1
    nmin, wins = turn_windows(rows) if turn_window else (0, None)

    def cell(i, b, iq):
        """ (value, saturated fraction) for one heat-map cell. """
        if not turn_window:
            m = rows[i]['metrics'][(b, iq)]
            return m[mi], m[2]
        lo, hi = wins[i]
        v = rows[i]['evp'][(b, iq)][lo:hi]
        v = v[np.isfinite(v)]
        if not len(v):
            return np.nan, np.nan
        return (float(np.min(v)) if metric == 'best' else float(np.mean(v)),
                float(np.mean(v >= 0.5 / LAM)))
    n_r, n_c = len(rows), len(Q_ZETA)
    sat_note = ('' if metric == 'best' else
                '\n"% sat" = fraction of windows pinned at the 1/λ ceiling '
                f'({1/LAM:g}); where that is high the mean mostly reflects it')
    xlim, ylim = family_limits(rows)
    fig = plt.figure(figsize=(4.4 + 1.35 * n_c * 2, 0.82 * n_r + 3.4))
    outer = fig.add_gridspec(1, 4, width_ratios=[1.05, 4.6, 4.6, 0.18], wspace=0.26)
    tgs = outer[0].subgridspec(n_r, 1, hspace=0.12)
    for k, r in enumerate(rows):
        tax = fig.add_subplot(tgs[k])
        # thumbnails carry ONE constant colour — all the colour information in this
        # figure belongs to the heat map. Wedges are enlarged because a thumbnail is
        # a fraction of the width of a full trajectory panel.
        draw_path(tax, r, color='#2b2b2b', lw_base=0.5, wedge_r=3.0 * WEDGE_R)
        tax.plot(r['X'][0, 0], r['X'][0, 1], 'o', color='#1a9850', ms=3, zorder=8)
        tax.plot(r['X'][-1, 0], r['X'][-1, 1], '*', color='#d73027', ms=5, zorder=8)
        tax.set_xlim(*xlim); tax.set_ylim(*ylim)
        tax.set_aspect('equal'); tax.axis('off')

    for col, b in enumerate(BASES):
        ax = fig.add_subplot(outer[1 + col])
        M = np.array([[cell(i, b, iq)[0] for iq in range(n_c)]
                      for i in range(n_r)])
        ax.imshow(M, cmap=EV_CMAP, norm=EV_NORM, aspect='auto')
        ax.set_xticks(range(n_c))
        ax.set_xticklabels([f'{q:g}' for q in Q_ZETA], fontsize=9)
        ax.set_xlabel('wind-direction process noise   Q_ζ   [rad²]')
        if col == 0:
            ax.set_yticks(range(n_r))
            ax.set_yticklabels([short_label(r) for r in rows], fontsize=8)
        else:
            ax.set_yticks([])
        for i in range(n_r):
            for j in range(n_c):
                if not np.isfinite(M[i, j]):
                    continue
                sh = EV_NORM(np.clip(M[i, j], EV_VMIN, EV_VMAX))
                txt = f'{M[i, j]:.1e}'
                if metric == 'mean':
                    sf = cell(i, b, j)[1]
                    if np.isfinite(sf):
                        txt += f'\n{100 * sf:.0f}% sat'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7, color='white' if sh > 0.55 else 'black')
        ax.set_title(f'stochastic {b} Gramian', fontsize=11, pad=10)
    cax = fig.add_subplot(outer[3])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=EV_NORM,
                                            cmap=plt.get_cmap(EV_CMAP)), cax=cax)
    span = (f'the common TURN window ({nmin} steps = {nmin*0.01:.2f} s)'
            if turn_window else 'the trajectory')
    cb.set_label(f'{"BEST (minimum)" if metric == "best" else "MEAN"} min-EV(ζ) '
                 f'over {span}  [rad²]\nlower = better estimated   '
                 '(fixed 1e-4…1e6 scale on every figure)', fontsize=9)
    fig.suptitle(f'{"Best" if metric == "best" else "Average"} min-EV(ζ) — '
                 f'{title}\n{subtitle}\n'
                 f'window {W} steps ({W*0.01:g} s), λ = {LAM:g}, sensors φ, ψ, γ'
                 + (f' — scored ONLY over the fastest turn\'s {nmin} steps '
                    f'({nmin*0.01:.2f} s), centred on each row\'s turn midpoint'
                    if turn_window else '')
                 + sat_note, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84 if metric == 'mean' else 0.86))
    out = os.path.join(outdir, f'matrix_{"turnwindow_" if turn_window else ""}'
                               f'{metric}_w{W}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def wrap_about(z, centre=None):
    """ Wrap an angle into the branch [centre−π, centre+π). Plotting ζ on the usual
    (−π, π] branch would put the truth at ζ = −π exactly ON the discontinuity, so an
    estimate a hair above it jumps to the opposite end of the axis; centring the branch
    on ζ puts the truth mid-axis instead. """
    c = ZETA if centre is None else centre
    return c + wrap(np.asarray(z) - c)


def pi_ticks(lo=None, hi=None):
    """ π/2-spaced ticks with π-fraction labels across [lo, hi]. Defaults to one
    branch about ζ for callers that still want that. """
    if lo is None or hi is None:
        lo, hi = ZETA - np.pi, ZETA + np.pi
    k0 = int(np.floor(lo / (np.pi / 2)))
    k1 = int(np.ceil(hi / (np.pi / 2)))
    step = max(1, (k1 - k0) // 8)
    vals = [k * np.pi / 2 for k in range(k0, k1 + 1, step)]
    labs = []
    for v in vals:
        k = int(round(v / (np.pi / 2)))
        if k == 0:
            labs.append('0')
            continue
        num, den = abs(k), 2
        if num % 2 == 0:
            num, den = num // 2, 1
        core = 'π' if num == 1 else f'{num}π'
        labs.append(('−' if k < 0 else '') + core + ('' if den == 1 else '/2'))
    return vals, labs


def continuous(t, z):
    """ The angle on a CONTINUOUS branch: unwrapped sample to sample, then shifted
    by whole turns to sit nearest ζ so every row shares one frame.

    This replaces a branch-wrap + NaN-break pair. That drew each trace on the
    fixed branch [ζ−π, ζ+π) and cut the line wherever it crossed the seam, to
    avoid a spurious vertical jump — but the cuts read as arbitrary missing
    segments, which is worse than the problem they solved. Unwrapping removes
    both: no seam to cross, so no jump and nothing to break. The y-axis is then a
    real line rather than one branch, so a value below ζ−π is a genuine
    excursion, not a wrap. """
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    g = np.isfinite(z)
    if g.any():
        u = np.unwrap(z[g])
        out[g] = u + 2 * np.pi * np.round((ZETA - np.median(u)) / (2 * np.pi))
    return np.asarray(t, dtype=float), out


def fig_filter(rows, fname, title, subtitle, outdir, shift=False, variant=''):
    """ One figure per FILTER. Left: the path, arrowheads at the FIXED size used
    everywhere here, coloured by the constructability min-EV(ζ). Right: ζ̂ vs time,
    one colour per Q_ζ (unchanged from the earlier figures). A Q_ζ at which the filter
    diverged simply has no curve; the divergence is recorded in the CSV and the run
    log, not stamped on the figure.

    `variant` picks which pass of row_filters to draw, on identical axes and colours so
    the figures can be flipped between:
      ''        the figure of record — prior convention, honest P₀ (see row_filters)
      'noprior' diffuse P₀(ζ), guess as x̂₀|₀   → *_zeta_noprior.png
      'bad10'   BAD_STATES 10% wrong (baseline) → *_zeta_bad_initial_guesses_10.png
      'bad50'   BAD_STATES 50% wrong (stressed) → *_zeta_bad_initial_guesses_50.png
      'bad10i'  the same guess, P₀ inflated ×10 → *_bad_initial_guesses_10_inflated.png
      'bad50i'  the same guess, P₀ inflated ×10 → *_bad_initial_guesses_50_inflated.png
    Every non-default variant starts ON the guess, so the dotted line is drawn in every
    family for them, the shift ones included. """
    sfx = '' if variant == '' else '_np' if variant == 'noprior' else '_' + variant
    on_guess = variant != ''         # both extra variants start ON the guess
    n = len(rows)
    iq_ref = int(np.argmin(np.abs(Q_ZETA - Q_REF)))
    xlim, ylim = family_limits(rows)
    t_max = max(r['t'][-1] for r in rows)
    cmq = plt.get_cmap(Q_CMAP_FILT)
    fig = plt.figure(figsize=(16.4, 2.55 * n + 1.4))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.0, 2.9, 0.05], hspace=0.30,
                          wspace=0.16)
    for i, r in enumerate(rows):
        axL = fig.add_subplot(gs[i, 0])
        draw_path(axL, r, values=r['evp'][('constructability', iq_ref)])
        mark_ends(axL, r, ms_start=5, ms_end=8)
        axL.set_xlim(*xlim); axL.set_ylim(*ylim)
        axL.set_aspect('equal'); axL.grid(alpha=0.28)
        axL.set_ylabel(short_label(r), fontsize=9)
        axL.tick_params(labelsize=8)
        if i == n - 1:
            axL.set_xlabel('x [m]', fontsize=9)
        else:
            axL.set_xticklabels([])

        axR = fig.add_subplot(gs[i, 1])
        # ζ̂ drawn RAW on the branch centred on ζ (wrap_about is a pure branch
        # choice, not an unwrap): the y-axis is fixed to [ζ−π, ζ+π], so a circular
        # state is shown bounded, exactly as its wrapped value. ALL SIX Q_ζ are
        # drawn, coloured light→dark along the Q_CMAP_FILT ramp. In NOUPWIND mode
        # ζ = 0, so the axis is [−π, π] and the ±π seam is far from the action.
        for iq, qd in enumerate(Q_ZETA):
            axR.plot(r['t'], wrap_about(r['est' + sfx][(fname, qd)]), '-',
                     lw=1.7, alpha=0.95, color=cmq(iq / max(len(Q_ZETA) - 1, 1)))
        axR.plot(r['t'], wrap_about(r['zeta_true']), 'k-', lw=2.2, zorder=6)
        # the guess line is worth drawing in the shift families too once the curves
        # actually start on it, which only the no-prior convention makes true
        if on_guess or not shift:
            axR.axhline(wrap_about(ZETA + (ZETA_OFFSET_SHIFT if shift
                                           else ZETA_OFFSET)),
                        color='0.55', lw=1.0, ls=':')
        if shift:
            axR.axvline(SHIFT_STEP * 0.01, color='#c0392b', lw=1.5, ls='--')
        axR.axvspan(r['k_turn0'] * 0.01, (r['k_turn0'] + r['n_turn']) * 0.01,
                    color=r['color'], alpha=0.13, lw=0, zorder=0)
        # circular state, drawn bounded: one fixed branch [ζ−π, ζ+π]
        lo, hi = ZETA - np.pi, ZETA + np.pi
        axR.set_ylim(lo, hi)
        tv, tl = pi_ticks(lo, hi)
        axR.set_yticks(tv)
        axR.set_yticklabels(tl, fontsize=8)
        axR.set_xlim(0, t_max)
        axR.grid(alpha=0.3)
        axR.set_ylabel('ζ̂  [rad]', fontsize=9)
        axR.tick_params(labelsize=8)
        if i == n - 1:
            axR.set_xlabel('time [s]', fontsize=9)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=EV_NORM,
                                            cmap=plt.get_cmap(EV_CMAP)), cax=cax)
    cb.set_label(f'constructability min-EV(ζ) along the path  [rad²]  '
                 f'at Q_ζ = {Q_ZETA[iq_ref]:g}', fontsize=9)

    h = [Line2D([], [], color=cmq(iq / max(len(Q_ZETA) - 1, 1)), lw=2.5,
                label=f'Q_ζ = {qd:g}') for iq, qd in enumerate(Q_ZETA)]
    h += [Line2D([], [], color='k', lw=2.2, label='true ζ'),
          mpatches.Patch(color='0.5', alpha=0.2, label='the turn')]
    if shift:
        h += [Line2D([], [], color='#c0392b', ls='--', lw=1.6, label='wind shifts')]
    if on_guess or not shift:
        h += [Line2D([], [], color='0.55', ls=':', label='initial guess')]
    # below the axes, not above: the suptitle block here runs to four lines and a
    # top-anchored legend collides with the last of them
    fig.legend(handles=h, fontsize=9, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), framealpha=0.95)
    extra = ('\nUKF: Merwe scaled UT, α={alpha:g}, β={beta:g}, κ={kappa:g}, '
             'circular treatment of φ and ζ'.format(**UKF_KW) if fname == 'UKF' else
             '\nEKF: finite-difference Jacobians of the RK4 step and of h')
    if variant == 'noprior':
        extra += (f'\nNO PRIOR on ζ: σ₀(ζ) = π/√3 = {NOPRIOR_SIGMA_ZETA:.3f} rad '
                  '(uniform on the circle) instead of 1.048, and ζ̂(0) is x̂₀|₀ — the '
                  'k=0 correction is skipped and y₀ never consumed,\nso the curves '
                  'START on the dotted guess and the first update is the one at '
                  't = 0.01 s. The main figures keep the prior convention '
                  '(x̂₀|₋₁, y₀ consumed) that matches the Gramian recursion.')
    elif variant.startswith('bad'):
        pct, inf = variant[3:].rstrip('i'), variant.endswith('i')
        sg = rows[0][f'sig0_{variant}']
        rel = max(r[f'relerr_{variant}'] for r in rows)
        extra += (f'\nBAD INITIAL GUESSES at the {pct}% level — σ₀ = '
                  + ', '.join(f'{k} {v:.3g}' for k, v in sg.items())
                  + f' (worst realized {100*rel:.0f}% of |x₀|); '
                  'φ, ζ and the constants m, I, C_*, km1–4 left as usual.'
                  + (f'\nP₀ DELIBERATELY INFLATED ×{BAD_P0_INFLATE:g} in σ on those '
                     'states (×100 in variance): the guess is identical to the '
                     f'{pct}% figure, but the filter is told it is far less reliable '
                     'than it is,\nso it leans harder on the measurements. '
                     if inf else '\nP₀ honest, i.e. scaled to the draw. ')
                  + 'ζ̂(0) is x̂₀|₀, so the curves start on the dotted guess.')
    fig.suptitle(f'{fname} estimate of wind direction ζ — {title}\n{subtitle}\n'
                 f'sensors φ, ψ, γ, seed {SEED}; '
                 + ('the wind really steps +90° at t = '
                    f'{SHIFT_STEP*0.01:.2f} s and ζ̂(0) is set '
                    + ('CORRECT' if not ZETA_OFFSET_SHIFT else
                       f'{ZETA_OFFSET_SHIFT:g} rad wrong')
                    if shift else
                    f'ζ̂(0) is set {ZETA_OFFSET:g} rad wrong and the truth '
                    'holds ζ constant')
                 + ' — the SAME value in every row' + extra, fontsize=11.5)
    # headroom by how many lines the suptitle actually runs to: the no-prior block adds
    # four, the bad-guess block two, the figure of record none
    head = 0.34 if variant == 'noprior' else 0.26 if on_guess else 0.19
    fig.tight_layout(rect=(0, 0.05, 1, 0.99 - head / max(n, 1)))
    tag = ('' if variant == '' else '_noprior' if variant == 'noprior' else
           f'_bad_initial_guesses_{variant[3:].rstrip("i")}'
           + ('_inflated' if variant.endswith('i') else ''))
    out = os.path.join(outdir, f'{fname.lower()}_zeta{tag}.png')
    fig.savefig(out, dpi=135, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_filter_compare(rows, title, subtitle, outdir, shift=False, variant=''):
    """ One figure per family COMPARING the four filters (EKF, UKF, AI-KF, ANN)
    on shared axes. Rows are trajectories; the two columns are the extreme
    Q_ζ = 1e-5 (left) and Q_ζ = 1 (right), so a filter's Q-sensitivity reads off
    as the change between columns. The ANN takes measurements only and carries no
    Q, so its curve is identical in both columns by construction. `variant`
    selects the same row_filters pass as fig_filter, on identical ζ̂ axes. A Q_ζ
    at which a filter diverged simply has no curve (recorded in the CSV/log). """
    sfx = '' if variant == '' else '_np' if variant == 'noprior' else '_' + variant
    on_guess = variant != ''
    n = len(rows)
    qs = (Q_ZETA[0], Q_ZETA[-1])          # 1e-5, 1.0 — the PLOT_Q extremes
    fcol = {'EKF': '#1f77b4', 'UKF': '#ff7f0e', 'AIKF': '#2ca02c', 'ANN': '#d62728'}
    fname_lbl = lambda f: 'AI-KF' if f == 'AIKF' else f
    t_max = max(r['t'][-1] for r in rows)
    fig = plt.figure(figsize=(12.5, 2.5 * n + 1.3))
    gs = fig.add_gridspec(n, 2, hspace=0.30, wspace=0.10)
    for i, r in enumerate(rows):
        for j, qd in enumerate(qs):
            ax = fig.add_subplot(gs[i, j])
            for fname in FILTERS:
                ax.plot(r['t'], wrap_about(r['est' + sfx][(fname, qd)]), '-',
                        lw=1.6, alpha=0.9, color=fcol[fname])
            ax.plot(r['t'], wrap_about(r['zeta_true']), 'k-', lw=2.0, zorder=6)
            if on_guess or not shift:
                ax.axhline(wrap_about(ZETA + (ZETA_OFFSET_SHIFT if shift
                                              else ZETA_OFFSET)),
                           color='0.55', lw=1.0, ls=':')
            if shift:
                ax.axvline(SHIFT_STEP * 0.01, color='#c0392b', lw=1.5, ls='--')
            ax.axvspan(r['k_turn0'] * 0.01, (r['k_turn0'] + r['n_turn']) * 0.01,
                       color=r['color'], alpha=0.13, lw=0, zorder=0)
            lo, hi = ZETA - np.pi, ZETA + np.pi     # circular state, bounded branch
            ax.set_ylim(lo, hi)
            tv, tl = pi_ticks(lo, hi)
            ax.set_yticks(tv)
            ax.set_xlim(0, t_max)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(f'Q_ζ = {qd:g}', fontsize=10)
            if j == 0:
                ax.set_ylabel(short_label(r) + '\nζ̂ [rad]', fontsize=9)
                ax.set_yticklabels(tl, fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel('time [s]', fontsize=9)
            else:
                ax.set_xticklabels([])
    h = [Line2D([], [], color=fcol[f], lw=2.5, label=fname_lbl(f)) for f in FILTERS]
    h += [Line2D([], [], color='k', lw=2.2, label='true ζ'),
          mpatches.Patch(color='0.5', alpha=0.2, label='the turn')]
    if shift:
        h += [Line2D([], [], color='#c0392b', ls='--', lw=1.6, label='wind shifts')]
    if on_guess or not shift:
        h += [Line2D([], [], color='0.55', ls=':', label='initial guess')]
    fig.legend(handles=h, fontsize=9, ncol=len(h), loc='upper center',
               bbox_to_anchor=(0.5, 0.02), framealpha=0.95)
    vnote = ('\nno-prior pass: σ₀(ζ) = π/√3 (uniform on the circle), ζ̂(0) = x̂₀|₀'
             if variant == 'noprior' else
             '\nbad initial guess at the {}% level{}'.format(
                 variant[3:].rstrip('i'),
                 ', P₀ inflated ×{:g} in σ'.format(BAD_P0_INFLATE)
                 if variant.endswith('i') else '')
             if variant.startswith('bad') else '')
    fig.suptitle(f'filter comparison — estimate of wind direction ζ — {title}\n'
                 f'{subtitle}\nEKF, UKF, AI-KF (attached causal net), ANN; '
                 f'sensors φ, ψ, γ, seed {SEED}; columns are the extreme '
                 f'Q_ζ = {qs[0]:g} and {qs[1]:g} (ANN carries no Q, so it repeats)'
                 + vnote, fontsize=11)
    head = 0.28 if on_guess else 0.20
    fig.tight_layout(rect=(0, 0.05, 1, 0.99 - head / max(n, 1)))
    tag = ('' if variant == '' else '_noprior' if variant == 'noprior' else
           f'_bad_initial_guesses_{variant[3:].rstrip("i")}'
           + ('_inflated' if variant.endswith('i') else ''))
    out = os.path.join(outdir, f'filters_compare_zeta{tag}.png')
    fig.savefig(out, dpi=135, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────── data output ────────────────────────────────
def write_data(rows, key, datadir, shift=False):
    with open(os.path.join(datadir, 'trajectories.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['# family', key, 'system', SYSTEM, 'sensors', 'phi|psi|gamma',
                      'wind_m_per_s', WIND, 'x_end_m', X_END, 'window_steps', W,
                      'lambda', LAM])
        wtr.writerow(['label', 'group', 'amp_deg', 'omega_rad_per_s', 'v_m_per_s',
                      'radius_m', 'leg_m', 't_turn_s', 'T_total_s', 'N_steps',
                      'path_len_m', 'x_end_m', 'y_end_m', 'peak_phidot',
                      'hdg_err_deg_max',   # 0 now that R_TERM is small
                      # NB peak_phidot EXCEEDS the commanded omega: tracking a
                      # piecewise-linear heading ramp exactly requires rate overshoot at
                      # the ramp's two corners. omega is the MEAN rate over the turn,
                      # which is what sets the geometry.
                      'g_cross_m_per_s', 'dgamma_dzeta',
                      'path_dev_from_nominal_m', 'dev_phi_dot_rad_per_s',
                      'dev_phi_rad', 'dev_v_para_m_per_s'])
        for r in rows:
            wtr.writerow([r['label'], r['group'], f'{r["amp"]:g}',
                          f'{r["omega"]:.4f}', f'{r["v"]:.4f}', f'{r["radius"]:.6f}',
                          f'{r["leg"]:.6f}', f'{r["t_turn"]:.4f}', f'{r["T"]:.3f}',
                          r['N'], f'{r["path_len"]:.6f}', f'{r["end"][0]:.6f}',
                          f'{r["end"][1]:.6f}', f'{r["peak_phidot"]:.3f}',
                          f'{r["hdg_err_deg"]:.2f}', f'{r["g_cross"]:.5f}',
                          f'{dgamma_dzeta(r["g_cross"]):.4f}', f'{r["dev"]:.3e}',
                          f'{r["dev_phi_dot"]:.3e}', f'{r["dev_phi"]:.3e}',
                          f'{r["dev_v_para"]:.3e}'])

    with open(os.path.join(datadir, f'metrics_w{W}.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['# best/mean of min-EV(zeta) over the trajectory; family', key,
                      'window_steps', W, 'Q_zeta is the per-step covariance [rad^2]'])
        wtr.writerow(['label', 'group', 'amp_deg', 'omega_rad_per_s', 'v_m_per_s',
                      'basis', 'Q_zeta', 'best_minEV', 'mean_minEV',
                      'frac_windows_at_ceiling'])
        for r in rows:
            for b in BASES:
                for iq, qd in enumerate(Q_ZETA):
                    bst, mn, sf = r['metrics'][(b, iq)]
                    wtr.writerow([r['label'], r['group'], f'{r["amp"]:g}',
                                  f'{r["omega"]:.4f}', f'{r["v"]:.4f}', b, f'{qd:g}',
                                  f'{bst:.6e}', f'{mn:.6e}', f'{sf:.4f}'])

    with open(os.path.join(datadir, f'minev_timeseries_w{W}.csv'), 'w',
              newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['# sliding min-EV(zeta), reported at the window CENTRE; family',
                      key])
        wtr.writerow(['label', 'basis', 'Q_zeta', 'time_s', 'time_initial_s',
                      'minEV_zeta'])
        for r in rows:
            for b in BASES:
                for iq, qd in enumerate(Q_ZETA):
                    df = r['ev'][(b, iq)]
                    for tc, t0, v in zip(df['time'].values,
                                         df['time_initial'].values,
                                         df['zeta'].values):
                        wtr.writerow([r['label'], b, f'{qd:g}', f'{tc:.3f}',
                                      f'{t0:.3f}', f'{v:.6e}'])

    # the two extra passes, each in its own file so filter_zeta_error.csv keeps its
    # schema. t_to_0.1rad is the first time the wrapped error falls below 0.1 rad and
    # STAYS there ('never' therefore also covers "got there and drifted back out") —
    # under these conventions the estimate starts ON the guess, so "how long until the
    # trajectory has told the filter where the wind is" is finally a question the curve
    # can answer.
    extra_csvs = [('_np', 'filter_zeta_error_noprior.csv',
                   ['# NO-PRIOR pass: sigma0_zeta_rad', f'{NOPRIOR_SIGMA_ZETA:.6f}',
                    '(uniform on the circle)'])]
    for tag, sc in BAD_X0_SCALES:
        for isfx in ('', 'i'):
            v = f'bad{tag}{isfx}'
            extra_csvs.append(
                (f'_{v}', f'filter_zeta_error_bad_initial_guesses_{tag}'
                          + ('_inflated' if isfx else '') + '.csv',
                 [f'# BAD-GUESS pass at {tag}%: draw sigma =', f'{sc:g}',
                  'x 0.1 x max(|x0|,0.1) on', '|'.join(BAD_STATES),
                  'constants m|I|C_para|C_perp|C_phi|km1-4 and phi and zeta LEFT ALONE',
                  'P0', f'inflated x{BAD_P0_INFLATE:g} in sigma' if isfx else 'honest',
                  'P0_per_state_sigma', ';'.join(f'{k}={x:.4g}' for k, x
                                                 in rows[0][f'sig0_{v}'].items())]))
    for sfx, fn_csv, hdr in extra_csvs:
        with open(os.path.join(datadir, fn_csv), 'w', newline='') as f:
            wtr = csv.writer(f)
            wtr.writerow(hdr + ['x0 convention', 'posterior', '(y_0 not consumed)',
                                'family', key, 'seed', SEED, 'init_offset_rad',
                                ZETA_OFFSET_SHIFT if shift else ZETA_OFFSET])
            wtr.writerow(['label', 'group', 'filter', 'Q_zeta',
                          'zeta_hat_initial_rad', 'worst_rel_err_other_states',
                          'err_after_1_update_rad', 't_to_0.1rad_s',
                          'final_abs_err_rad', 'rms_err_2nd_half_rad'])
            for r in rows:
                rel = (f'{r["relerr" + sfx]:.4f}' if sfx.startswith('_bad')
                       else '0.1 (default)')
                for fname in FILTERS:
                    for qd in Q_ZETA:
                        z0 = r['zhat0']
                        if qd in r['diverged' + sfx][fname]:
                            wtr.writerow([r['label'], r['group'], fname, f'{qd:g}',
                                          f'{z0:.9f}', rel] + ['diverged'] * 4)
                            continue
                        e = np.abs(wrap(r['est' + sfx][(fname, qd)] - r['zeta_true']))
                        bad = np.flatnonzero(e >= 0.1)
                        k1 = (bad[-1] + 1) if len(bad) else 0
                        tc = f'{k1 * 0.01:.2f}' if k1 < len(e) else 'never'
                        wtr.writerow([r['label'], r['group'], fname, f'{qd:g}',
                                      f'{z0:.9f}', rel, f'{e[1]:.6e}', tc,
                                      f'{e[-1]:.6e}',
                                      f'{np.sqrt(np.mean(e[len(e)//2:]**2)):.6e}'])

    with open(os.path.join(datadir, 'filter_zeta_error.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['# EKF/UKF wrapped |zeta_hat - zeta_true|; family', key,
                      'seed', SEED,
                      'init_offset_rad', 0.0 if shift else ZETA_OFFSET,
                      'truth', 'zeta steps +90deg at step 25' if shift
                      else 'zeta constant'])
        wtr.writerow(['label', 'group', 'filter', 'Q_zeta', 'zeta_hat_initial_rad',
                      'final_abs_err_rad', 'rms_err_2nd_half_rad',
                      '# "diverged" = the filter blew up and no estimate exists'])
        for r in rows:
            for fname in FILTERS:
                for qd in Q_ZETA:
                    z0 = r['zhat0']
                    if qd in r['diverged'][fname]:
                        wtr.writerow([r['label'], r['group'], fname, f'{qd:g}',
                                      f'{z0:.9f}', 'diverged', 'diverged'])
                        continue
                    e = np.abs(wrap(r['est'][(fname, qd)] - r['zeta_true']))
                    wtr.writerow([r['label'], r['group'], fname, f'{qd:g}',
                                  f'{z0:.9f}', f'{e[-1]:.6e}',
                                  f'{np.sqrt(np.mean(e[len(e)//2:]**2)):.6e}'])

    np.savez(os.path.join(datadir, 'results.npz'),
             family=key, q_zeta=Q_ZETA, window=W, lam=LAM, wind=WIND,
             labels=np.array([r['label'] for r in rows]),
             amp_deg=np.array([r['amp'] for r in rows]),
             omega=np.array([r['omega'] for r in rows]),
             speed=np.array([r['v'] for r in rows]),
             **{f'{b}|{m}': np.array([[r['metrics'][(b, iq)][mi]
                                       for iq in range(len(Q_ZETA))]
                                      for r in rows])
                for b in BASES
                for mi, m in enumerate(('best', 'mean', 'satfrac'))},
             **{f'X|{r["label"]}': r['X'][:, :2] for r in rows})




# ────────────────────────────────── driver ──────────────────────────────────
FAMILIES = (
    ('a_amplitude', spec_a,
     'FAMILY (a) — turn AMPLITUDE, small → large',
     f'control rule: hold ω = {A_OMEGA:g} rad/s and v = {V_CRUISE:g} m/s, sweep the '
     f'net turn Δ = {", ".join(f"{a:.0f}" for a in A_AMPS)}° (the turn is simply '
     f'held longer).\ngoal: does a BIGGER turn make the wind direction more '
     f'estimable, and how does that change with the assumed wind volatility Q_ζ?\n'
     f'every row starts at (0,0), ends at ({X_END:g},0) and turns the same way; the '
     f'straight legs absorb Δ, so the total DURATION differs by design.'),
    ('b_turn_rate', spec_b,
     'FAMILY (b) — turn RATE, slow/spread → fast saccade',
     f'control rule: hold Δ = {B_AMP:.0f}° and v = {V_CRUISE:g} m/s, sweep the turn '
     f'rate ω = {", ".join(f"{o:g}" for o in B_OMEGAS)} rad/s.\n'
     f'goal: at a FIXED net heading change, does concentrating the turn into a brief '
     f'fast saccade beat spreading it out slowly — and does the answer flip with '
     f'Q_ζ?\nevery row starts at (0,0), ends at ({X_END:g},0) and turns the same '
     f'way; only the turn rate (hence radius) differs.'),
    ('c_speed', spec_c,
     f'FAMILY (c) — overall SPEED, at {len(C_AMPS)} different turn angles',
     f'control rule: fix the PATH (turn radius {C_RADIUS*1e3:.0f} mm) at each of '
     f'Δ = {", ".join(f"{a:.0f}" for a in C_AMPS)}°, and fly it at '
     f'v = {", ".join(f"{v:.3f}" for v in C_SPEEDS)} m/s; ω = v/R and T = L/v follow.'
     f'\ngoal: separate "sharper corner" from "faster flight" — within a panel every '
     f'speed traces the IDENTICAL ground track, so any change is speed alone.\n'
     f'heading INTO the wind, ∂γ/∂ζ = +w/(v+w) has no pole: it falls smoothly from '
     f'{dgamma_dzeta(C_SPEEDS[0]):+.2f} to {dgamma_dzeta(C_SPEEDS[-1]):+.2f} across '
     f'the family. a fast row is also observed for fewer samples.'),
    ('d_wind_shift', spec_d,
     'FAMILY (d) — turn RATE, with a wind direction that really moves',
     f'control rule: family (b) exactly (Δ = {B_AMP:.0f}°, v = {V_CRUISE:g} m/s, '
     f'ω = {", ".join(f"{o:g}" for o in B_OMEGAS)} rad/s), but the true wind '
     f'direction STEPS +90° (from {np.degrees(ZETA):.0f}° to '
     f'{np.degrees(ZETA+D_SHIFT_RAD):.0f}°) at time step {D_SHIFT_STEP} '
     f'(t = {D_SHIFT_STEP*0.01:.2f} s).\n'
     f'goal: when the wind actually changes, which turn tempo recovers the new '
     f'direction fastest?  the MPC RE-PLANS from the step onward, so the ground track '
     f'is held to ~0.15 mm,\nbut it is a closed-loop recovery: φ̇ wobbles at the '
     f'shift — 0.47 rad/s in the ω=2.5 row, where step {D_SHIFT_STEP} falls MID-TURN, '
     f'down to 3e-5 in the fast rows,\nwhose longer straight lead-in leaves the MPC '
     f'0.14–0.19 s to settle before the turn begins.  family (e) removes that '
     f'transient entirely.'),
    ('e_invariant_course_wind_shift', spec_e,
     'FAMILY (e) — the same wind shift, with a trajectory that does not change at all',
     f'control rule: family (d)\'s wind shift (+90° at step {D_SHIFT_STEP}), but the '
     f'fly re-aims its force output to cancel the wind\'s changed drag FEED-FORWARD, '
     f'so the motion is invariant.\ngoal: separate the pure INFORMATIONAL effect of '
     f'the wind change from the motion transient (d) still contains — here nothing '
     f'about the trajectory can contribute.  NB this changes (d) by only 1–3.5%, and '
     f'mostly in the ω=2.5 row;\nit is a control that confirms (d) was not '
     f'contaminated, not a different answer.\nζ reaches the dynamics only through the '
     f'airflow terms w·cos(φ−ζ), w·sin(φ−ζ), which enter alongside the controls, so '
     f'u has full authority over exactly what ζ perturbs;\nthe control that reproduces '
     f'the nominal next state is solved per step (Newton, 3 unknowns). Achieved: '
     f'v_para, v_perp, φ̇ matched to 1e-16, φ to ~2e-4 rad, position to ~40 µm.'),
    ('f_late_wind_shift', spec_f,
     'FAMILY (f) — the same wind shift, arriving LATE (t = 0.60 s)',
     f'control rule: family (d)\'s trajectories and method exactly, but the wind holds '
     f'ζ = {np.degrees(ZETA):.0f}° until t = {F_SHIFT_STEP*0.01:.2f} s '
     f'(step {F_SHIFT_STEP}) and only then steps +90° to {np.degrees(ZETA+D_SHIFT_RAD):.0f}°.\ngoal: in (d) the shift lands BEFORE the turn in every row but the '
     f'slowest; at t = {F_SHIFT_STEP*0.01:.2f} s it lands DURING or just AFTER it '
     f'instead — so this is the complementary case,\nand only 21–36 steps remain '
     f'afterwards for the estimate to recover. Same two-stage MPC re-plan as (d), so '
     f'the ground track is unchanged.'),
)


def process_family(idx):
    """ Build and plot ONE family end to end, in its own process. Families are
    fully independent — separate output subdir, own engine — so running them in
    parallel just needs each worker to rebuild the engine (the ANN attaches
    lazily in row_filters). """
    global SHIFT_STEP
    key, spec_fn, title, subtitle = FAMILIES[idx]
    spec = SYSTEMS[SYSTEM]()
    spec.build_mpc = build_mpc_tracking          # local override only; see R_TERM
    engine = ObservabilityEngine(spec)
    shift = key in ('d_wind_shift', 'e_invariant_course_wind_shift',
                    'f_late_wind_shift')
    hold = (key == 'e_invariant_course_wind_shift')
    SHIFT_STEP = F_SHIFT_STEP if key == 'f_late_wind_shift' else D_SHIFT_STEP
    fam = os.path.join(OUTDIR, key)
    plots, data = os.path.join(fam, 'plots'), os.path.join(fam, 'data')
    os.makedirs(plots, exist_ok=True)
    os.makedirs(data, exist_ok=True)
    rows = build_family(engine, spec, key, spec_fn(),
                        shift=((SHIFT_STEP, D_SHIFT_RAD) if shift else None),
                        offset=(ZETA_OFFSET_SHIFT if shift else ZETA_OFFSET),
                        hold=hold)
    if key == 'c_speed':
        fig_family(rows, title, subtitle, plots, leg_title='overall speed',
                   note='the four speeds of one turn angle trace the SAME path by '
                        'construction, so they overplot;\nthey are separated in '
                        'family_by_speed.png, where the arrowhead spacing '
                        '(= samples per metre) differs')
        fig_family_grid(rows, title, subtitle, plots)
    else:
        fig_family(rows, title, subtitle, plots, shift=shift,
                   leg_title=('turn amplitude' if key == 'a_amplitude'
                              else 'turn rate'))
    for r in rows:
        fig_traj(r, rows, title, plots, shift=shift)
    for m in ('best', 'mean'):
        fig_matrix(rows, m, title, subtitle.split('\n')[0], plots)
        # the turn-window matrices are only meaningful where the turn DURATION
        # varies across the family, i.e. the ω sweeps of (d)–(f)
        if shift:
            fig_matrix(rows, m, title, subtitle.split('\n')[0], plots,
                       turn_window=True)
    variants = (('', 'noprior')
                + tuple(f'bad{t}{i}' for t, _ in BAD_X0_SCALES
                        for i in ('', 'i')))
    for fname in FILTERS:
        for var in variants:
            fig_filter(rows, fname, title, subtitle.split('\n')[0], plots,
                       shift=shift, variant=var)
    # cross-filter comparison (EKF/UKF/AI-KF/ANN) at Q_ζ = 1e-5 and 1, per pass
    #for var in variants:
        #fig_filter_compare(rows, title, subtitle.split('\n')[0], plots,
        #                   shift=shift, variant=var)
    #write_data(rows, key, data, shift=shift)
    fam_lines = []
    print(f'  → {plots}', flush=True)
    return key, fam_lines


def main():
    spec = SYSTEMS[SYSTEM]()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'system {SYSTEM}: {len(spec.state_names)} states, sensors '
          f'{tuple(spec.fim_sensors)} (angular only), reporting min-EV of zeta\n'
          f'wind {WIND:g} m/s at ζ = {np.degrees(ZETA):.0f}°, '
          f'window {W} steps ({W*0.01:g} s), λ = {LAM:g}\n'
          f'Q_ζ (per-step) {list(Q_ZETA)} rad²; ζ̂ line plots draw Q_ζ = '
          f'{", ".join(f"{q:g}" for q in PLOT_Q)}; trajectory panels coloured at '
          f'Q_ζ = {Q_REF:g}\nMPC input-rate penalty R_TERM = {R_TERM:g} '
          f'(local override; the spec default 1e-6 makes tracking fidelity depend on ω)'
          f'\noutputs → {OUTDIR}   (N_JOBS={N_JOBS})', flush=True)

    idxs = range(len(FAMILIES))
    if N_JOBS > 1:
        import multiprocessing as mp
        # keep each worker single-threaded so the families run truly in parallel
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[v] = '1'
        nj = min(N_JOBS, len(FAMILIES))
        print(f'  running {len(FAMILIES)} families across {nj} workers', flush=True)
        with mp.get_context('spawn').Pool(nj) as pool:
            results = pool.map(process_family, idxs)
    else:
        results = [process_family(i) for i in idxs]



if __name__ == '__main__':
    main()
