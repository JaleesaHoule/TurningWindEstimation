# TurningWindEstimation

Investigating the relationship between **observability / constructability** and
**estimation** for systems with process noise — specifically, estimating wind
direction ζ (and related states) for a fly / drone from angular sensors, and how
turning maneuvers make ζ observable.

The analyses compare model-based bounds (stochastic observability /
constructability Gramians, per-timestep Cramér–Rao bounds) against real
estimators: an EKF, a UKF, an AI-KF, and feed-forward ζ-ANNs.

## Layout

```
twe/                 core library package (import as `twe.*`)
analysis_scripts/    runnable analyses (each writes an output folder at the repo root)
models/              vendored pre-trained networks
zeta_ann_causal/     the causal ζ-ANN model (network.h5)
pyproject.toml       package + dependencies
setup.sh             one-shot uv environment setup
```

The only external project dependency is
[`pybounds`](https://github.com/vanbreugel-lab/pybounds) (on PyPI); everything
else is either a standard Python package or vendored here in `twe/`.

## Setup (uv)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it for you).

```bash
./setup.sh                 # == uv sync : creates .venv (Py 3.11) + installs all deps editable
```

or manually:

```bash
uv sync                    # create .venv and install this package + dependencies
```

Then run any analysis (uv activates the environment automatically):

```bash
uv run python analysis_scripts/ann_papernet_constant_minev.py
uv run python analysis_scripts/turn_angle_analysis.py
```

Each script creates its output folder **at the repo root** (e.g.
`ann_observability_probe_output/`); outputs are regenerable and git-ignored.
The `twe/observability_gui.py` notebook GUI is optional —
`uv sync --extra gui` adds `ipywidgets`/`ipython`; the analysis scripts don't
need it.

## Analyses → scripts

| Output folder | Scripts (`analysis_scripts/`) | What it produces |
|---|---|---|
| `turn_control_sweep_noupwind_output/` | `turn_control_sweep.py`, `q0_estimator_comparison.py` | Turn-rate / turn-amplitude sweeps: EKF/UKF/AI-KF/ANN ζ estimates vs. the constructability min-EV, off the upwind axis. |
| `zeta_ann_causal/` | `train_zeta_ann.py`, `plot_zeta_ann_training_examples.py` | Trains the causal ζ-ANN (`network.h5`) and visualizes the observability-curated training set. |
| `ann_observability_probe_output/` | `ann_observability_probe.py`, `ann_observability_probe_w.py`, `ann_timevarying_zeta.py`, `ann_papernet_fwd_bwd.py`, `ann_papernet_constant_minev.py`, `ann_mynet_constant.py` | Probes of *what the ANN actually estimates*: does it track ζ on "unobservable" segments (conditional vs. marginal), dependence on wind speed, time-varying wind, forward vs. backward windows, and the paper's pre-trained net vs. ours. |
| `turn_angle_analysis/` | `turn_angle_analysis.py`, `angular_velocity_sweep.py` | Wind-direction-independent maps of ζ observability vs. turn angle and angular velocity. |

## Core library (`twe/`)

- `observability_gui.py` — the compute engine: MPC trajectory simulation +
  stochastic/empirical observability + EKF/UKF/AI-KF (`ObservabilityEngine`,
  `SYSTEMS`, `build_setpoints`). The ipywidgets/IPython GUI is optional.
- `drone_model.py` — 3D kinematic drone model (the paper's `kinematic_3d`).
- `stochastic_observability.py` / `stochastic_observability_gramian.py` —
  linearized and nonlinear stochastic observability/constructability Gramians.
- `ann_estimator.py` — the AI-KF's observability→measurement-noise (`motif_R`) map.
- `zeta_ann_estimator.py` — `CausalZetaANN`, loads `zeta_ann_causal/network.h5`.
- `single_turn_tempo.py`, `tempo_traj_ev_plots.py`, `tempo_filter_sweep.py` —
  turn-tempo trajectory / EV / plotting helpers used by `turn_control_sweep.py`.

## Models

- `zeta_ann_causal/network.h5` — the causal ζ-ANN trained by `train_zeta_ann.py`
  (inputs φ, ψ, γ; backward 4-step window).
- `models/papernet.keras` — the paper's pre-trained drone ζ-network
  (percentile=70, inputs psi/beta/gamma = heading/course/airflow, forward 4-step
  window), from *Discovering and exploiting active sensing motifs for
  estimation*. **Note:** it is an HDF5 file despite the `.keras` extension;
  `ann_papernet_fwd_bwd.load_net()` copies it to `.h5` before loading.
