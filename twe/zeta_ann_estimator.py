#!/opt/anaconda3/envs/Python311/bin/python
"""
The causal ζ-ANN of train_zeta_ann.py, wrapped in the interface the engine's
AI-KF expects (`predict_series`, plus the attributes its cache reads), so it can
be dropped into ObservabilityEngine._ann_cache and used by run_aikf.

CAUSAL, unlike the networks shipped in the authors' repo: the window for step k
is [k-omega+1 .. k], most recent sample FIRST, matching what
`augment_time_history(direction='backward')` emits and what their own Methods
describe ("omega prior time steps"). The first omega-1 steps of a run have no
history and come back NaN; run_aikf drops the ANN channel on those steps.

The three inputs are the fly's angular measurements, in fly_h's column order:
    phi    heading                 (the paper's psi)
    psi    optic flow angle        (the paper's beta)
    gamma  apparent airflow angle  (the paper's gamma)
"""

import os

import numpy as np

from .paths import CAUSAL_NET
DEFAULT_NET = CAUSAL_NET                          # ANNmodels/zeta_ann_causal/network.h5
SENSORS = ('phi', 'psi', 'gamma')


class CausalZetaANN:

    def __init__(self, spec, path=DEFAULT_NET, time_steps=None, layers=(64, 64, 64)):
        import keras
        self.model = keras.models.load_model(path, compile=False)
        n_in = int(self.model.input_shape[-1])
        inferred = n_in // len(SENSORS)
        if time_steps is None:
            time_steps = inferred
        if time_steps * len(SENSORS) != n_in:
            raise ValueError(
                'network expects %d inputs, which is not %d sensors x %d steps'
                % (n_in, len(SENSORS), time_steps))
        self.time_steps = int(time_steps)
        self.layers = tuple(layers)
        self.spec = spec
        self.target, self.circular = 'zeta', True
        self.sensors = list(SENSORS)
        self.sidx = [spec.measurement_names.index(m) for m in SENSORS]

    def predict_series(self, Y, U=None):
        """ One ζ estimate per step, on the CAUSAL window ending at that step.
        Steps without a full history return NaN. """
        A = np.asarray(Y, dtype=float)[:, self.sidx]
        n, T = len(A), self.time_steps
        out = np.full(n, np.nan)
        if n < T:
            return out
        # row k (k >= T-1): [phi_k, phi_{k-1}, .., phi_{k-T+1}, psi.., gamma..]
        cols = [A[T - 1 - j: n - j, s]
                for s in range(A.shape[1]) for j in range(T)]
        F = np.stack(cols, axis=-1)
        p = np.asarray(self.model.predict(F, verbose=0)).ravel()
        out[T - 1:] = np.arctan2(np.sin(p), np.cos(p))
        return out


def attach(engine, spec, path=DEFAULT_NET, seed=0, **cache_kw):
    """ Put the net where run_aikf will find it, and return (ann, kwargs) with
    the run_aikf arguments that match it. Delegates the key construction to
    ``engine.attach_ann`` so this wrapper cannot drift from the engine's own
    cache-key layout. """
    ann = CausalZetaANN(spec, path)
    kw = engine.attach_ann(ann, seed=seed,
                           n_traj=cache_kw.get('n_traj', 16),
                           epochs=cache_kw.get('epochs', 100),
                           batch=cache_kw.get('batch', 256),
                           noise_std=cache_kw.get('noise_std', 0.0))
    return ann, kw
