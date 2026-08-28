"""Centralized filesystem locations for TurningWindEstimation.

Scripts import these instead of recomputing repo-relative paths, so the layout
(models under ANNmodels/, all figures under analysis_output/) lives in one place.
The repo root is the parent of this package directory.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- pre-trained networks ---------------------------------------------------
ANN_MODELS = os.path.join(REPO_ROOT, 'ANNmodels')
CAUSAL_NET = os.path.join(ANN_MODELS, 'zeta_ann_causal', 'network.h5')   # ours
PAPER_NET = os.path.join(ANN_MODELS, 'pybounds_paper', 'papernet.keras')  # the paper's

# --- analysis outputs -------------------------------------------------------
ANALYSIS_OUTPUT = os.path.join(REPO_ROOT, 'analysis_output')


def output_dir(name):
    """Return (creating it) ``REPO_ROOT/analysis_output/<name>``."""
    d = os.path.join(ANALYSIS_OUTPUT, name)
    os.makedirs(d, exist_ok=True)
    return d
