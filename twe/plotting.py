"""Reusable plotting helpers for the trajectory / observability figures.

Currently the paper-style heading-arrowhead markers used to draw a flown x-y path
coloured by a per-point scalar (e.g. min error variance of zeta). Vendored from
``observability-gui/render.py`` (a simplification of fly_plot_lib's
get_wedges_for_heading_plot, which is not in the installed pybounds).
"""
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def heading_of(engine, spec):
    """ Orientation for the arrowhead markers. Prefer yaw ``psi`` when the model
    has both psi and phi (in a 3D model phi is ROLL); otherwise phi, which IS the
    heading in the planar fly models; failing that, the direction of travel. """
    names = list(spec.state_names)
    order = ('psi', 'phi') if ('psi' in names and 'phi' in names) else ('phi', 'psi')
    for nm in order:
        if nm in names:
            return engine.X[:, names.index(nm)]
    X = engine.X                                   # no heading state: use travel dir
    return np.arctan2(np.gradient(X[:, 1]), np.gradient(X[:, 0]))


def heading_wedges(x, y, color, orientation_rad, norm, cmap, size_radius,
                   size_angle=20.0, nskip=0, center_offset_fraction=0.75):
    """ Paper-style heading arrowheads: a PatchCollection of coloured wedges
    pointing along ``orientation_rad``, coloured by ``color`` through ``norm`` /
    ``cmap``. """
    idx = np.arange(0, len(x), nskip + 1)
    orient = np.asarray(orientation_rad, dtype=float) * 180.0 / np.pi + 180.0
    wedges = []
    for i in idx:
        cx = x[i] - np.cos(np.radians(orient[i])) * size_radius * center_offset_fraction
        cy = y[i] - np.sin(np.radians(orient[i])) * size_radius * center_offset_fraction
        wedges.append(mpatches.Wedge((cx, cy), size_radius,
                                     orient[i] - size_angle / 2.0,
                                     orient[i] + size_angle / 2.0))
    pc = PatchCollection(wedges, cmap=cmap, norm=norm)
    pc.set_edgecolors('none')
    pc.set_linewidth(0.25)
    pc.set_array(np.asarray(color)[idx])
    return pc
