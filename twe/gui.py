"""Optional ipywidgets/IPython notebook GUI for the observability engine.

Requires the ``gui`` extra:  uv sync --extra gui
Usage (in a notebook):
    from twe.gui import ObservabilityGUI
    ObservabilityGUI().display()          # system='fly' by default

Split out of observability_gui.py so the compute engine has no ipywidgets/IPython
dependency; everything analytical lives in observability_gui.py.
"""
import io
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import ipywidgets as W
from IPython.display import display

from .observability import (SYSTEMS, ObservabilityEngine, SystemSpec,
                                build_setpoints)

MOTIF_LABELS = {
    'straight': lambda s: f"straight {s['duration']:g}s",
    'turn': lambda s: f"turn {s['angle']:g} rad / {s['duration']:g}s",
    'circle': lambda s: f"circle {s['angle']:g} rad / {s['duration']:g}s",
    'speed': lambda s: f"speed → {s['speed']:g} m/s / {s['duration']:g}s",
    'weave': lambda s: f"weave ±{s['amp']:g} rad @ {s['freq']:g}Hz / {s['duration']:g}s",
}

class _LiveFig:
    """ A matplotlib panel rendered into an ipywidgets Output area.

    By default (``interactive=False``) it renders each update to a static PNG
    Image widget — this is deliberately NON-interactive: the ipympl live canvas
    streams every redraw to the browser and made the in-notebook GUI sluggish,
    so the main panels use plain static images (for full zoom/pan
    interactivity run the desktop app `observability_app.py`). Pass
    ``interactive=True`` to opt back into the ipympl zoom/pan canvas. """

    def __init__(self, figsize, interactive=False):
        self._figsize = tuple(figsize)
        if interactive:
            try:
                from matplotlib.figure import Figure as _F
                from ipympl.backend_nbagg import (Canvas as _C,
                                                  FigureManager as _FM)
                self.fig = _F(figsize=figsize)
                self.canvas = _C(self.fig)
                self.manager = _FM(self.canvas, 0)
                self.canvas.header_visible = False
                self.canvas.footer_visible = False
                self.canvas.toolbar_position = 'right'
                self.canvas.capture_scroll = False
                self.widget = self.canvas
                self.live = True
                return
            except Exception:
                pass
        self.fig = None
        self.widget = W.Image(format='png',
                              layout=W.Layout(width='100%',
                                              max_width='1150px'))
        self.live = False

    def draw(self, render_fn, figsize=None):
        if self.live:
            if figsize is not None and tuple(figsize) != self._figsize:
                self._figsize = tuple(figsize)
                self.fig.set_size_inches(*figsize, forward=True)
            self.fig.clf()
            render_fn(self.fig)
            self.canvas.draw_idle()
        else:
            fig = plt.figure(figsize=figsize or self._figsize)
            render_fn(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=105, bbox_inches='tight')
            plt.close(fig)
            self.widget.value = buf.getvalue()


class ObservabilityGUI:
    def __init__(self, system='fly', dt=None):
        """ :param dt: optional time-step override [s]; None uses the
        system's default (fly/fly7: 0.01, drone: 0.1). """
        self._dt_override = dt
        self.spec = self._make_spec(system)
        self.engine = ObservabilityEngine(self.spec)
        self.segments = []            # empty -> system's default trajectory
        self._suspend = False

        self._build_widgets(system)
        self._wire_callbacks()
        self._apply_system_defaults()
        self._rebuild_trajectory()    # initial simulate + compute + draw

    # -- widget construction ---------------------------------------------------
    def _build_widgets(self, system):
        L = dict(style={'description_width': '52px'})
        self.w_system = W.Dropdown(options=[('fly (pybounds example, 18 states)', 'fly'),
                                            ('fly (7-state, calibrated)', 'fly7'),
                                            ('drone (kinematic 3D)', 'drone'),
                                            ('altitude 2D (paper Fig. 4b)', 'alt2d')],
                                   value=system, description='system', **L)
        # trajectory builder
        self.w_motif = W.Dropdown(options=['straight', 'turn', 'circle',
                                           'speed', 'weave'],
                                  value='turn', description='motif', **L)
        self.w_duration = W.FloatText(value=0.1, description='dur [s]', **L)
        self.w_angle = W.FloatText(value=-np.pi / 2, description='angle rad', **L)
        self.w_speed = W.FloatText(value=0.4, description='v [m/s]', **L)
        self.w_freq = W.FloatText(value=2.0, description='f [Hz]', **L)
        self.w_amp = W.FloatText(value=np.pi / 6, description='amp rad', **L)
        self.b_add = W.Button(description='+ add segment', button_style='primary',
                              layout=W.Layout(width='120px'))
        self.b_undo = W.Button(description='undo', layout=W.Layout(width='70px'))
        self.b_clear = W.Button(description='clear (→ default traj)',
                                layout=W.Layout(width='150px'))
        self.b_left = W.Button(description='↰ +π/2 left', layout=W.Layout(width='95px'))
        self.b_right = W.Button(description='↱ −π/2 right', layout=W.Layout(width='95px'))
        self.b_fwd = W.Button(description='↑ straight', layout=W.Layout(width='95px'))
        self.b_faster = W.Button(description='+Δv', layout=W.Layout(width='60px'))
        self.b_slower = W.Button(description='−Δv', layout=W.Layout(width='60px'))
        self.html_segments = W.HTML()
        self.w_v0 = W.FloatText(value=0.3, description='v₀', **L)
        self.w_wind = W.FloatText(value=0.4, description='wind w', **L)
        self.w_zeta = W.FloatText(value=np.pi, description='ζ [rad]', **L)
        self.b_sim = W.Button(description='▶ simulate (MPC)',
                              button_style='success', layout=W.Layout(width='160px'))

        # observability params — λ range floor (1e-8) stays above the
        # round-off noise floor eps_mach * λ_max(F) for these systems
        self.w_win = W.IntSlider(value=4, min=3, max=60, description='window w',
                                 continuous_update=False, **L)
        self.w_eps = W.FloatLogSlider(value=1e-4, base=10, min=-6, max=-2, step=0.5,
                                      description='ε', continuous_update=False,
                                      readout_format='.1e', **L)
        self.w_lam = W.FloatLogSlider(value=1e-6, base=10, min=-8, max=-3, step=1,
                                      description='λ', continuous_update=False,
                                      readout_format='.0e', **L)
        self.w_color_state = W.Dropdown(options=list(self.spec.state_names),
                                        value=self.spec.color_state_default,
                                        description='color by', **L)
        self.w_ev_states = W.SelectMultiple(options=list(self.spec.state_names),
                                            value=self.spec.ev_states_default,
                                            rows=6, description='EV states', **L)
        self.w_est_states = W.SelectMultiple(options=list(self.spec.state_names),
                                             value=self.spec.ev_states_default,
                                             rows=6, description='est. states',
                                             **L)
        self.w_meas = W.SelectMultiple(options=list(self.spec.measurement_names),
                                       value=self._meas_default(self.spec),
                                       rows=6, description='sensors', **L)
        self.w_fim_sensors = W.SelectMultiple(
            options=list(self.spec.measurement_names),
            value=tuple(self.spec.fim_sensors), rows=6,
            description='FIM sens.', **L)
        self.w_fim_states = W.SelectMultiple(
            options=list(self.spec.state_names),
            value=tuple(self.spec.fim_states), rows=8,
            description='FIM states', **L)
        self.w_empirical = W.Checkbox(value=True, indent=False,
                                      description='pybounds empirical (slower, cached)')
        self.w_mc = W.Checkbox(value=False, indent=False,
                               description='nonlinear MC Gramian (slow, cached)')
        self.w_nmc = W.BoundedIntText(value=100, min=10, max=2000, step=10,
                                      description='MC traj.', **L)
        self.w_init = W.BoundedFloatText(value=1.0, min=0.0, max=100.0,
                                         step=0.5, description='init err ×',
                                         tooltip='scales the EKF/UKF initial '
                                         'estimate error and P0 (filters '
                                         'only — unrelated to the MC Gramian, '
                                         'whose knob is "MC traj.")', **L)
        self.w_seed = W.BoundedIntText(value=0, min=0, max=2**31 - 1,
                                       description='seed', **L)
        self.w_x0off = W.Text(value='', description='x̂₀ offset',
                              placeholder='e.g. z:7.5',
                              tooltip='deterministic offsets added to the '
                              'filters\' initial estimate, "state:value" '
                              'comma-separated (paper Fig. 4d: z:7.5 for the '
                              'z₀=9 guess, z:-1 for 0.5); P₀ widens '
                              'consistently', **L)
        self.w_unoise = W.BoundedFloatText(value=0.0, min=0.0, max=1e6,
                                           step=0.01, description='u noise',
                                           tooltip='variance of the noise on '
                                           'the measured inputs the filters '
                                           'receive (paper: 1e-2); truth '
                                           'flies clean inputs', **L)
        self.w_ubias = W.FloatText(value=0.0, description='u bias',
                                   tooltip='constant bias added to one input '
                                   'channel inside [t0, t1)', **L)
        self.w_ubias_ch = W.Dropdown(options=list(self.spec.input_names),
                                     description='bias ch.', **L)
        self.w_ubias_t0 = W.FloatText(value=20.0, description='bias t₀', **L)
        self.w_ubias_t1 = W.FloatText(value=28.0, description='bias t₁', **L)

        self.gb_r = W.GridBox([], layout=W.Layout(
            width='100%', grid_template_columns='repeat(auto-fill, 128px)'))
        self.gb_q = W.GridBox([], layout=W.Layout(
            width='100%', grid_template_columns='repeat(auto-fill, 128px)'))
        self.r_boxes = {}
        self.q_boxes = {}
        self.b_q_tiny = W.Button(description='Q → 0', layout=W.Layout(width='80px'))
        self.b_q_real = W.Button(description='realistic Q', layout=W.Layout(width='100px'))
        self.b_r_default = W.Button(description='reset R', layout=W.Layout(width='80px'))

        self.status = W.HTML()
        self.w_prog = W.FloatProgress(value=0.0, min=0.0, max=1.0,
                                      bar_style='info',
                                      layout=W.Layout(width='320px',
                                                      height='22px'))
        # persistent interactive figures (zoom/pan via the ipympl toolbar),
        # redrawn in place — no Output-widget duplication possible
        self.fig_main = _LiveFig((11, 7.6))
        self.fig_meas = _LiveFig((11, 4.4))
        self.fig_est = _LiveFig((11, 4.4))
        self.b_dice = W.Button(description='🎲 new noise realization',
                               layout=W.Layout(width='190px'))

        # ── trackpad trajectory recorder (interactive ipympl canvas) ──
        self.recorded = None        # (speed_sp, heading_sp) from a drawn path
        self._rec_pts = []
        self._rec_drawing = False
        self._rec_speed = self.spec.v0_default
        try:
            from matplotlib.figure import Figure as _Figure
            from ipympl.backend_nbagg import (Canvas as _Canvas,
                                              FigureManager as _FigureManager)
            rfig = _Figure(figsize=(5, 5))
            self._rec_canvas = _Canvas(rfig)
            # the canvas needs a FigureManager: frontend messages (resize,
            # dpi, events) are dispatched via canvas.manager, which is None
            # otherwise and crashes on the first resize message
            self._rec_manager = _FigureManager(self._rec_canvas, 0)
            self._rec_canvas.header_visible = False
            self._rec_canvas.footer_visible = False
            self._rec_canvas.toolbar_visible = False
            self._rec_canvas.capture_scroll = False
            self._rec_ax = rfig.add_subplot(111)
            self._rec_line, = self._rec_ax.plot([], [], 'b-', lw=2)
            self._rec_start, = self._rec_ax.plot([], [], 'go', ms=6)
            self._rec_canvas.mpl_connect('button_press_event', self._rec_on_press)
            self._rec_canvas.mpl_connect('motion_notify_event', self._rec_on_move)
            self._rec_canvas.mpl_connect('button_release_event', self._rec_on_release)
            self._rec_canvas.mpl_connect('key_press_event', self._rec_on_key)
            rec_widget = self._rec_canvas
        except Exception as e:                      # ipympl missing/broken
            self._rec_canvas = None
            rec_widget = W.HTML(f'<i>trackpad recorder unavailable ({e})</i>')
        self.b_rec_apply = W.Button(description='✓ use drawn path',
                                    button_style='success',
                                    layout=W.Layout(width='150px'))
        self.b_rec_clear = W.Button(description='✗ clear drawing',
                                    layout=W.Layout(width='130px'))
        self.rec_status = W.HTML()
        rec_help = W.HTML(
            'Draw the flight path with the mouse/trackpad (click-drag; extra '
            'strokes extend the path). Click the canvas once to give it '
            'keyboard focus, then: <b>↑/↓</b> speed, <b>c</b> clear, '
            '<b>enter</b> apply. The MPC tracks the drawn path\'s '
            '<i>heading</i> at constant speed, so the flown trajectory follows '
            'the drawn shape (wind and dynamics permitting), not its exact '
            'position.')
        rec_row = W.Layout(display='flex', flex_flow='row wrap')
        self.acc_rec = W.Accordion(children=[W.VBox(
            [rec_help, rec_widget,
             W.HBox([self.b_rec_apply, self.b_rec_clear, self.rec_status],
                    layout=rec_row)])], selected_index=None)
        self.acc_rec.set_title(0, '🎨 record trajectory (trackpad + keyboard)')

        wrap = W.Layout(display='flex', flex_flow='row wrap', width='100%')
        motif_params = W.HBox([self.w_motif, self.w_duration, self.w_angle,
                               self.w_speed, self.w_freq, self.w_amp],
                              layout=wrap)
        seg_buttons = W.HBox([self.b_add, self.b_undo, self.b_clear,
                              W.Label('  quick:'), self.b_left, self.b_right,
                              self.b_fwd, self.b_faster, self.b_slower],
                             layout=wrap)
        sim_row = W.HBox([self.w_system, self.w_v0, self.w_wind,
                          self.w_zeta, self.b_sim], layout=wrap)
        traj_box = W.VBox([W.HTML('Motif parameters: turn/circle→angle, '
                                  'speed→v, weave→f &amp; amp; empty segment '
                                  'list = the system\'s default trajectory'),
                           sim_row, motif_params, seg_buttons,
                           self.acc_rec, self.html_segments],
                          layout=W.Layout(width='100%'))

        # ── section 1: computation settings (these change the numbers) ──
        col_params = W.VBox([W.HTML('<b>analysis parameters</b>'),
                             self.w_win, self.w_eps, self.w_lam])
        col_methods = W.VBox([W.HTML('<b>EV methods</b>'),
                              self.w_empirical, self.w_mc, self.w_nmc])
        fim_col = W.VBox([W.HTML('<b>FIM subset</b> (sensors & states used '
                                 'by all EV pipelines)'),
                          self.w_fim_sensors, self.w_fim_states])
        r_grid = W.VBox([W.HTML('<b>R</b> (sensor noise variance — '
                                'angle sensors in <b>rad²</b>)'),
                         self.gb_r, self.b_r_default],
                        layout=W.Layout(flex='1 1 280px', max_width='420px'))
        q_grid = W.VBox([W.HTML('<b>Q</b> (per-step process-noise covariance '
                                'Q<sub>d</sub> — angle states in <b>rad²</b>; '
                                'e.g. Q<sub>ζ</sub> = 1e-3 ⇒ σ<sub>ζ</sub> ≈ '
                                '0.032·√n rad over n steps)'),
                         self.gb_q,
                         W.HBox([self.b_q_tiny, self.b_q_real])],
                        layout=W.Layout(flex='1 1 280px', max_width='420px'))
        comp_box = W.HBox([col_params, col_methods, fim_col, r_grid, q_grid],
                          layout=W.Layout(display='flex', flex_flow='row wrap',
                                          width='100%', padding='6px',
                                          border='1px solid #4682b4'))

        # ── section 2: display settings (plot selection only) ──
        disp_box = W.HBox([W.VBox([self.w_color_state]), self.w_ev_states],
                          layout=W.Layout(display='flex', flex_flow='row wrap',
                                          width='100%', padding='6px',
                                          border='1px solid #999999'))

        # per-panel control rows (live next to the panel they configure)
        panel_row = W.Layout(display='flex', flex_flow='row wrap',
                             width='100%', padding='4px',
                             border='1px solid #cccccc')
        meas_row = W.HBox([self.w_meas], layout=panel_row)
        bias_box = W.VBox([W.HTML('<b>biased input noise</b> (constant '
                                  'offset on one measured-input channel '
                                  'inside [t₀, t₁))'),
                           W.HBox([self.w_ubias, self.w_ubias_ch,
                                   self.w_ubias_t0, self.w_ubias_t1],
                                  layout=W.Layout(display='flex',
                                                  flex_flow='row wrap'))],
                          layout=W.Layout(border='1px solid #cc8888',
                                          padding='4px', margin='2px'))
        est_row = W.HBox([self.w_est_states,
                          W.VBox([self.w_seed, W.HBox([self.b_dice])]),
                          W.VBox([self.w_init, self.w_x0off, self.w_unoise]),
                          bias_box],
                         layout=panel_row)

        def _acc(title, box, open_=True):
            a = W.Accordion(children=[box],
                            selected_index=0 if open_ else None,
                            layout=W.Layout(width='100%'))
            a.set_title(0, title)
            return a

        acc_traj = _acc('✈ Trajectory — defines the nominal trajectory '
                        '(every result depends on it)', traj_box)
        acc_comp = _acc('⚙ Computation settings — change the calculations '
                        '(windows, λ/ε, R, Q, FIM subsets, EV methods)',
                        comp_box)
        acc_disp = _acc('🎨 Display settings — plot selection only '
                        '(cached; nothing recomputed)', disp_box, open_=False)
        acc_meas = _acc('Measurements panel — sensor selection (display '
                        'only). Gray dots: noisy samples the filters see '
                        '(R + estimator seed); black: noise-free h(x,u); '
                        'blue dashed: EKF prediction h(x̂)',
                        meas_row, open_=False)
        acc_est = _acc('⚙ Estimator settings — change the filter results: '
                       'seed/🎲 realization, init err ×, x̂₀ offset, '
                       'u noise / biased input noise (est. states is '
                       'display-only)', est_row, open_=False)

        self.ui = W.VBox([acc_traj, acc_comp, acc_disp,
                          W.HBox([self.w_prog, self.status],
                                 layout=W.Layout(display='flex',
                                                 flex_flow='row wrap',
                                                 align_items='center',
                                                 width='100%')),
                          self.fig_main.widget,
                          acc_meas, self.fig_meas.widget,
                          acc_est, self.fig_est.widget],
                         layout=W.Layout(width='100%'))

    def _wire_callbacks(self):
        self.w_system.observe(self._on_system_change, names='value')
        self.b_add.on_click(lambda b: self._add_segment_from_widgets())
        self.b_undo.on_click(lambda b: (self.segments and self.segments.pop(),
                                        self._refresh_segment_list()))
        self.b_clear.on_click(lambda b: (self.segments.clear(),
                                         setattr(self, 'recorded', None),
                                         self._refresh_segment_list()))
        self.b_rec_apply.on_click(lambda b: self._rec_apply())
        self.b_rec_clear.on_click(lambda b: self._rec_clear())
        # a turn needs follow-through: the set-point ramp alone would end the
        # trajectory before the vehicle finishes responding, so append a
        # straight hold after the ramp
        self.b_left.on_click(lambda b: self._quick(
            dict(motif='turn', duration=self.spec.quick_turn_dur, angle=np.pi / 2),
            dict(motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_right.on_click(lambda b: self._quick(
            dict(motif='turn', duration=self.spec.quick_turn_dur, angle=-np.pi / 2),
            dict(motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_fwd.on_click(lambda b: self._quick(dict(
            motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_faster.on_click(lambda b: self._quick_speed(+self.spec.speed_step))
        self.b_slower.on_click(lambda b: self._quick_speed(-self.spec.speed_step))
        self.b_sim.on_click(lambda b: self._rebuild_trajectory())
        self.b_q_tiny.on_click(lambda b: self._set_boxes(self.q_boxes,
                                                         self.spec.q_tiny))
        self.b_q_real.on_click(lambda b: self._set_boxes(self.q_boxes,
                                                         self.spec.q_realistic))
        self.b_r_default.on_click(lambda b: self._set_boxes(self.r_boxes,
                                                            self.spec.r_default))
        self.b_dice.on_click(lambda b: setattr(
            self.w_seed, 'value', int(np.random.default_rng().integers(1e6))))
        for wd in (self.w_win, self.w_eps, self.w_lam,
                   self.w_color_state, self.w_ev_states, self.w_est_states,
                   self.w_meas, self.w_fim_sensors, self.w_fim_states,
                   self.w_empirical, self.w_mc, self.w_nmc, self.w_init,
                   self.w_seed, self.w_unoise, self.w_ubias, self.w_ubias_ch,
                   self.w_ubias_t0, self.w_ubias_t1, self.w_x0off):
            wd.observe(self._on_param_change, names='value')

    def _make_spec(self, name):
        if self._dt_override is not None:
            return SYSTEMS[name](dt=float(self._dt_override))
        return SYSTEMS[name]()

    @staticmethod
    def _meas_default(spec):
        picks = tuple(m for m in ('gamma', 'a', 'g')
                      if m in spec.measurement_names)
        return picks or tuple(spec.measurement_names[:3])

    # -- system switching --------------------------------------------------------
    def _apply_system_defaults(self):
        """ Push the current spec's defaults into all widgets (no recompute). """
        s = self.spec
        self._suspend = True
        self.w_v0.value = s.v0_default
        self.w_wind.value = s.wind_default
        self.w_zeta.value = s.zeta_default
        self.w_win.value = s.w_default
        self.w_eps.value = s.eps_default
        self.w_duration.value = s.seg_duration
        self.w_speed.value = s.v0_default
        self.w_color_state.options = list(s.state_names)
        self.w_color_state.value = s.color_state_default
        self.w_ev_states.options = list(s.state_names)
        self.w_ev_states.value = s.ev_states_default
        self.w_est_states.options = list(s.state_names)
        self.w_est_states.value = s.ev_states_default
        self.w_meas.options = list(s.measurement_names)
        self.w_meas.value = self._meas_default(s)
        self.w_fim_sensors.options = list(s.measurement_names)
        self.w_fim_sensors.value = tuple(s.fim_sensors)
        self.w_ubias_ch.options = list(s.input_names)
        self.w_unoise.value = getattr(s, 'u_noise_default', 0.0)
        ub = getattr(s, 'u_bias_default', None)
        if ub:
            ch, mag, t0, t1 = ub
            self.w_ubias_ch.value = ch
            self.w_ubias.value = mag
            self.w_ubias_t0.value = t0
            self.w_ubias_t1.value = t1
        else:
            self.w_ubias_ch.value = s.input_names[0]
            self.w_ubias.value = 0.0
        self.w_x0off.value = ''
        self.w_fim_states.options = list(s.state_names)
        self.w_fim_states.value = tuple(s.fim_states)
        txt = dict(layout=W.Layout(width='118px'),
                   style={'description_width': '48px'})
        self.r_boxes = {
            m: W.FloatText(value=s.r_default[m], description=m,
                           tooltip=(f'{m} noise variance '
                                    + ('[rad²]' if m in s.angle_measurements
                                       else '[native units²]')), **txt)
            for m in s.measurement_names}
        self.q_boxes = {
            n: W.FloatText(value=s.q_tiny[n], description=n,
                           tooltip=(f'{n} per-step process-noise covariance Q_d '
                                    + ('[rad²]' if n in s.angle_states
                                       else '[native units²]')), **txt)
            for n in s.state_names}
        for wd in list(self.r_boxes.values()) + list(self.q_boxes.values()):
            wd.observe(self._on_param_change, names='value')
        self.gb_r.children = list(self.r_boxes.values())
        self.gb_q.children = list(self.q_boxes.values())
        self._suspend = False
        self._rec_reset()

    def _on_system_change(self, change):
        if self._suspend:
            return
        self.spec = self._make_spec(change['new'])
        self.engine = ObservabilityEngine(self.spec)
        self.segments = []
        self._apply_system_defaults()
        self._rebuild_trajectory()

    # -- trackpad recorder --------------------------------------------------------
    def _rec_reset(self):
        """ Configure the recorder canvas for the current system. """
        self.recorded = None
        self._rec_pts = []
        self._rec_drawing = False
        self._rec_speed = self.spec.v0_default
        self.rec_status.value = ''
        if self._rec_canvas is None:
            return
        ax = self._rec_ax
        ax.set_xlim(*self.spec.rec_xlim)
        ax.set_ylim(*self.spec.rec_ylim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        self._rec_line.set_data([], [])
        self._rec_start.set_data([], [])
        self._rec_set_title()
        self._rec_canvas.draw_idle()

    def _rec_set_title(self):
        self._rec_ax.set_title(
            f'draw path — speed {self._rec_speed:.2f} m/s (↑/↓ to change)',
            fontsize=9)

    def _rec_redraw(self):
        pts = np.asarray(self._rec_pts)
        self._rec_line.set_data(pts[:, 0], pts[:, 1])
        self._rec_start.set_data(pts[:1, 0], pts[:1, 1])
        self._rec_canvas.draw_idle()

    def _rec_on_press(self, ev):
        if ev.inaxes is self._rec_ax and ev.xdata is not None:
            self._rec_drawing = True
            self._rec_pts.append((ev.xdata, ev.ydata))
            self._rec_redraw()

    def _rec_on_move(self, ev):
        if (self._rec_drawing and ev.inaxes is self._rec_ax
                and ev.xdata is not None):
            self._rec_pts.append((ev.xdata, ev.ydata))
            self._rec_redraw()

    def _rec_on_release(self, ev):
        self._rec_drawing = False

    def _rec_on_key(self, ev):
        if ev.key == 'up':
            self._rec_speed *= 1.25
        elif ev.key == 'down':
            self._rec_speed = max(self._rec_speed / 1.25, 1e-3)
        elif ev.key == 'c':
            self._rec_pts = []
            self._rec_line.set_data([], [])
            self._rec_start.set_data([], [])
        elif ev.key == 'enter':
            self._rec_apply()
            return
        self._rec_set_title()
        self._rec_canvas.draw_idle()

    def _rec_clear(self):
        self._rec_pts = []
        self.recorded = None
        self.rec_status.value = ''
        if self._rec_canvas is not None:
            self._rec_line.set_data([], [])
            self._rec_start.set_data([], [])
            self._rec_canvas.draw_idle()
        self._refresh_segment_list()

    def _rec_apply(self):
        """ Convert the drawn path into (speed, heading) set-points at the
        recorder speed, resampled uniformly in arc length, and simulate. """
        pts = np.asarray(self._rec_pts, dtype=float)
        if len(pts) >= 2:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            pts = pts[np.concatenate([[True], d > 1e-12])]
        if len(pts) < 5:
            self.rec_status.value = ('<span style="color:firebrick">draw a '
                                     'longer path first</span>')
            return
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        sarc = np.concatenate([[0.0], np.cumsum(seglen)])
        v, dt = self._rec_speed, self.spec.dt
        N = int(np.clip(round(sarc[-1] / (v * dt)), 8, 800))
        si = np.linspace(0.0, sarc[-1], N)
        xi = np.interp(si, sarc, pts[:, 0])
        yi = np.interp(si, sarc, pts[:, 1])
        heading_sp = np.unwrap(np.arctan2(np.gradient(yi), np.gradient(xi)))
        self.recorded = (v * np.ones(N), heading_sp)
        self.segments.clear()
        self.rec_status.value = (f'recorded {sarc[-1]:.2f} m path → {N} steps '
                                 f'({N * dt:.2f} s at {v:.2f} m/s)')
        self._refresh_segment_list()
        self._rebuild_trajectory()

    # -- segment helpers --------------------------------------------------------
    def _add_segment_from_widgets(self):
        seg = dict(motif=self.w_motif.value, duration=float(self.w_duration.value),
                   angle=float(self.w_angle.value), speed=float(self.w_speed.value),
                   freq=float(self.w_freq.value), amp=float(self.w_amp.value))
        self.segments.append(seg)
        self._refresh_segment_list()

    def _quick(self, *segs):
        self.segments.extend(segs)
        self._refresh_segment_list()

    def _quick_speed(self, dv):
        v_last = float(self.w_v0.value)
        for s in self.segments:
            if s['motif'] == 'speed':
                v_last = s['speed']
        self.segments.append(dict(motif='speed',
                                  duration=self.spec.seg_duration,
                                  speed=v_last + dv))
        self._refresh_segment_list()

    def _refresh_segment_list(self):
        if not self.segments:
            if self.recorded is not None:
                n_rec = len(self.recorded[0])
                self.html_segments.value = (
                    f'<i>🎨 recorded trackpad trajectory ({n_rec} steps, '
                    f'{n_rec * self.spec.dt:.2f} s) — clear to discard</i>')
            else:
                self.html_segments.value = (
                    f'<i>default: {self.spec.default_traj_label} '
                    f'— add segments to build your own</i>')
            return
        items = ' &nbsp;→&nbsp; '.join(
            f'<b>{i+1}.</b> {MOTIF_LABELS[s["motif"]](s)}'
            for i, s in enumerate(self.segments))
        total = sum(s['duration'] for s in self.segments)
        self.html_segments.value = (f'{items} &nbsp; <i>(total {total:g} s — press '
                                    f'▶ simulate to apply)</i>')

    def _set_boxes(self, boxes, values):
        for k, wd in boxes.items():
            wd.unobserve(self._on_param_change, names='value')
            wd.value = values[k]
            wd.observe(self._on_param_change, names='value')
        self._on_param_change(None)

    def _parse_x0_offset(self):
        txt = self.w_x0off.value.strip()
        if not txt:
            return None
        out = {}
        for part in txt.split(','):
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip()
                if k in self.spec.state_names:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        return out or None

    def _set_status(self, txt, frac=None):
        """ Large status line + stage progress bar (updates render live
        mid-callback). """
        self.status.value = ('<div style="font-size:135%; font-weight:600; '
                             'padding:3px 0">' + txt + '</div>')
        if frac is not None:
            self.w_prog.value = float(frac)
            self.w_prog.bar_style = 'success' if frac >= 1.0 else 'info'

    # -- parameter collection ----------------------------------------------------
    def _params(self):
        r_diag = {m: max(float(wd.value), 1e-30) for m, wd in self.r_boxes.items()}
        q_diag = {n: max(float(wd.value), 1e-30) for n, wd in self.q_boxes.items()}
        w = int(min(self.w_win.value, max(self.engine.N - 1, 3)))
        sensors = tuple(self.w_fim_sensors.value) or tuple(
            self.spec.measurement_names)
        states = tuple(self.w_fim_states.value) or tuple(self.spec.state_names)
        return dict(w=w, eps=float(self.w_eps.value), lam=float(self.w_lam.value),
                    r_diag=r_diag, q_diag=q_diag, sensors=sensors,
                    states=states)

    # -- actions -----------------------------------------------------------------
    def _rebuild_trajectory(self):
        self._refresh_segment_list()
        s = self.spec
        w_wind = float(self.w_wind.value)
        zeta = float(self.w_zeta.value)
        if self.segments:
            speed_sp, heading_sp = build_setpoints(
                self.segments, s.dt, v0=float(self.w_v0.value),
                heading0=s.heading0)
            setpoint = s.make_setpoint(speed_sp, heading_sp, w_wind, zeta)
        elif self.recorded is not None:
            speed_sp, heading_sp = self.recorded
            setpoint = s.make_setpoint(speed_sp, heading_sp, w_wind, zeta)
        else:
            setpoint = s.default_setpoint(w_wind, zeta)
        self._set_status('⏳ solving MPC trajectory …', 0.05)
        t0 = time.time()
        self.engine.simulate_mpc(setpoint)
        self._set_status('trajectory solved — computing observability …', 0.2)
        t_sim = time.time() - t0
        self._update(f'trajectory: {self.engine.N} steps in {t_sim:.1f} s')

    def _on_param_change(self, change):
        if self._suspend or not self.engine.N:
            return
        self._update()

    # -- compute + draw ------------------------------------------------------------
    def _update(self, note=''):
        try:
            self._update_inner(note)
        except np.linalg.LinAlgError as e:
            self._set_status('<span style="color:firebrick">⚠ singular '
                             f'matrix at these settings ({e}) — typically '
                             'λ far below the noise floor of F, or a '
                             'degenerate R/Q combination. Raise λ or '
                             'adjust R/Q; plots show the previous '
                             'result.</span>', 1.0)

    def _update_inner(self, note=''):
        p = self._params()
        self._set_status('⏳ linearized Eq. (33) pipeline …', 0.25)
        t0 = time.time()
        ev_lin = self.engine.linearized_ev(p['w'], p['q_diag'], p['r_diag'],
                                           p['lam'], sensors=p['sensors'],
                                           states=p['states'])
        t_lin = time.time() - t0

        ev_mc, mc_note = None, ''
        if self.w_mc.value:
            self._set_status('⏳ nonlinear Monte-Carlo Gramian … '
                             '(scales with MC traj. × windows; cached)', 0.4)
            t0 = time.time()
            n_mc = int(self.w_nmc.value)
            ev_mc, mc_cached = self.engine.mc_ev(
                p['w'], p['q_diag'], p['r_diag'], p['lam'],
                sensors=p['sensors'], states=p['states'],
                n_mc=n_mc, seed=0)
            mc_note = (' | MC: cached' if mc_cached
                       else f' | MC: {time.time() - t0:.1f} s')

        ev_emp, emp_note = None, ''
        if self.w_empirical.value:
            self._set_status('⏳ pybounds empirical pipeline … (slower the '
                             'first time for a given trajectory / w / ε; '
                             'cached afterwards)', 0.55)
            t0 = time.time()
            ev_emp, cached = self.engine.empirical_ev(
                p['w'], p['eps'], p['r_diag'], p['lam'],
                sensors=p['sensors'], states=p['states'])
            emp_note = (' | empirical: cached' if cached
                        else f' | empirical: {time.time() - t0:.1f} s')

        self._set_status('⏳ filters (EKF / UKF) + rendering …', 0.8)
        self._draw(ev_lin, ev_emp, p, ev_mc=ev_mc)
        try:
            seed = int(self.w_seed.value)
            scale = float(self.w_init.value)
            x0off = self._parse_x0_offset()
            unv = max(float(self.w_unoise.value), 0.0)
            ub = None
            if float(self.w_ubias.value) != 0.0:
                ub = (self.spec.input_names.index(self.w_ubias_ch.value),
                      float(self.w_ubias.value),
                      float(self.w_ubias_t0.value),
                      float(self.w_ubias_t1.value))
            results = {}
            for name, fn in (('EKF', self.engine.run_ekf),
                             ('UKF', self.engine.run_ukf)):
                fkey = (name, self.engine._version,
                        tuple(f'{p["q_diag"][k]:.3e}'
                              for k in self.spec.state_names),
                        tuple(f'{p["r_diag"][m]:.3e}'
                              for m in self.spec.measurement_names),
                        seed, round(scale, 6), round(unv, 9), ub,
                        tuple(sorted(x0off.items())) if x0off else None)
                if fkey not in self.engine._filt_cache:
                    self.engine._filt_cache[fkey] = fn(
                        p['q_diag'], p['r_diag'], seed=seed, x0_scale=scale,
                        x0_offset=x0off, u_noise_var=unv, u_bias=ub)
                results[name] = self.engine._filt_cache[fkey]
            self._draw_estimates(results)
            self._draw_measurements(results['EKF'][0], p)
        except np.linalg.LinAlgError:
            pass  # keep the previous panel if a filter breaks down
        floor = self.engine.lam_noise_floor
        floor_note = ''
        if p['lam'] < 10 * floor:
            floor_note = (f' | <span style="color:firebrick">⚠ λ = {p["lam"]:.0e} '
                          f'is at/below the round-off noise floor of F '
                          f'(eps·λ_max ≈ {floor:.0e}) — EVs may be unreliable; '
                          f'raise λ</span>')
        self._set_status(f'✓ done — {note + " | " if note else ""}'
                         f'linearized: {t_lin * 1e3:.0f} ms{emp_note}'
                         f'{mc_note}{floor_note}', 1.0)

    def _draw(self, ev_lin, ev_emp, p, ev_mc=None):
        self.fig_main.draw(
            lambda f: self._render(f, ev_lin, ev_emp, p, ev_mc=ev_mc))

    def _draw_estimates(self, results):
        s = self.spec
        t, X = self.engine.t, self.engine.X
        states = list(self.w_est_states.value) or [self.spec.color_state_default]
        states = states[:8]
        ncols = 2 if len(states) > 3 else 1
        nrows = int(np.ceil(len(states) / ncols))

        def render(fig):
            axs = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
            colors = {'EKF': 'royalblue', 'UKF': 'darkorange'}
            styles = {'EKF': '--', 'UKF': '-.'}
            for i, st in enumerate(states):
                ax = axs.ravel()[i]
                j = s.state_names.index(st)
                for name, (X_hat, P_diag) in results.items():
                    sd = np.sqrt(np.clip(P_diag[:, j], 0.0, None))
                    ax.fill_between(t, X_hat[:, j] - 3 * sd,
                                    X_hat[:, j] + 3 * sd,
                                    color=colors[name], alpha=0.12, lw=0)
                    ax.plot(t, X_hat[:, j], styles[name], color=colors[name],
                            lw=1.5, label=f'{name} ±3σ')
                ax.plot(t, X[:, j], 'k-', lw=1.8, label='true')
                ax.set_ylabel(st)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7, loc='best')
            for ax in axs.ravel()[len(states):]:
                ax.set_visible(False)
            for ax in axs[-1, :]:
                ax.set_xlabel('time [s]')
            fig.suptitle('true vs estimated states — EKF vs UKF', fontsize=10)
            fig.tight_layout()

        self.fig_est.draw(render, figsize=(11, max(2.2 * nrows, 2.4)))

    def _draw_measurements(self, X_hat_ekf, p):
        """ Selected sensors vs time: the noisy samples the filters consume
        (drawn with the current R and seed — identical to the filter input),
        the noise-free h along the true trajectory, and the EKF's measurement
        prediction h(x_hat) (its smoothness/aggressiveness is set by Q). """
        s = self.spec
        t, X, U = self.engine.t, self.engine.X, self.engine.U
        sensors = list(self.w_meas.value) or list(self._meas_default(s))
        sensors = sensors[:8]
        Y_noisy, *_ = self.engine._estimation_problem(
            p['r_diag'], int(self.w_seed.value))
        Y_true = np.array([np.asarray(s.h(X[k], U[k]), dtype=float)
                           for k in range(self.engine.N)])
        Y_hat = np.array([np.asarray(s.h(X_hat_ekf[k], U[k]), dtype=float)
                          for k in range(self.engine.N)])
        ncols = 2 if len(sensors) > 3 else 1
        nrows = int(np.ceil(len(sensors) / ncols))

        def render(fig):
            axs = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
            for i, m in enumerate(sensors):
                ax = axs.ravel()[i]
                j = s.measurement_names.index(m)
                ax.plot(t, Y_noisy[:, j], '.', color='0.55', ms=3.5,
                        label='noisy samples (R)')
                ax.plot(t, Y_true[:, j], 'k-', lw=1.6,
                        label='noise-free h(x,u)')
                ax.plot(t, Y_hat[:, j], '--', color='royalblue', lw=1.4,
                        label='EKF prediction h(x̂)')
                unit = ' [rad]' if m in s.angle_measurements else ''
                ax.set_ylabel(m + unit)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7, loc='best')
            for ax in axs.ravel()[len(sensors):]:
                ax.set_visible(False)
            for ax in axs[-1, :]:
                ax.set_xlabel('time [s]')
            fig.suptitle('measurements: noisy (R) | true | EKF prediction',
                         fontsize=10)
            fig.tight_layout()

        self.fig_meas.draw(render, figsize=(11, max(2.0 * nrows, 2.2)))

    def _ev_along_path(self, src, cs):
        """ Map a pipeline's min-EV of state `cs` onto trajectory points
        (window-start alignment, edges padded). """
        ev_path = np.full(self.engine.N, np.nan)
        vals = src[cs].values
        ti = src['time_initial'].values
        ok = np.isfinite(ti) & np.isfinite(vals)
        ev_path[np.rint(ti[ok] / self.engine.dt).astype(int)] = vals[ok]
        fin = np.flatnonzero(np.isfinite(ev_path))
        if len(fin):
            ev_path[:fin[0]] = ev_path[fin[0]]
            ev_path[fin[-1]:] = ev_path[fin[-1]]
        return ev_path

    def _plot_colored_traj(self, fig, ax, ev_path, norm, cs, src_lab):
        X = self.engine.X
        ax.plot(X[:, 0], X[:, 1], 'k-', lw=0.3, alpha=0.4)
        if ev_path is not None and norm is not None:
            with contextlib.redirect_stdout(io.StringIO()):  # colorline prints its norm
                lc = colorline(X[:, 0], X[:, 1], ev_path, ax=ax,
                               cmap=plt.get_cmap('inferno_r'), norm=norm,
                               linewidth=4)
            cb = fig.colorbar(lc, ax=ax, pad=0.02)
            cb.set_label(f'min EV: {cs}', fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            ax.annotate(f'no EV to show for "{cs}" —\nadd it to "FIM states"'
                        '\n(or enable pybounds empirical)',
                        xy=(0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center', fontsize=9, color='gray')
        ax.plot(X[0, 0], X[0, 1], 'go', ms=7, zorder=4, label='start')
        self._draw_wind_glyph(ax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'min EV of {cs} — {src_lab}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    def _draw_wind_glyph(self, ax):
        """ Corner glyph: arrow along the wind direction zeta, with a shaded
        fan spanning ±2σ where σ = sqrt(Q_ζ · T) is the wind-direction drift
        the current Q implies over the whole trajectory. """
        s = self.spec
        if 'w' not in s.state_names or 'zeta' not in s.state_names:
            return
        X = self.engine.X
        w_wind = X[0, s.state_names.index('w')]
        zeta = X[0, s.state_names.index('zeta')]
        q_zeta = max(float(self.q_boxes['zeta'].value), 0.0)   # per-step Q_d
        # zeta is a random walk, so its drift over a window of n_w STEPS has
        # Var = Q_d * n_w. (With the old PSD convention this read Q_c * T_w, which
        # is the same quantity; with Q_d it must be steps, NOT seconds.)
        n_w = min(int(self.w_win.value), max(self.engine.N - 1, 1))
        sigma = np.sqrt(q_zeta * n_w)   # drift over one analysis window
        half = min(2.0 * sigma, np.pi)          # fan half-angle = 2σ, capped
        cx, cy, r = 0.13, 0.86, 0.10            # axes-fraction center/radius
        tf = ax.transAxes
        if half > 0.01:
            ax.add_patch(mpatches.Wedge(
                (cx, cy), r, np.degrees(zeta - half), np.degrees(zeta + half),
                transform=tf, facecolor='steelblue', alpha=0.25, lw=0,
                zorder=5, clip_on=False))
        ax.add_patch(mpatches.FancyArrow(
            cx - 0.85 * r * np.cos(zeta), cy - 0.85 * r * np.sin(zeta),
            1.7 * r * np.cos(zeta), 1.7 * r * np.sin(zeta),
            transform=tf, width=0.012, color='steelblue',
            length_includes_head=True, zorder=6, clip_on=False))
        label = f'wind {w_wind:.2g} m/s, ζ={zeta:.2f} rad'
        if half > 0.01:
            label += f'\n±2σ(ζ) = {half:.2f} rad over window w = {T_w:.2g} s'
        ax.text(cx, cy - r - 0.02, label, transform=tf, ha='center', va='top',
                fontsize=6.5, color='steelblue', zorder=6)

    def _render(self, fig, ev_lin, ev_emp, p, ev_mc=None):
        cs = self.w_color_state.value
        ev_states = list(self.w_ev_states.value) or [self.spec.color_state_default]
        gs = fig.add_gridspec(2, 2)
        ax_lin = fig.add_subplot(gs[0, 0])
        ax_emp = fig.add_subplot(gs[0, 1])
        ax_ev = fig.add_subplot(gs[1, :])

        # (top row) trajectory colored by min EV — linearized | pybounds,
        # on a SHARED log color scale so the two colorings are comparable
        path_lin = self._ev_along_path(ev_lin, cs) if cs in ev_lin else None
        path_emp = (self._ev_along_path(ev_emp, cs)
                    if ev_emp is not None and cs in ev_emp else None)
        arrays = [v[np.isfinite(v)] for v in (path_lin, path_emp)
                  if v is not None]
        allv = np.concatenate(arrays) if arrays else np.array([])
        allv = allv[allv > 0]
        norm = None
        if len(allv):
            lo = int(np.floor(np.log10(allv.min())))
            hi = int(np.ceil(np.log10(allv.max())))
            norm = mpl.colors.LogNorm(10.0 ** lo, 10.0 ** max(hi, lo + 1))
        self._plot_colored_traj(fig, ax_lin, path_lin, norm, cs,
                                'linearized Eq. (33)')
        self._plot_colored_traj(fig, ax_emp, path_emp,
                                norm if path_emp is not None else None,
                                cs, 'pybounds empirical')

        # (bottom row) per-state COLOR, per-method LINESTYLE:
        # solid = linearized Eq.(33), dashed = pybounds empirical,
        # dotted = nonlinear Monte-Carlo Gramian
        from matplotlib.lines import Line2D
        cmap = plt.get_cmap('tab10')

        def plot_methods(ax, x_of, lin_df, emp_df, mc_df, plotter):
            handles = []
            for i, s in enumerate(ev_states):
                c = cmap(i % 10)
                if s in lin_df:
                    plotter(ax, x_of(lin_df), lin_df[s], '-', c, 1.8)
                if emp_df is not None and s in emp_df:
                    plotter(ax, x_of(emp_df), emp_df[s], '--', c, 1.3)
                if mc_df is not None and s in mc_df:
                    plotter(ax, x_of(mc_df), mc_df[s], ':', c, 1.6)
                handles.append(Line2D([], [], color=c, label=s))
            handles.append(Line2D([], [], color='k', ls='-',
                                  label='linearized'))
            if emp_df is not None:
                handles.append(Line2D([], [], color='k', ls='--',
                                      label='pybounds'))
            if mc_df is not None:
                handles.append(Line2D([], [], color='k', ls=':',
                                      label='nonlinear MC'))
            return handles

        semi = lambda ax, x, y, ls, c, lw: ax.semilogy(x, y, ls, color=c, lw=lw)

        h1 = plot_methods(ax_ev, lambda d: d['time'],
                          ev_lin, ev_emp, ev_mc, semi)
        ax_ev.axhline(1.0 / p['lam'], color='gray', ls=':', lw=1)
        h1.append(Line2D([], [], color='gray', ls=':', label='1/λ floor'))
        ax_ev.set_xlabel('window start time [s]')
        ax_ev.set_ylabel('min error variance')
        ax_ev.set_title(f'min EV, sliding window w={p["w"]}', fontsize=9)
        ax_ev.legend(handles=h1, fontsize=6, ncol=3)
        ax_ev.grid(alpha=0.3, which='both')

        fig.tight_layout()

    def display(self):
        display(self.ui)
