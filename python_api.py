"""Connects Python to the compiled solver (cavity_solver.dll on Windows, libcavity_solver.so on Linux).

Nothing is computed here: settings go in, NumPy arrays come out. The three records below must list the
same fields, in the same order and with the same types, as Config / Report / Diagnostics in backend.cu."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np


class Config(ct.Structure):
    _fields_ = ([(name, ct.c_int) for name in
                 'nx ny nz lid_shaped impulse_order adaptive_dt dt_lag phi_extrapolate fdm_max_passes '
                 'grid_stretch n_probes'.split()]
                + [(name, ct.c_double) for name in
                   'length u_lid nu lz lid_ramp_t reinit_r impulse_order4_ramp grid_dx_wall grid_max_aspect '
                   'dt_fixed dt_max cfl_target diffusion_limit fdm_rtol'.split()]
                + [('probe_x', ct.c_double*8), ('probe_y', ct.c_double*8)])


class Report(ct.Structure):
    _fields_ = [('steps', ct.c_longlong), ('time', ct.c_double), ('dt', ct.c_double), ('cfl', ct.c_double),
                ('phi_rel', ct.c_double), ('dt_reason', ct.c_int), ('phi_passes', ct.c_int)]


class Diagnostics(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in
                'kinetic_energy enstrophy umax div_rms cfl_rate gauge_ratio gauge_x gauge_y'.split()]


DT_REASONS = ('fixed_DT', 'DT_MAX', 'CFL', 'viscous')
FIELD_NUMBER = {'u': 0, 'v': 1, 'w': 2, 'mx': 3, 'my': 4, 'mz': 5, 'phi': 6}


def pointer(a):
    """C pointer to a NumPy array's data (None stays None = NULL)."""
    return None if a is None else a.ctypes.data_as(ct.c_void_p)


class Solver:
    """The native solver. Create one, call initialize(), then step() in a loop."""

    def __init__(self, settings, library=None):
        folder = Path(__file__).resolve().parent
        if library is None:
            library = folder / ('cavity_solver.dll' if os.name == 'nt' else 'libcavity_solver.so')
        if os.name == 'nt' and os.environ.get('CUDA_PATH'):          # where Windows finds cublas64_*.dll
            cuda_bin = Path(os.environ['CUDA_PATH']) / 'bin'
            if cuda_bin.is_dir():
                self._dll_dir = os.add_dll_directory(str(cuda_bin))
        if not Path(library).is_file():
            raise FileNotFoundError(f'Build the solver first (build_windows.bat). Missing: {library}')
        self.lib = ct.CDLL(str(library))
        self.lib.solver_last_error.restype = ct.c_char_p
        self.lib.solver_set_clock.argtypes = [ct.c_double, ct.c_longlong, ct.c_double]
        self.lib.solver_kick.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_int]
        self.lib.solver_add_statistics.argtypes = [ct.c_double, ct.c_void_p]

        config = Config()
        for name, _ in Config._fields_:
            if name in settings:
                setattr(config, name, settings[name])
        probes = list(settings.get('probes', ()))[:8]
        config.n_probes = len(probes)
        for k, (px, py) in enumerate(probes):
            config.probe_x[k], config.probe_y[k] = px, py
        self.config = config
        self.call('solver_create', ct.byref(config))

        nx, ny, nz = config.nx, config.ny, config.nz
        self.nx, self.ny, self.nz = nx, ny, nz
        # NumPy shapes [z, y, x] of every field
        self.shapes = {'u': (nz, ny, nx+1), 'v': (nz, ny+1, nx), 'w': (nz+1, ny, nx), 'phi': (nz, ny, nx)}
        self.shapes.update(mx=self.shapes['u'], my=self.shapes['v'], mz=self.shapes['w'])
        # grid: faces and cell centres per axis
        self.grid = {}
        for axis, (name, n) in enumerate((('x', nx), ('y', ny), ('z', nz))):
            faces, centres = np.empty(n+1), np.empty(n)
            self.call('solver_grid', axis, pointer(faces), pointer(centres))
            self.grid[name] = (faces, centres)

    def call(self, name, *args):
        if getattr(self.lib, name)(*args):
            raise RuntimeError(self.lib.solver_last_error().decode())

    def initialize(self, v_noise=None, w_noise=None):
        v = None if v_noise is None else np.ascontiguousarray(v_noise, dtype=np.float64)
        w = None if w_noise is None else np.ascontiguousarray(w_noise, dtype=np.float64)
        self.call('solver_initialize', pointer(v), pointer(w))

    def step(self):
        r = Report()
        self.call('solver_step', ct.byref(r))
        return self._as_dict(r)

    def report(self):
        r = Report()
        self.call('solver_report', ct.byref(r))
        return self._as_dict(r)

    @staticmethod
    def _as_dict(r):
        d = {name: getattr(r, name) for name, _ in Report._fields_}
        d['dt_reason'] = DT_REASONS[r.dt_reason]
        return d

    def kick(self, noise, amplitude, passes):
        n = [np.ascontiguousarray(a, dtype=np.float64) for a in noise]
        self.call('solver_kick', pointer(n[0]), pointer(n[1]), pointer(n[2]), float(amplitude), int(passes))

    def diagnostics(self):
        d = Diagnostics()
        self.call('solver_diagnostics', ct.byref(d))
        return {name: getattr(d, name) for name, _ in Diagnostics._fields_}

    def field(self, name):
        out = np.empty(self.shapes[name], dtype=np.float64 if name == 'phi' else np.float32)
        self.call('solver_get_field', FIELD_NUMBER[name], pointer(out))
        return out

    def set_field(self, name, values):
        a = np.ascontiguousarray(values, dtype=np.float64 if name == 'phi' else np.float32)
        assert a.shape == self.shapes[name], f'{name}: shape {a.shape}, expected {self.shapes[name]}'
        self.call('solver_set_field', FIELD_NUMBER[name], pointer(a))

    def set_clock(self, time, steps, last_dt):
        self.call('solver_set_clock', float(time), int(steps), float(last_dt))

    def probes(self):
        out = np.zeros(2*self.config.n_probes, dtype=np.float32)
        self.call('solver_probes', pointer(out))
        return out

    def add_statistics(self, weight):
        """Add the current mid-span field to the statistics; returns the centreline sample (u, v, uu, vv, ww)."""
        line = np.empty((5, self.nx))
        self.call('solver_add_statistics', float(weight), pointer(line))
        return line

    def statistics(self):
        sums = np.empty((6, self.ny, self.nx))
        self.call('solver_get_statistics', pointer(sums))
        return dict(zip(('u', 'v', 'uu', 'vv', 'ww', 'uv'), sums))

    def set_statistics(self, sums):
        a = np.ascontiguousarray([sums[k] for k in ('u', 'v', 'uu', 'vv', 'ww', 'uv')], dtype=np.float64)
        self.call('solver_set_statistics', pointer(a))

    def midplane(self):
        out = np.empty((4, self.ny, self.nx), dtype=np.float32)
        self.call('solver_midplane', pointer(out))
        return out

    def q_volume(self, colour_w=True):
        q = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        colour = np.empty_like(q)
        self.call('solver_q_volume', int(bool(colour_w)), pointer(q), pointer(colour))
        return q, colour

    def close(self):
        self.call('solver_destroy')
