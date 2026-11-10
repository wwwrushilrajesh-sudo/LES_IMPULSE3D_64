"""Lid-driven cavity LES: Python front end of the native CUDA C++ solver.

Edit the settings below and run this file (Spyder: F5). Python only sets the run up, asks the backend to
advance, and writes CSV files / plots; every numerical stage runs in the compiled library (backend.cu).
Options:  --restart <checkpoint.npz>   continue a run
"""
import csv
import os
import sys
import time
import numpy as np

from python_api import Solver

# =============================================================================
# SETTINGS  (same names and meaning as in the CuPy version)
# =============================================================================
# ----------------------------- problem --------------------------------------
L = 1.0                          # cavity side
U_LID = 1.0                      # lid speed
RE = 22000.0
NU = U_LID*L/RE
LZ_L = 1                         # spanwise extent in units of L (1 = cubic cavity)
Z_WALLS = True                   # solid end walls at z = 0 and z = LZ (this version: walls only)
LID_PROFILE = 'regularised'      # 'regularised' (corner-smoothed lid) | 'uniform'
LID_RAMP_T = 0.3                 # 0 = impulsive start; > 0 = linear ramp over this time
# ----------------------------- grid -----------------------------------------
MESH = 'FINE'
MESH_OPTIONS = {'TEST': (32, 8), 'VALID': (64, 4), 'COARSE': (96, 24), 'MEDIUM': (256, 4), 'FINE': (128, 128)}
GRID_STRETCH = True              # two-sided tanh clustering towards every wall
GRID_DX_WALL = 0.001             # first-cell size at the walls [L]
GRID_MAX_ASPECT = 4              # cap on (largest cell)/(smallest cell) per axis
# ----------------------------- scheme (Eulerian impulse, SSP-RK3) -----------
REINIT_R = 0.98                  # gauge retention per step
IMPULSE_ORDER = 4                # 4th-order Lamb vector away from the walls (2 = 2nd order everywhere)
IMPULSE_ORDER4_RAMP = 1          # cells over which the 4th-order correction is switched on
PHI_EXTRAPOLATE = True           # warm-start each projection from the previous stage potentials
FDM_RTOL = 1.0e-9                # pressure solve ||b - L phi|| / ||b||
FDM_MAX_PASSES = 8
# ----------------------------- time step ------------------------------------
ADAPTIVE_DT = True
DT_FIXED = 0.002
CFL_TARGET = 0.4
DT_MAX = 0.004
STRETCH_DIFFUSION_LIMIT = 2.0    # dt * max_cell nu * sum 4/(dx dmin) <= this
DT_LAG = True                    # use the previous step's velocity for dt (as the original GPU version did)
# ----------------------------- perturbations --------------------------------
INIT_PERTURB = 0.005             # relative noise on v, w at t = 0
PERTURB_SEED = 3
KICK_T = 100                     # first kick
KICK_COUNT = 3
KICK_EVERY = 50.0
KICK_AMP = 0.005                 # rms of each kick relative to U_LID (0 = off)
KICK_CELLS = 20                  # smoothing width of the kick noise
# ----------------------------- run and output -------------------------------
T_END = 350.0
N_STEPS = 6000000
STATS_START_T = 45.0             # start of mean / rms accumulation
STATS_WINDOW_T = 100.0           # trailing window of the centreline mean / rms
PRINT_EVERY = 100
PLOT_EVERY_STEPS = 5000
CENTERLINE_EVERY_STEPS = 5000
CHECKPOINT_EVERY = 5000
STRUCTURE_EVERY_STEPS = 5000     # Q isosurface image (needs pyvista); 0 = off
Q_ISOLEVEL = None                # fixed Q L^2/U^2, or None = Q_AUTO_PERCENTILE of the positive Q values
Q_AUTO_PERCENTILE = 99.0
Q_COLOR = 'w'                    # 'w' (spanwise velocity) | 'speed'
OUT_DIR = 'cavity_output'
CHECKPOINT = 'cavity_checkpoint.npz'
PROBES = ((0.5, 0.5), (0.8, 0.8), (0.2, 0.2), (0.9, 0.1))   # (x/L, y/L) time-series probes
# Features of the CuPy version that this native version does not implement (must stay off):
ADVECTION_SCHEME = 'eulerian_impulse'
IMPULSE_FORM = 'consistent'
LES_ON = False
WALL_MODEL_ON = False
TWO_D = False


def native_settings(s):
    """Solver settings from a namespace with the names above (this module, or the original solver file)."""
    unsupported = [(s['ADVECTION_SCHEME'] != 'eulerian_impulse', "ADVECTION_SCHEME = 'eulerian_impulse'"),
                   (s['IMPULSE_FORM'] != 'consistent', "IMPULSE_FORM = 'consistent'"),
                   (bool(s['LES_ON']), 'LES_ON = False'), (bool(s['WALL_MODEL_ON']), 'WALL_MODEL_ON = False'),
                   (bool(s['TWO_D']), 'TWO_D = False'), (not s['Z_WALLS'], 'Z_WALLS = True'),
                   (s['LID_PROFILE'] not in ('regularised', 'uniform'), "LID_PROFILE 'regularised' or 'uniform'")]
    bad = [need for wrong, need in unsupported if wrong]
    if bad:
        raise ValueError('The native solver implements the Eulerian impulse scheme only; set ' + ', '.join(bad))
    n, nzc = s['MESH_OPTIONS'][s['MESH']]
    return dict(nx=n, ny=n, nz=int(nzc), lid_shaped=int(s['LID_PROFILE'] == 'regularised'),
                impulse_order=int(s['IMPULSE_ORDER']), adaptive_dt=int(bool(s['ADAPTIVE_DT'])),
                dt_lag=int(bool(s['DT_LAG'])), phi_extrapolate=int(bool(s['PHI_EXTRAPOLATE'])),
                fdm_max_passes=int(s['FDM_MAX_PASSES']), grid_stretch=int(bool(s['GRID_STRETCH'])),
                length=s['L'], u_lid=s['U_LID'], nu=s['NU'], lz=s['LZ_L']*s['L'], lid_ramp_t=s['LID_RAMP_T'],
                reinit_r=s['REINIT_R'], impulse_order4_ramp=s['IMPULSE_ORDER4_RAMP'],
                grid_dx_wall=s['GRID_DX_WALL'], grid_max_aspect=s['GRID_MAX_ASPECT'] or 0.0,
                dt_fixed=s['DT_FIXED'], dt_max=s['DT_MAX'], cfl_target=s['CFL_TARGET'],
                diffusion_limit=s['STRETCH_DIFFUSION_LIMIT'], fdm_rtol=s['FDM_RTOL'], probes=tuple(s['PROBES']))


class CentrelineWindow:
    """Rolling time average of the centreline samples over the last STATS_WINDOW_T time units."""

    def __init__(self, length):
        self.length = length
        self.entries = []              # (begin, end, sample)
        self.total = 0.0               # sum of weight * sample
        self.weight = 0.0

    def add(self, begin, end, sample):
        self.entries.append((begin, end, sample))
        self.total = self.total + (end-begin)*sample
        self.weight += end-begin
        cutoff = end-self.length
        while self.entries and self.entries[0][0] < cutoff:       # drop what fell out of the window
            b, e, v = self.entries.pop(0)
            removed = min(e, cutoff)-b
            self.total = self.total - removed*v
            self.weight -= removed
            if e > cutoff:
                self.entries.insert(0, (cutoff, e, v))
                break

    def mean(self):
        if self.weight <= 0:
            return None
        return self.total/self.weight


def initial_noise(sim, s):
    """v, w noise of the CuPy version's initial_condition (same generator and order)."""
    if s['INIT_PERTURB'] <= 0:
        return None, None
    rng = np.random.RandomState(s['PERTURB_SEED'])
    v = s['INIT_PERTURB']*s['U_LID']*rng.uniform(-1, 1, sim.shapes['v'])
    w = s['INIT_PERTURB']*s['U_LID']*rng.uniform(-1, 1, sim.shapes['w'])
    return v, w


def kick_noise(sim, s, number):
    """Standard-normal noise of kick `number` (1-based), same generator as the CuPy version."""
    rng = np.random.default_rng(s['PERTURB_SEED']+1000+number-1)
    return [rng.standard_normal(sim.shapes[c]) for c in 'uvw']


# =============================================================================
# OUTPUT
# =============================================================================
class Run:
    """What the front end keeps besides the solver: statistics weight, centreline window, probe history."""

    def __init__(self, sim, s):
        self.sim, self.s = sim, s
        self.stats_weight = 0.0
        self.window = CentrelineWindow(s['STATS_WINDOW_T'])
        self.probe_t, self.probe_v = [], []


def save_checkpoint(run, path):
    sim = run.sim
    rep = sim.report()
    data = {k: sim.field(k) for k in ('u', 'v', 'w', 'mx', 'my', 'mz', 'phi')}
    data.update(t=rep['time'], steps=rep['steps'], last_dt=rep['dt'], stat_w=run.stats_weight,
                mesh=np.array([sim.nx, sim.ny, sim.nz]),
                line_times=np.array([(b, e) for b, e, _ in run.window.entries]).reshape(-1, 2),
                line_values=np.array([v for _, _, v in run.window.entries]).reshape(-1, 5, sim.nx),
                probe_t=np.array(run.probe_t), probe_hist=np.array(run.probe_v).reshape(len(run.probe_t), -1))
    for k, v in sim.statistics().items():
        data['stat_'+k] = v
    tmp = path + '.tmp.npz'
    np.savez(tmp, **data)
    os.replace(tmp, path)


def load_checkpoint(run, path):
    sim = run.sim
    d = np.load(path)
    if tuple(d['mesh']) != (sim.nx, sim.ny, sim.nz):
        raise ValueError(f'checkpoint mesh {tuple(d["mesh"])} does not match {(sim.nx, sim.ny, sim.nz)}')
    for k in ('u', 'v', 'w', 'mx', 'my', 'mz', 'phi'):
        sim.set_field(k, d[k])
    sim.set_clock(float(d['t']), int(d['steps']), float(d['last_dt']) if 'last_dt' in d else 0.0)
    if 'stat_u' in d:
        sim.set_statistics({k: d['stat_'+k] for k in ('u', 'v', 'uu', 'vv', 'ww', 'uv')})
        run.stats_weight = float(d['stat_w'])
    if 'line_times' in d:
        for (b, e), v in zip(d['line_times'], d['line_values']):
            run.window.add(b, e, v)
    if 'probe_t' in d:
        run.probe_t, run.probe_v = list(d['probe_t']), list(d['probe_hist'])
    print(f'Restarted from {path}: t={float(d["t"]):.6g}, step={int(d["steps"])}', flush=True)


def write_probes(run, path):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t'] + [f'{c}_{px:g}_{py:g}' for px, py in run.s['PROBES'][:8] for c in 'uv'])
        for t, v in zip(run.probe_t, run.probe_v):
            w.writerow([f'{t:.8g}'] + [f'{x:.6g}' for x in v])


def centerline_values(sim, s):
    """Instantaneous u(y) on x = L/2 and v(x) on y = L/2, on the mid-span plane."""
    u, v = sim.field('u'), sim.field('v')
    n = sim.nx
    k2 = sim.nz//2
    k1 = k2-1 if sim.nz % 2 == 0 else k2
    uc = 0.5*(u[k1, :, n//2]+u[k2, :, n//2])/s['U_LID']
    vc = 0.5*(v[k1, n//2, :]+v[k2, n//2, :])/s['U_LID']
    return uc, vc


def write_centerlines(run, path):
    sim, s = run.sim, run.s
    uc, vc = centerline_values(sim, s)
    mean = run.window.mean()
    y, x = sim.grid['y'][1]/s['L'], sim.grid['x'][1]/s['L']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['y/L', 'u/U (instant, x=L/2)', 'u_mean/U', 'u_rms/U', 'x/L', 'v/U (instant, y=L/2)',
                    'v_mean/U', 'v_rms/U', 'w_rms/U'])
        if mean is not None:
            um, vm, uu, vv, ww = mean
            urms, vrms, wrms = (np.sqrt(np.maximum(a, 0)) for a in (uu-um*um, vv-vm*vm, ww))
        for j in range(len(y)):
            row = [y[j], uc[j]] + ([um[j], urms[j]] if mean is not None else ['', ''])
            row += [x[j], vc[j]] + ([vm[j], vrms[j], wrms[j]] if mean is not None else ['', '', ''])
            w.writerow([f'{q:.7g}' if q != '' else '' for q in row])


def save_fields_plot(run, tag):
    sim, s = run.sim, run.s
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    mp = sim.midplane()
    xf, yf = sim.grid['x'][0]/s['L'], sim.grid['y'][0]/s['L']
    speed = np.sqrt(mp[0]**2+mp[1]**2)/s['U_LID']
    wmax = max(float(np.abs(mp[2]).max())/s['U_LID'], 1e-12)
    om = mp[3]*s['L']/s['U_LID']
    panels = ((speed, '|u_xy| / U (mid-span)', 'viridis', {}),
              (mp[2]/s['U_LID'], 'w / U (mid-span)', 'RdBu_r', dict(vmin=-wmax, vmax=wmax)),
              (om, '|omega| L/U (mid-span)', 'magma',
               dict(norm=LogNorm(vmin=max(float(om.max())*1e-3, 1e-6), vmax=max(float(om.max()), 1e-5)))))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (f, title, cmap, kw) in zip(axes, panels):
        pc = ax.pcolormesh(xf, yf, f, cmap=cmap, shading='flat', **kw)
        ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x/L'); ax.set_ylabel('y/L')
        fig.colorbar(pc, ax=ax, shrink=0.8)
    rep = sim.report()
    fig.suptitle(f'Re={s["RE"]:g}  t={rep["time"]:.3f}  step {rep["steps"]}')
    fig.tight_layout()
    fig.savefig(os.path.join(s['OUT_DIR'], f'fields_{tag}.png'), dpi=110)
    plt.close(fig)


def save_centerline_plot(run, tag):
    sim, s = run.sim, run.s
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    uc, vc = centerline_values(sim, s)
    mean = run.window.mean()
    y, x = sim.grid['y'][1]/s['L'], sim.grid['x'][1]/s['L']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(uc, y, 'k:', label='instant')
    axes[1].plot(x, vc, 'k:', label='instant')
    if mean is not None:
        um, vm, uu, vv, ww = mean
        axes[0].plot(um, y, 'b-', label=f'mean ({run.window.weight:.1f} L/U window)')
        axes[0].plot(np.sqrt(np.maximum(uu-um*um, 0))*10, y, 'r--', label='10 x u_rms')
        axes[1].plot(x, vm, 'b-', label='mean')
        axes[1].plot(x, np.sqrt(np.maximum(vv-vm*vm, 0))*10, 'r--', label='10 x v_rms')
    axes[0].set_xlabel('u/U at x = L/2'); axes[0].set_ylabel('y/L')
    axes[1].set_xlabel('x/L'); axes[1].set_ylabel('v/U at y = L/2')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(s['OUT_DIR'], f'centerlines_{tag}.png'), dpi=110)
    plt.close(fig)


def save_rms_plot(run, tag):
    """Time-averaged mid-span mean speed and rms fluctuations (statistics since STATS_START_T)."""
    sim, s = run.sim, run.s
    w = run.stats_weight
    if w <= 0:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = {k: v/w for k, v in sim.statistics().items()}
    U = s['U_LID']
    fields = ((np.sqrt(m['u']**2+m['v']**2)/U, 'mean |u_xy| / U'),
              (np.sqrt(np.maximum(m['uu']-m['u']**2, 0))/U, 'u_rms / U'),
              (np.sqrt(np.maximum(m['vv']-m['v']**2, 0))/U, 'v_rms / U'),
              (np.sqrt(np.maximum(m['ww'], 0))/U, 'w_rms / U'))
    xf, yf = sim.grid['x'][0]/s['L'], sim.grid['y'][0]/s['L']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (f, title) in zip(axes, fields):
        pc = ax.pcolormesh(xf, yf, f, cmap='viridis', shading='flat')
        ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x/L'); ax.set_ylabel('y/L')
        fig.colorbar(pc, ax=ax, shrink=0.8)
    fig.suptitle(f'mid-span statistics over {w:.2f} L/U (from t = {s["STATS_START_T"]:g})')
    fig.tight_layout()
    fig.savefig(os.path.join(s['OUT_DIR'], f'rms_{tag}.png'), dpi=110)
    plt.close(fig)


def save_q_image(run, tag):
    """Q-criterion isosurface (Q computed on the GPU; contouring and rendering with PyVista on the CPU)."""
    sim, s = run.sim, run.s
    import pyvista as pv
    q, colour = sim.q_volume(colour_w=(s['Q_COLOR'] == 'w'))
    level = s['Q_ISOLEVEL']
    if level is None:
        pos = q[q > 0]
        if pos.size == 0:
            return
        level = float(np.percentile(pos, s['Q_AUTO_PERCENTILE']))
    grid = pv.RectilinearGrid(sim.grid['x'][1]/s['L'], sim.grid['y'][1]/s['L'], sim.grid['z'][1]/s['L'])
    grid.point_data['Q'] = q.ravel(order='C')                  # [z, y, x] C order = x fastest, as VTK expects
    grid.point_data['colour'] = colour.ravel(order='C')
    surf = grid.contour([level], scalars='Q')
    if surf.n_points == 0:
        return
    p = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    if s['Q_COLOR'] == 'w':
        lim = max(float(np.abs(colour).max()), 1e-6)
        p.add_mesh(surf, scalars='colour', cmap='RdBu_r', clim=(-lim, lim), scalar_bar_args={'title': 'w/U'})
    else:
        p.add_mesh(surf, scalars='colour', cmap='viridis', scalar_bar_args={'title': '|u|/U'})
    p.add_mesh(pv.Box(bounds=(0, 1, 0, 1, 0, sim.grid['z'][0][-1]/s['L'])), style='wireframe', color='grey')
    p.add_text(f'Q L^2/U^2 = {level:.3g}   t = {sim.report()["time"]:.2f}', font_size=10)
    p.camera_position = 'iso'
    p.screenshot(os.path.join(s['OUT_DIR'], f'Q_{tag}.png'))
    p.close()


# =============================================================================
# RUN
# =============================================================================
def advance(run):
    """One time step, then (in this order) probes, any kick that is due, and the statistics."""
    sim, s = run.sim, run.s
    old_t = sim.report()['time']
    rep = sim.step()
    t = rep['time']
    run.probe_t.append(t)
    run.probe_v.append(sim.probes())
    if s['KICK_AMP'] > 0:
        for k in range(int(s['KICK_COUNT'])):
            if old_t < s['KICK_T']+k*s['KICK_EVERY'] <= t:
                before = sim.diagnostics()['kinetic_energy']
                sim.kick(kick_noise(sim, s, k+1), s['KICK_AMP'], max(1, int(s['KICK_CELLS'])//2))
                after = sim.diagnostics()['kinetic_energy']
                print(f'[kick {k+1}/{s["KICK_COUNT"]}] t={t:.3f}: added {s["KICK_AMP"]:.3g} U_lid rms smooth noise '
                      f'(~{s["KICK_CELLS"]}-cell eddies), KE {before:.4g} -> {after:.4g}', flush=True)
    if t > s['STATS_START_T']:
        weight = t-max(old_t, s['STATS_START_T'])
        run.stats_weight += weight
        run.window.add(t-weight, t, sim.add_statistics(weight))
    return rep


def main(settings=None):
    s = dict(globals()) if settings is None else settings
    restart = sys.argv[sys.argv.index('--restart')+1] if '--restart' in sys.argv else None
    os.makedirs(s['OUT_DIR'], exist_ok=True)
    sim = Solver(native_settings(s))
    run = Run(sim, s)
    if restart:
        sim.initialize()
        load_checkpoint(run, restart)
    else:
        sim.initialize(*initial_noise(sim, s))
    gx = np.diff(sim.grid['x'][0])
    print(f'Lid-driven cavity (native CUDA C++) Re={s["RE"]:g} mesh={s["MESH"]} ({sim.nx}x{sim.ny}x{sim.nz}), '
          f'NU={s["NU"]:.4g}, wall cell {gx[0]:.4g}, centre cell {gx[len(gx)//2]:.4g}, '
          f'impulse order {s["IMPULSE_ORDER"]}', flush=True)
    hist_path = os.path.join(s['OUT_DIR'], 'history.csv')
    new_file = not (restart and os.path.exists(hist_path))
    hist = open(hist_path, 'w' if new_file else 'a', newline='')
    hw = csv.writer(hist)
    if new_file:
        hw.writerow(['step', 't', 'dt', 'dt_reason', 'KE', 'enstrophy', 'umax', 'div_rms', 'CFL', 'phi_passes',
                     'gauge_ratio', 'gauge_x', 'gauge_y', 'w_rms_mid', 'w_max_mid'])
    wall0 = time.perf_counter()
    rep = sim.report()
    start_steps = rep['steps']
    ckpt = os.path.join(s['OUT_DIR'], s['CHECKPOINT'])
    try:
        while rep['time'] < s['T_END'] and rep['steps'] < s['N_STEPS']:
            rep = advance(run)
            n = rep['steps']
            if n % s['PRINT_EVERY'] == 0:
                d = sim.diagnostics()
                wmid = sim.midplane()[2]
                w_rms = float(np.sqrt(np.mean(np.square(wmid, dtype=np.float64))))/s['U_LID']
                w_max = float(np.abs(wmid).max())/s['U_LID']
                cfl = rep['dt']*d['cfl_rate']
                rate = (n-start_steps)/(time.perf_counter()-wall0)
                print(f'step {n:7d} t={rep["time"]:9.4f} dt={rep["dt"]:.3e} ({rep["dt_reason"]}) CFL={cfl:.3f} '
                      f'KE={d["kinetic_energy"]:.5f} Z={d["enstrophy"]:.3f} umax={d["umax"]:.3f} '
                      f'div={d["div_rms"]:.2e} phi_passes={rep["phi_passes"]} gauge={d["gauge_ratio"]:.2f}'
                      f'@({d["gauge_x"]:.3f},{d["gauge_y"]:.3f}) w_rms(mid)={w_rms:.2e} {rate:.2f} steps/s',
                      flush=True)
                hw.writerow([n, f'{rep["time"]:.8g}', f'{rep["dt"]:.6g}', rep['dt_reason'],
                             f'{d["kinetic_energy"]:.8g}', f'{d["enstrophy"]:.8g}', f'{d["umax"]:.6g}',
                             f'{d["div_rms"]:.3g}', f'{cfl:.4g}', rep['phi_passes'], f'{d["gauge_ratio"]:.5g}',
                             f'{d["gauge_x"]:.4f}', f'{d["gauge_y"]:.4f}', f'{w_rms:.5g}', f'{w_max:.5g}'])
                hist.flush()
            if s['PLOT_EVERY_STEPS'] and n % s['PLOT_EVERY_STEPS'] == 0:
                save_fields_plot(run, f'{n:07d}')
                save_rms_plot(run, f'{n:07d}')
            if s['CENTERLINE_EVERY_STEPS'] and n % s['CENTERLINE_EVERY_STEPS'] == 0:
                write_centerlines(run, os.path.join(s['OUT_DIR'], 'centerlines.csv'))
                save_centerline_plot(run, f'{n:07d}')
            if s['STRUCTURE_EVERY_STEPS'] and n % s['STRUCTURE_EVERY_STEPS'] == 0:
                try:
                    save_q_image(run, f'{n:07d}')
                except Exception as exc:          # an optional image never stops the run
                    print(f'[Q] skipped at step {n}: {type(exc).__name__}: {exc}', flush=True)
            if s['CHECKPOINT_EVERY'] and n % s['CHECKPOINT_EVERY'] == 0:
                save_checkpoint(run, ckpt)
    finally:
        hist.close()
        save_checkpoint(run, ckpt)
        write_probes(run, os.path.join(s['OUT_DIR'], 'probes.csv'))
        write_centerlines(run, os.path.join(s['OUT_DIR'], 'centerlines.csv'))
        save_fields_plot(run, 'final')
        save_centerline_plot(run, 'final')
        save_rms_plot(run, 'final')
        rep = sim.report()
        print(f'Done: t={rep["time"]:.4f}, steps={rep["steps"]}, wall={time.perf_counter()-wall0:.1f}s. '
              f'Outputs in {s["OUT_DIR"]}/', flush=True)
        sim.close()


if __name__ == '__main__':
    main()
