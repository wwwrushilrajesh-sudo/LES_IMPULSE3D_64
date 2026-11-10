"""Checks of the compiled solver on your GPU. Run after building:

    python validate_gpu.py                                    # quick self-checks on the 32x32x8 TEST mesh
    python validate_gpu.py --reference my_original_solver.py  # + compare step by step with the original Python

The reference runs on the GPU if CuPy is installed, otherwise on the CPU (slower)."""
import importlib.util
import sys
import numpy as np

import main as front
from python_api import Solver

STEPS = 60
all_ok = True


def check(name, ok, detail=''):
    global all_ok
    all_ok &= bool(ok)
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}', flush=True)


def settings(**changes):
    s = dict(vars(front))
    s.update(MESH='TEST')
    s.update(changes)
    return s


def new_run(s):
    sim = Solver(front.native_settings(s))
    sim.initialize(*front.initial_noise(sim, s))
    return front.Run(sim, s)


# 1. a short run stays finite and divergence-free
run = new_run(settings(LID_RAMP_T=0.0))
for _ in range(STEPS):
    rep = front.advance(run)
d = run.sim.diagnostics()
f = {k: run.sim.field(k) for k in ('u', 'v', 'w', 'mx', 'my', 'mz', 'phi')}
check(f'finite after {STEPS} steps', all(np.isfinite(a).all() for a in f.values()))
check('divergence-free', d['div_rms'] < 1e-5, f'div_rms = {d["div_rms"]:.2e}, KE = {d["kinetic_energy"]:.4g}, '
      f't = {rep["time"]:.4g}, pressure passes = {rep["phi_passes"]}')
check('no flow through the walls', np.abs(f['u'][..., [0, -1]]).max() == 0
      and np.abs(f['v'][:, [0, -1]]).max() == 0 and np.abs(f['w'][[0, -1]]).max() == 0)
run.sim.close()

# 2. CFL-limited time step
run = new_run(settings(LID_RAMP_T=0.0, DT_MAX=1.0, DT_LAG=False))
for _ in range(20):
    front.advance(run)
rate = run.sim.diagnostics()['cfl_rate']
rep = front.advance(run)
check('CFL-limited dt', rep['dt_reason'] != 'CFL' or abs(rep['dt']*rate - 0.4) < 1e-4,
      f'dt = {rep["dt"]:.4g} ({rep["dt_reason"]}), dt * rate = {rep["dt"]*rate:.4f}')
run.sim.close()

# 3. optional: the original solver with the same settings and noise
if '--reference' in sys.argv:
    path = sys.argv[sys.argv.index('--reference')+1]
    spec = importlib.util.spec_from_file_location('reference', path)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    try:
        import cupy  # noqa: F401
        backend = 'gpu'
    except Exception:
        backend = 'cpu'
    ref_mod.MESH, ref_mod.BACKEND = 'TEST', backend
    rs = dict(vars(ref_mod))
    rs['DT_LAG'] = bool(rs.get('DT_LAG', False)) and backend == 'gpu'     # the original CPU path never lags dt
    run = new_run(rs)
    ref = ref_mod.Cavity3D(backend)
    for _ in range(STEPS):
        ref.step()
        front.advance(run)
    worst = 0.0
    for name in ('u', 'v', 'w', 'mx', 'my', 'mz', 'phi'):
        a = run.sim.field(name).astype(np.float64)
        b = np.asarray(ref.host(getattr(ref, name)), dtype=np.float64)
        rel = float(np.abs(a-b).max()/max(np.abs(b).max(), 1e-30))
        worst = max(worst, rel)
        print(f'      {name}: largest relative difference {rel:.2e}')
    check(f'matches the original ({backend}) after {STEPS} steps', worst < 1e-4, f'worst {worst:.2e}')
    run.sim.close()

print('\nALL CHECKS PASSED' if all_ok else '\nSOME CHECKS FAILED', flush=True)
sys.exit(0 if all_ok else 1)
