# Lid-driven cavity LES — Python front end, CUDA C++ solver in one file

| File | What it is |
| --- | --- |
| `backend.cu` | The whole solver (GPU kernels + time step + pressure solve). Read it top to bottom: Part 1 helpers, Part 2 GPU kernels, Part 3 the time step (`step()`), Part 4 the functions Python calls |
| `main.py` | Settings at the top, the run loop, kicks, statistics, CSV files, plots, checkpoints |
| `python_api.py` | Connects Python to the compiled `backend.cu`; no maths |
| `validate_gpu.py` | Checks after building; can compare with your original Python solver |
| `build_windows.bat` | Compiles `backend.cu` into `cavity_solver.dll` |

## Steps (Windows)

1. Install the CUDA Toolkit and the Visual Studio C++ Build Tools.
2. Open **x64 Native Tools Command Prompt for VS**, go to this folder, run `build_windows.bat`.
3. `python -m pip install -r requirements.txt`
4. `python validate_gpu.py --reference path\to\your_original_solver.py`
5. Open `main.py` in Spyder, edit the settings, press F5.

Rebuild (step 2) only after editing `backend.cu`. Continue a run with `python main.py --restart cavity_output/cavity_checkpoint.npz`.

## What it solves

Your default scheme: Eulerian impulse, consistent form, 4th- (or 2nd-) order Lamb vector, SSP-RK3, stretched or
uniform grid, walls on all six sides, regularised or uniform lid, adaptive or fixed dt, initial noise and kicks,
mid-span mean/RMS statistics, probes. Not included: LES, the wall model, the other advection schemes, periodic z,
2-D runs and the energy budget. `main.py` stops with a message if one of those is switched on.


