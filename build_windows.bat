@echo off
rem Builds cavity_solver.dll from backend.cu. Run in an "x64 Native Tools Command Prompt for VS".
cd /d "%~dp0"
nvcc -O2 -std=c++17 --fmad=false -shared -Xcompiler "/MD /EHsc" backend.cu -o cavity_solver.dll -lcublas
if errorlevel 1 exit /b 1
echo Built cavity_solver.dll. Next: python validate_gpu.py, then run main.py (Spyder: F5).
