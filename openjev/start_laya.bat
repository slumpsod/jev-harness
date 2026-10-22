@echo off
rem The Laya bridge (a small Jev model trained on the harness's decisions) on 127.0.0.1:33004.
rem GPU: set CUDA_VISIBLE_DEVICES first (about 2 GB). CPU: set LAYA_DEVICE=cpu.
rem The checkpoint: openjev\models\laya-harness, or set LAYA_CHECKPOINT. Use it with config jev.backend: laya.
cd /d "%~dp0"
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
%PY% laya_bridge.py
