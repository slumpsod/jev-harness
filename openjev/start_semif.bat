@echo off
rem The SemIf bridge (the local Jev model) on 127.0.0.1:33003. Pick the GPU: set CUDA_VISIBLE_DEVICES=0 first.
cd /d "%~dp0"
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
%PY% semif_bridge.py
