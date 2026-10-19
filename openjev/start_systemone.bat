@echo off
rem The System One adapter jev-harness talks to (127.0.0.1:33001), forwarding to the SemIf bridge (start_semif.bat).
cd /d "%~dp0"
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
%PY% systemone.py
