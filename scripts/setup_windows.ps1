$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe check_environment.py
& .\.venv\Scripts\python.exe -m pytest -q