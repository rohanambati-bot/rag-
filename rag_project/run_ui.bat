@echo off
REM Launches the Streamlit UI on Windows
setlocal
if exist .venv-2\Scripts\python.exe (
    set PY=.venv-2\Scripts\python.exe
) else if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else if exist .venv-1\Scripts\python.exe (
    set PY=.venv-1\Scripts\python.exe
) else (
    set PY=python
)
echo Using interpreter: %PY%
%PY% -m streamlit run app.py
endlocal
