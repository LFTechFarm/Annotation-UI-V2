@echo off
REM Simple launcher for the Tkinter app
SETLOCAL
pushd %~dp0

if not exist "%~dp0requirements.txt" goto RUN
echo Installing dependencies (if needed)...
python -m pip install -r requirements.txt >NUL 2>&1

:RUN
echo Launching Yolo_Annotations_2.0.py
python Yolo_Annotations_2.0.py
popd
ENDLOCAL
