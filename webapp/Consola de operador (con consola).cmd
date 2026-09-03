@echo off
REM Lanzador con consola visible (para ver logs / errores). Ctrl+C o cierra la ventana para salir.
REM La version sin consola es "Consola de operador.vbs".
cd /d "%~dp0.."
".venv\Scripts\python.exe" "webapp\desktop.py"
pause
