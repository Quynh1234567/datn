@echo off
rem Nhap dup file nay de mo Demo 1 Launcher
cd /d "%~dp0"
python scripts\demo1_launcher.py
if errorlevel 1 pause
