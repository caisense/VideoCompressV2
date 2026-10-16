@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_video_quality.ps1" %*
if errorlevel 1 pause
