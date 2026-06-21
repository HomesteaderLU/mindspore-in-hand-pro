@echo off
chcp 65001 >nul
echo Testing MindSpore Lite Converter...
echo.

set CONVERTER=D:\Desktop\PythonFile\mindspore-lite-2.9.0-win-x64\tools\converter\converter\converter_lite.exe

echo Running converter with debug output:
echo ======================================================================
"%CONVERTER%" --help
echo.
echo ======================================================================
pause
