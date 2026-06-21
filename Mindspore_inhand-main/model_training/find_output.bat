@echo off
chcp 65001 >nul
echo ======================================================================
echo 查找转换后的模型文件
echo ======================================================================
echo.

echo [1] 当前目录的 .ms 文件:
dir *.ms 2>nul
if %errorlevel% neq 0 echo   (无)
echo.

echo [2] style_classifier 相关文件:
dir style_classifier* 2>nul
if %errorlevel% neq 0 echo   (无)
echo.

echo [3] 最近修改的文件 (前10个):
dir /O:D /B | more
echo.

echo [4] Converter 目录的 .ms 文件:
dir D:\Desktop\PythonFile\mindspore-lite-2.9.0-win-x64\tools\converter\converter\*.ms 2>nul
if %errorlevel% neq 0 echo   (无)
echo.

pause
