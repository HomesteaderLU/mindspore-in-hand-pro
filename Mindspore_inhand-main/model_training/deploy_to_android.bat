@echo off
chcp 65001 >nul
echo ======================================================================
echo 部署模型到 Android 项目
echo ======================================================================
echo.

set MS_FILE=style_classifier_artbench.ms
set ASSETS_DIR=..\custommodel\src\main\assets

echo [1/3] 检查 .ms 文件...
if not exist "%MS_FILE%" (
    echo ❌ 错误: 找不到 %MS_FILE%
    echo.
    echo 请先转换模型:
    echo   python convert_with_python.py
    pause
    exit /b 1
)
echo ✓ 找到 %MS_FILE%
for %%A in ("%MS_FILE%") do echo   文件大小: %%~zA bytes
echo.

echo [2/3] 检查目标目录...
if not exist "%ASSETS_DIR%" (
    echo ⚠️  目录不存在，正在创建...
    mkdir "%ASSETS_DIR%"
)
echo ✓ 目标目录: %ASSETS_DIR%
echo.

echo [3/3] 复制文件...
copy "%MS_FILE%" "%ASSETS_DIR%\" >nul
if %errorlevel% equ 0 (
    echo ✓ 复制成功！
    echo.
    echo 文件位置: %ASSETS_DIR%\%MS_FILE%
) else (
    echo ❌ 复制失败
    pause
    exit /b 1
)
echo.

echo ======================================================================
echo ✅ 部署完成！
echo ======================================================================
echo.
echo 下一步：
echo   1. 在 Android Studio 中打开项目
echo   2. 确保 StyleClassifierExecutor.java 加载正确的模型文件
echo   3. 编译并运行应用
echo.
echo 或者使用命令行编译:
echo   cd ..
echo   gradlew assembleDebug
echo.

pause
