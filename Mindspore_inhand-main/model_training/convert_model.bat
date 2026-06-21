@echo off
chcp 65001 >nul
echo ======================================================================
echo MindSpore Lite 模型转换工具
echo ======================================================================
echo.

set CONVERTER=D:\Desktop\PythonFile\mindspore-lite-2.9.0-win-x64\tools\converter\converter\converter_lite.exe
set INPUT_MODEL=style_classifier_artbench.mindir
set OUTPUT_MODEL=style_classifier

echo [1/3] 检查转换器...
if not exist "%CONVERTER%" (
    echo ❌ 错误: 找不到转换器
    echo    路径: %CONVERTER%
    echo.
    echo 请检查 MindSpore Lite 是否正确安装
    pause
    exit /b 1
)
echo ✓ 转换器存在

echo.
echo [2/3] 检查输入模型...
if not exist "%INPUT_MODEL%" (
    echo ❌ 错误: 找不到输入模型文件
    echo    文件: %INPUT_MODEL%
    echo.
    echo 请先运行训练脚本生成 MINDIR 文件:
    echo   python train_with_artbench10.py
    pause
    exit /b 1
)
echo ✓ 输入模型存在

echo.
echo [3/3] 开始转换...
echo ----------------------------------------------------------------------
"%CONVERTER%" --fmk=MINDIR --modelFile=%INPUT_MODEL% --outputFile=%OUTPUT_MODEL% > convert_log.txt 2>&1

type convert_log.txt

if %errorlevel% equ 0 (
    echo.
    
    REM 检查输出文件
    if exist "%OUTPUT_MODEL%.ms" (
        echo ======================================================================
        echo ✓ 转换成功！
        echo ======================================================================
        echo.
        echo 输出文件: %OUTPUT_MODEL%.ms
        for %%A in ("%OUTPUT_MODEL%.ms") do echo 文件大小: %%~zA bytes
        echo.
        echo 下一步：
        echo   1. 将 %OUTPUT_MODEL%.ms 复制到 Android 项目的 assets 目录
        echo   2. 编译并运行 Android 应用
        echo.
    ) else if exist "%OUTPUT_MODEL%" (
        echo ======================================================================
        echo ✓ 转换成功（无扩展名）！
        echo ======================================================================
        echo.
        echo 输出文件: %OUTPUT_MODEL%
        for %%A in ("%OUTPUT_MODEL%") do echo 文件大小: %%~zA bytes
        echo.
        echo 注意: 文件没有 .ms 扩展名，可能需要手动重命名
        echo   ren %OUTPUT_MODEL% %OUTPUT_MODEL%.ms
        echo.
    ) else (
        echo ======================================================================
        echo ⚠️  转换完成但未找到输出文件
        echo ======================================================================
        echo.
        echo 正在搜索生成的文件...
        dir /b *.ms 2>nul
        dir /b %OUTPUT_MODEL%* 2>nul
        echo.
        echo 请检查上述列出的文件
        echo.
    )
) else (
    echo.
    echo ======================================================================
    echo ❌ 转换失败！
    echo ======================================================================
    echo.
    echo 可能的原因：
    echo   1. MindSpore Lite 版本与训练版本不兼容
    echo   2. MINDIR 文件损坏或不完整
    echo   3. 磁盘空间不足
    echo.
    echo 建议：
    echo   - 检查 MindSpore Lite 版本是否与训练时使用的版本一致
    echo   - 重新运行训练脚本生成 MINDIR 文件
    echo   - 查看上面的错误信息进行排查
    echo.
)

pause
