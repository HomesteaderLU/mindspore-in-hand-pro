@echo off
chcp 65001 >nul
REM 快速修复 ArtBench-10 数据集问题 (Windows)

echo ================================================
echo   ArtBench-10 数据集快速修复工具
echo ================================================
echo.

REM 步骤 1: 检查当前状态
echo [1/4] 检查当前状态...
if exist "artbench10_data" (
    echo 发现现有数据集目录
    
    REM 检查目录结构
    if exist "artbench10_data\artbench-10-batches-py" (
        echo ✓ 目录结构正确
        
        REM 检查文件
        if exist "artbench10_data\artbench-10-batches-py\data_batch_1" (
            echo ✓ 数据文件存在
            
            echo.
            echo ✅ 数据集看起来正常！
            echo 尝试运行训练脚本...
            echo.
            python train_with_artbench10.py
            exit /b 0
        )
    )
    
    echo ⚠️  目录结构可能不正确
) else (
    echo 未发现数据集目录
)

echo.

REM 步骤 2: 询问用户
echo [2/4] 选择操作:
echo 1. 重新下载数据集（推荐）
echo 2. 仅修复目录结构（如果文件已存在）
echo 3. 退出
echo.
set /p choice="请选择 (1/2/3): "

if "%choice%"=="1" goto redownload
if "%choice%"=="2" goto fixstructure
if "%choice%"=="3" goto exit
goto invalid

:redownload
echo.
echo [3/4] 清理旧数据...
if exist "artbench10_data" rmdir /s /q artbench10_data
if exist "artbench-10-python.tar.gz" del artbench-10-python.tar.gz
echo ✓ 清理完成

echo.
echo [4/4] 重新下载...
python download_artbench10.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ 下载完成！
    echo.
    echo 下一步：训练模型
    echo   python train_with_artbench10.py
) else (
    echo.
    echo ❌ 下载失败
    echo 请查看错误信息并重试
    pause
    exit /b 1
)
goto end

:fixstructure
echo.
echo [3/4] 修复目录结构...

REM 检查文件是否在根目录
if exist "artbench10_data\data_batch_1" (
    echo 发现文件在根目录，正在移动...
    
    if not exist "artbench10_data\artbench-10-batches-py" mkdir artbench10_data\artbench-10-batches-py
    
    REM 移动所有批次文件
    move artbench10_data\data_batch_* artbench10_data\artbench-10-batches-py\ >nul 2>&1
    move artbench10_data\test_batch artbench10_data\artbench-10-batches-py\ >nul 2>&1
    move artbench10_data\meta artbench10_data\artbench-10-batches-py\ >nul 2>&1
    
    echo ✓ 目录结构已修复
) else (
    echo ❌ 找不到数据文件
    echo 可能需要重新下载
    pause
    exit /b 1
)

echo.
echo [4/4] 验证修复...
python download_artbench10.py --verify

if %errorlevel% equ 0 (
    echo.
    echo ✅ 修复成功！
    echo.
    echo 下一步：训练模型
    echo   python train_with_artbench10.py
) else (
    echo.
    echo ❌ 验证失败
    echo 建议重新下载数据集
    pause
    exit /b 1
)
goto end

:exit
echo 退出
exit /b 0

:invalid
echo 无效选择
pause
exit /b 1

:end
echo.
echo ================================================
echo   ✅ 完成！
echo ================================================
pause
