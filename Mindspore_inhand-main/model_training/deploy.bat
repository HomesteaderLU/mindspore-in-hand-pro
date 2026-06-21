@echo off
chcp 65001 >nul
REM 一键部署脚本 (Windows 版本)
REM 从模型训练到 Android 部署的自动化流程

echo ================================================
echo   图像风格分类 - 一键部署脚本 (Windows)
echo ================================================
echo.

REM 检查 Python
echo [1/7] 检查 Python 环境...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python
    pause
    exit /b 1
)
echo ✓ Python 已安装: 
python --version
echo.

REM 安装依赖
echo [2/7] 安装 Python 依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 依赖安装失败！
    pause
    exit /b 1
)
echo ✓ 依赖安装完成
echo.

REM 检查数据集
echo [3/7] 检查数据集...
if not exist "..\style_dataset" (
    echo 警告: 数据集目录不存在
    set /p response="是否要准备数据集？(y/n): "
    if /i "%response%"=="y" (
        python prepare_dataset.py
    ) else (
        echo 跳过数据集准备，使用模拟数据继续...
    )
) else (
    echo ✓ 数据集已存在
)
echo.

REM 训练模型
echo [4/7] 训练模型...
if exist "..\style_dataset" (
    python train_style_classifier.py
    if %errorlevel% neq 0 (
        echo 训练失败！
        pause
        exit /b 1
    )
    echo ✓ 训练完成
) else (
    echo 跳过训练（无数据集）
)
echo.

REM 转换模型
echo [5/7] 转换模型为 MS 格式...
if exist "style_classifier.air" (
    python convert_model.py --input style_classifier.air --output style_classifier.ms --android-path ..\
    if %errorlevel% neq 0 (
        echo 转换失败！
        pause
        exit /b 1
    )
    echo ✓ 模型转换完成
) else (
    echo 跳过转换（AIR 文件不存在）
    echo 提示：可以手动运行：
    echo   python convert_model.py --input style_classifier.air --output style_classifier.ms --android-path ..\
)
echo.

REM 更新 Android 代码
echo [6/7] 配置 Android 项目...
echo 请手动完成以下步骤：
echo   1. 打开 StyleClassifierExecutor.java
echo   2. 取消注释 MindSpore Lite API 调用（第 78-106 行）
echo   3. 取消注释推理代码（第 167-205 行）
echo   4. 打开 StyleClassifierFragment.java
echo   5. 取消注释模型加载代码（第 90-113 行）
echo.
pause
echo ✓ 配置说明已显示
echo.

REM 编译 Android 项目
echo [7/7] 编译 Android 项目...
cd ..

if not exist "gradlew.bat" (
    echo 错误: 找不到 gradlew.bat
    pause
    exit /b 1
)

call gradlew.bat clean
call gradlew.bat assembleDebug

if %errorlevel% equ 0 (
    echo ✓ 编译成功！
    echo.
    echo APK 位置: app\build\outputs\apk\debug\app-debug.apk
    echo.
    set /p response="是否安装到设备？(y/n): "
    if /i "%response%"=="y" (
        adb install -r app\build\outputs\apk\debug\app-debug.apk
        echo ✓ 安装完成！
    )
) else (
    echo 编译失败！
    pause
    exit /b 1
)

echo.
echo ================================================
echo   ✅ 部署完成！
echo ================================================
echo.
echo 下一步：
echo   1. 在设备上打开应用
echo   2. 点击底部 '风格分类' Tab
echo   3. 选择图片进行测试
echo.
echo 如有问题，请查看日志：
echo   adb logcat ^| findstr StyleClassifier
echo.
pause
