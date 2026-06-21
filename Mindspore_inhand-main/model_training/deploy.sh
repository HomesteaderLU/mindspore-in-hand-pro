#!/bin/bash

# 一键部署脚本
# 从模型训练到 Android 部署的自动化流程

set -e  # 遇到错误立即退出

echo "================================================"
echo "  图像风格分类 - 一键部署脚本"
echo "================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Python
echo -e "${YELLOW}[1/7] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安装: $(python3 --version)${NC}"
echo ""

# 安装依赖
echo -e "${YELLOW}[2/7] 安装 Python 依赖...${NC}"
pip3 install -r requirements.txt
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 检查数据集
echo -e "${YELLOW}[3/7] 检查数据集...${NC}"
if [ ! -d "../style_dataset" ]; then
    echo -e "${RED}警告: 数据集目录不存在${NC}"
    echo "是否要准备数据集？(y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        python3 prepare_dataset.py
    else
        echo -e "${YELLOW}跳过数据集准备，使用模拟数据继续...${NC}"
    fi
else
    echo -e "${GREEN}✓ 数据集已存在${NC}"
fi
echo ""

# 训练模型
echo -e "${YELLOW}[4/7] 训练模型...${NC}"
if [ -d "../style_dataset" ]; then
    python3 train_style_classifier.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}训练失败！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 训练完成${NC}"
else
    echo -e "${YELLOW}跳过训练（无数据集）${NC}"
fi
echo ""

# 转换模型
echo -e "${YELLOW}[5/7] 转换模型为 MS 格式...${NC}"
if [ -f "style_classifier.air" ]; then
    python3 convert_model.py \
        --input style_classifier.air \
        --output style_classifier.ms \
        --android-path ../
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}转换失败！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 模型转换完成${NC}"
else
    echo -e "${YELLOW}跳过转换（AIR 文件不存在）${NC}"
    echo "提示：可以手动运行："
    echo "  python3 convert_model.py --input style_classifier.air --output style_classifier.ms --android-path ../"
fi
echo ""

# 更新 Android 代码
echo -e "${YELLOW}[6/7] 配置 Android 项目...${NC}"
echo "请手动完成以下步骤："
echo "  1. 打开 StyleClassifierExecutor.java"
echo "  2. 取消注释 MindSpore Lite API 调用（第 78-106 行）"
echo "  3. 取消注释推理代码（第 167-205 行）"
echo "  4. 打开 StyleClassifierFragment.java"
echo "  5. 取消注释模型加载代码（第 90-113 行）"
echo ""
read -p "按回车键继续..."
echo -e "${GREEN}✓ 配置说明已显示${NC}"
echo ""

# 编译 Android 项目
echo -e "${YELLOW}[7/7] 编译 Android 项目...${NC}"
cd ..

if [ ! -f "gradlew" ]; then
    echo -e "${RED}错误: 找不到 gradlew${NC}"
    exit 1
fi

chmod +x gradlew
./gradlew clean
./gradlew assembleDebug

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 编译成功！${NC}"
    echo ""
    echo "APK 位置: app/build/outputs/apk/debug/app-debug.apk"
    echo ""
    echo "是否安装到设备？(y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        adb install -r app/build/outputs/apk/debug/app-debug.apk
        echo -e "${GREEN}✓ 安装完成！${NC}"
    fi
else
    echo -e "${RED}编译失败！${NC}"
    exit 1
fi

echo ""
echo "================================================"
echo -e "${GREEN}  ✅ 部署完成！${NC}"
echo "================================================"
echo ""
echo "下一步："
echo "  1. 在设备上打开应用"
echo "  2. 点击底部 '风格分类' Tab"
echo "  3. 选择图片进行测试"
echo ""
echo "如有问题，请查看日志："
echo "  adb logcat | grep StyleClassifier"
echo ""
