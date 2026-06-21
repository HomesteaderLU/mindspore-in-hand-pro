#!/bin/bash

# 快速修复 ArtBench-10 数据集问题

echo "================================================"
echo "  ArtBench-10 数据集快速修复工具"
echo "================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步骤 1: 检查当前状态
echo -e "${YELLOW}[1/4] 检查当前状态...${NC}"
if [ -d "artbench10_data" ]; then
    echo "发现现有数据集目录"
    
    # 检查目录结构
    if [ -d "artbench10_data/artbench-10-batches-py" ]; then
        echo -e "${GREEN}✓ 目录结构正确${NC}"
        
        # 检查文件
        if [ -f "artbench10_data/artbench-10-batches-py/data_batch_1" ]; then
            echo -e "${GREEN}✓ 数据文件存在${NC}"
            
            # 验证大小
            size=$(du -sh artbench10_data | cut -f1)
            echo "  数据集大小: $size"
            
            echo ""
            echo -e "${GREEN}✅ 数据集看起来正常！${NC}"
            echo "尝试运行训练脚本..."
            echo ""
            python train_with_artbench10.py
            exit 0
        fi
    fi
    
    echo -e "${YELLOW}⚠️  目录结构可能不正确${NC}"
else
    echo "未发现数据集目录"
fi

echo ""

# 步骤 2: 询问用户
echo -e "${YELLOW}[2/4] 选择操作:${NC}"
echo "1. 重新下载数据集（推荐）"
echo "2. 仅修复目录结构（如果文件已存在）"
echo "3. 退出"
echo ""
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}[3/4] 清理旧数据...${NC}"
        rm -rf artbench10_data
        rm -f artbench-10-python.tar.gz
        echo -e "${GREEN}✓ 清理完成${NC}"
        
        echo ""
        echo -e "${YELLOW}[4/4] 重新下载...${NC}"
        python download_artbench10.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 下载完成！${NC}"
            echo ""
            echo "下一步：训练模型"
            echo "  python train_with_artbench10.py"
        else
            echo ""
            echo -e "${RED}❌ 下载失败${NC}"
            echo "请查看错误信息并重试"
            exit 1
        fi
        ;;
    
    2)
        echo ""
        echo -e "${YELLOW}[3/4] 修复目录结构...${NC}"
        
        # 检查文件是否在根目录
        if [ -f "artbench10_data/data_batch_1" ]; then
            echo "发现文件在根目录，正在移动..."
            
            mkdir -p artbench10_data/artbench-10-batches-py
            
            # 移动所有批次文件
            mv artbench10_data/data_batch_* artbench10_data/artbench-10-batches-py/ 2>/dev/null
            mv artbench10_data/test_batch artbench10_data/artbench-10-batches-py/ 2>/dev/null
            mv artbench10_data/meta artbench10_data/artbench-10-batches-py/ 2>/dev/null
            
            echo -e "${GREEN}✓ 目录结构已修复${NC}"
        else
            echo -e "${RED}❌ 找不到数据文件${NC}"
            echo "可能需要重新下载"
            exit 1
        fi
        
        echo ""
        echo -e "${YELLOW}[4/4] 验证修复...${NC}"
        python download_artbench10.py --verify
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 修复成功！${NC}"
            echo ""
            echo "下一步：训练模型"
            echo "  python train_with_artbench10.py"
        else
            echo ""
            echo -e "${RED}❌ 验证失败${NC}"
            echo "建议重新下载数据集"
            exit 1
        fi
        ;;
    
    3)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo -e "${GREEN}  ✅ 完成！${NC}"
echo "================================================"
