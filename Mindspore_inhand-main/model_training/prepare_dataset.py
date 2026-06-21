"""
数据集准备工具
用于下载、整理和预处理图像风格分类数据集
"""

import os
import shutil
from pathlib import Path
import requests
from PIL import Image
import random


# ==================== 配置 ====================

# 目标目录结构
DATASET_ROOT = "./style_dataset"
TRAIN_RATIO = 0.8  # 训练集比例

# 风格类别
STYLES = [
    "oil_painting",
    "watercolor", 
    "sketch",
    "realistic",
    "abstract",
    "impressionism",
    "modern_art",
    "classical",
    "pop_art",
    "minimalism"
]

# 每个类别的目标图片数量
TARGET_IMAGES_PER_CLASS = 1000


def create_directory_structure():
    """
    创建数据集目录结构
    """
    print("创建目录结构...")
    
    for split in ["train", "val"]:
        for style in STYLES:
            path = os.path.join(DATASET_ROOT, split, style)
            os.makedirs(path, exist_ok=True)
            print(f"  ✓ {path}")
    
    print("目录结构创建完成！\n")


def organize_images(source_dir):
    """
    将图片按类别组织到训练集和验证集
    
    Args:
        source_dir: 源图片目录，应该包含以风格命名的子文件夹
    """
    print(f"从 {source_dir} 组织图片...\n")
    
    for style in STYLES:
        source_style_dir = os.path.join(source_dir, style)
        
        if not os.path.exists(source_style_dir):
            print(f"⚠️  跳过: {style} (目录不存在)")
            continue
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(source_style_dir).glob(ext))
        
        print(f"处理 {style}: 找到 {len(image_files)} 张图片")
        
        if len(image_files) == 0:
            continue
        
        # 随机打乱
        random.shuffle(image_files)
        
        # 分割训练集和验证集
        split_idx = int(len(image_files) * TRAIN_RATIO)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # 复制文件到目标目录
        for img_file in train_files:
            dest = os.path.join(DATASET_ROOT, "train", style, img_file.name)
            shutil.copy2(img_file, dest)
        
        for img_file in val_files:
            dest = os.path.join(DATASET_ROOT, "val", style, img_file.name)
            shutil.copy2(img_file, dest)
        
        print(f"  ✓ 训练集: {len(train_files)}, 验证集: {len(val_files)}\n")
    
    print("图片组织完成！\n")


def validate_images():
    """
    验证数据集中的图片是否可以正常读取
    """
    print("验证图片完整性...\n")
    
    total_valid = 0
    total_invalid = 0
    
    for split in ["train", "val"]:
        for style in STYLES:
            style_dir = os.path.join(DATASET_ROOT, split, style)
            
            if not os.path.exists(style_dir):
                continue
            
            image_files = list(Path(style_dir).glob('*'))
            
            for img_file in image_files:
                try:
                    # 尝试打开图片
                    img = Image.open(img_file)
                    img.verify()  # 验证图片完整性
                    
                    # 重新打开（verify 后会关闭）
                    img = Image.open(img_file)
                    
                    # 检查是否为 RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(img_file)
                    
                    total_valid += 1
                    
                except Exception as e:
                    print(f"  ✗ 无效图片: {img_file}")
                    print(f"    错误: {str(e)}")
                    total_invalid += 1
                    
                    # 删除无效图片
                    try:
                        os.remove(img_file)
                    except:
                        pass
    
    print(f"\n验证完成:")
    print(f"  ✓ 有效图片: {total_valid}")
    print(f"  ✗ 无效图片: {total_invalid}\n")


def print_statistics():
    """
    打印数据集统计信息
    """
    print("=" * 60)
    print("数据集统计信息")
    print("=" * 60 + "\n")
    
    for split in ["train", "val"]:
        print(f"{split.upper()} 集:")
        total = 0
        
        for style in STYLES:
            style_dir = os.path.join(DATASET_ROOT, split, style)
            
            if os.path.exists(style_dir):
                count = len(list(Path(style_dir).glob('*')))
                total += count
                print(f"  {style:20s}: {count:5d} 张")
        
        print(f"  {'总计':20s}: {total:5d} 张\n")


def download_sample_data():
    """
    下载示例数据集（可选）
    
    这里提供一个简单的示例，实际使用时应该从真实数据源下载
    """
    print("注意：此函数仅提供示例")
    print("实际使用时，请从以下来源获取数据：")
    print("  1. WikiArt 数据集: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset")
    print("  2. Kaggle: https://www.kaggle.com/datasets/search?q=art+style")
    print("  3. 自行收集和标注\n")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("图像风格分类数据集准备工具")
    print("=" * 60 + "\n")
    
    # 步骤 1: 创建目录结构
    create_directory_structure()
    
    # 步骤 2: 如果有源数据，组织图片
    source_dir = "./raw_images"  # 修改为你的源图片目录
    if os.path.exists(source_dir):
        organize_images(source_dir)
    else:
        print(f"⚠️  未找到源图片目录: {source_dir}")
        print("请将图片按风格分类放在此目录中，或手动准备数据集\n")
        download_sample_data()
        return
    
    # 步骤 3: 验证图片
    validate_images()
    
    # 步骤 4: 打印统计信息
    print_statistics()
    
    print("=" * 60)
    print("✅ 数据集准备完成！")
    print("=" * 60)
    print(f"\n数据集位置: {DATASET_ROOT}")
    print("\n下一步：运行训练脚本")
    print("  python train_style_classifier.py\n")


if __name__ == "__main__":
    main()
