"""
使用 PyTorch 加载 ArtBench-10，然后转换为 MindSpore 数据集

这个方法可以绕过 MindSpore Cifar10Dataset 的格式限制
"""

import os
import numpy as np
import pickle
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.dataset.vision as vision


def load_cifar_batch(filename):
    """
    加载单个批次文件（支持多种格式）
    
    Args:
        filename: 批次文件路径
    
    Returns:
        images: numpy array, shape (N, 32, 32, 3)
        labels: numpy array, shape (N,)
    """
    with open(filename, 'rb') as f:
        # ArtBench-10 使用字符串键名（不是 bytes）
        dict_data = pickle.load(f, encoding='latin1')
        
        # ArtBench-10 使用 'data' 和 'labels' 键（字符串）
        if 'data' not in dict_data or 'labels' not in dict_data:
            raise KeyError(
                f"❌ ArtBench-10 格式错误！\n"
                f"   文件: {filename}\n"
                f"   期望键: 'data', 'labels'\n"
                f"   实际键: {list(dict_data.keys())}"
            )
        
        # 提取数据
        images = dict_data['data']
        labels = dict_data['labels']
        
        # 转换为 numpy 数组
        if not isinstance(images, np.ndarray):
            images = np.array(images)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # ArtBench-10 的数据形状可能是 (N, 3072) 或 (N, 3, 32, 32)
        if images.ndim == 2:
            # (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        elif images.ndim == 4 and images.shape[1] == 3:
            # (N, 3, 32, 32) -> (N, 32, 32, 3)
            images = images.transpose(0, 2, 3, 1)
        
        print(f"  ✓ 加载成功: {len(images)} 张图片, 形状: {images.shape}")
        
        return images, labels


def load_artbench10(data_dir, usage='train'):
    """
    加载 ArtBench-10 数据集
    
    Args:
        data_dir: 数据集目录（包含 data_batch_* 文件的目录）
        usage: 'train' 或 'test'
    
    Returns:
        images: numpy array
        labels: numpy array
    """
    
    print(f"加载 ArtBench-10 {usage} 集...")
    
    all_images = []
    all_labels = []
    
    if usage == 'train':
        # ArtBench-10 可能有不同的文件组织方式
        # 先检查是否有标准的 5 个批次文件
        batch_files = []
        for i in range(1, 6):
            batch_file = os.path.join(data_dir, f'data_batch_{i}')
            if os.path.exists(batch_file):
                batch_files.append(batch_file)
        
        # 如果找不到标准批次，尝试加载所有 data_batch 文件
        if not batch_files:
            import glob
            batch_files = sorted(glob.glob(os.path.join(data_dir, 'data_batch_*')))
        
        if not batch_files:
            raise FileNotFoundError(f"在 {data_dir} 中找不到任何训练批次文件")
        
        print(f"  找到 {len(batch_files)} 个批次文件")
        
        # 加载所有批次
        for batch_file in batch_files:
            print(f"  加载 {os.path.basename(batch_file)}...")
            images, labels = load_cifar_batch(batch_file)
            all_images.append(images)
            all_labels.append(labels)
            print(f"    ✓ {len(images)} 张图片")
    else:
        # 加载测试批次
        batch_file = os.path.join(data_dir, 'test_batch')
        if os.path.exists(batch_file):
            print(f"  加载 {os.path.basename(batch_file)}...")
            images, labels = load_cifar_batch(batch_file)
            all_images.append(images)
            all_labels.append(labels)
            print(f"    ✓ {len(images)} 张图片")
        else:
            raise FileNotFoundError(f"找不到测试批次文件: {batch_file}")
    
    if all_images:
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"✓ 加载完成: {len(images)} 张图片")
        print(f"  图像形状: {images.shape}")
        print(f"  标签范围: [{labels.min()}, {labels.max()}]")
        return images, labels
    else:
        raise FileNotFoundError("未找到任何批次文件")


def create_generator_dataset(images, labels, transform=None):
    """
    创建生成器数据集
    
    Args:
        images: numpy array, shape (N, H, W, C)
        labels: numpy array, shape (N,)
        transform: 可选的变换函数
    
    Returns:
        MindSpore Dataset
    """
    
    def generator():
        for img, lbl in zip(images, labels):
            yield (img, np.array(lbl, dtype=np.int32))
    
    # 创建数据集
    dataset = ds.GeneratorDataset(generator, column_names=['image', 'label'])
    
    # 应用变换
    if transform:
        dataset = dataset.map(operations=transform, input_columns=['image'])
    
    return dataset


def create_artbench10_mindspore_dataset(data_dir, batch_size=32, is_training=True, image_size=224, use_enhanced_augment=False):
    """
    创建 ArtBench-10 MindSpore 数据集（完整流程）
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        is_training: 是否为训练模式
        image_size: 输出图像尺寸
        use_enhanced_augment: 是否使用增强数据增强
    
    Returns:
        MindSpore Dataset
    """
    
    # 确定正确的数据目录
    cifar_dir = os.path.join(data_dir, "artbench-10-batches-py")
    if not os.path.exists(cifar_dir):
        if os.path.exists(os.path.join(data_dir, "data_batch_1")):
            cifar_dir = data_dir
        else:
            raise FileNotFoundError(
                f"找不到 ArtBench-10 数据文件。\n"
                f"请检查目录结构或使用 check_dataset.py 诊断"
            )
    
    print(f"使用数据目录: {cifar_dir}")
    
    # 加载数据
    usage = "train" if is_training else "test"
    images, labels = load_artbench10(cifar_dir, usage)
    
    # 定义数据增强（注意：数据已经是 numpy 数组，不需要 Decode）
    if is_training:
        if use_enhanced_augment:
            # 增强版数据增强
            transform = [
                vision.RandomCrop(image_size, padding=4),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.RandomVerticalFlip(prob=0.2),  # 新增垂直翻转
                vision.RandomRotation(degrees=30),     # 增大旋转角度
                vision.RandomColorAdjust(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3
                ),  # 新增颜色调整
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
        else:
            # 基础版数据增强
            transform = [
                vision.RandomCrop(image_size, padding=4),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.RandomRotation(degrees=15),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
    else:
        transform = [
            vision.CenterCrop(image_size),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]
    
    # 创建数据集
    dataset = create_generator_dataset(images, labels, transform=transform)
    
    # 批处理（使用并行）
    try:
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=4)
    except TypeError:
        # 如果不支持 num_parallel_workers，使用默认方式
        dataset = dataset.batch(batch_size, drop_remainder=True)
    
    print(f"✓ 数据集创建完成")
    print(f"  样本数量: {len(images)}")
    print(f"  批次数量: {dataset.get_dataset_size()}")
    
    return dataset


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("  测试 ArtBench-10 数据加载")
    print("=" * 70)
    print()
    
    try:
        # 测试加载
        dataset = create_artbench10_mindspore_dataset(
            "./artbench10_data",
            batch_size=4,
            is_training=True,
            image_size=224
        )
        
        print("\n测试读取数据...")
        for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
            if i >= 2:  # 只读取 2 个 batch
                break
            
            images = data['image']
            labels = data['label']
            
            print(f"\nBatch {i+1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  标签: {labels}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        
        print("\n✅ 测试成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
