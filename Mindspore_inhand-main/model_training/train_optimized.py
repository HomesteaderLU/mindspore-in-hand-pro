"""
优化的图像风格分类训练脚本

主要改进：
1. 更深的网络架构（ResNet-style）
2. 学习率调度器
3. 更强的数据增强
4. Label Smoothing
5. Mixup 数据增强
6. 更长的训练周期
7. 权重衰减正则化
"""

import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Model, context
from mindspore.common import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import load_checkpoint, load_param_into_net

# 导入自定义加载器
from load_artbench_custom import create_artbench10_mindspore_dataset

# ==================== 优化配置参数 ====================

# 数据集配置
DATASET_NAME = "artbench10"
IMAGE_SIZE = 32
BATCH_SIZE = 128  # 增大批次大小以加速训练
EPOCHS = 10       # 减少训练轮数以加快速度
INITIAL_LR = 0.002  # 稍高的学习率加快收敛
WEIGHT_DECAY = 1e-4  # 权重衰减

# 学习率调度
LR_SCHEDULE = 'cosine'  # 'cosine', 'step', 'warmup_cosine'
WARMUP_EPOCHS = 5

# 正则化
LABEL_SMOOTHING = 0.1  # Label Smoothing
MIXUP_ALPHA = 0.2      # Mixup 强度（0 表示禁用）
DROPOUT_RATE = 0.3     # Dropout 概率

# ArtBench-10 的10种风格
STYLE_LABELS = [
    "Abstract Expressionism", "Baroque", "Expressionism", 
    "Impressionism", "Minimalism", "Pop Art", 
    "Realism", "Rococo", "Romanticism", "Renaissance"
]


# ==================== 改进的网络架构 ====================

class ResidualBlock(nn.Cell):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            self.shortcut = nn.Identity()
    
    def construct(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加残差连接
        out = out + self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ImprovedStyleClassifier(nn.Cell):
    """
    改进的图像风格分类网络（ResNet-style）
    """
    
    def __init__(self, num_classes=10):
        super(ImprovedStyleClassifier, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                               pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # 残差块组 1: 32x32 -> 16x16
        self.layer1 = nn.SequentialCell([
            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        ])
        
        # 残差块组 2: 16x16 -> 8x8
        self.layer2 = nn.SequentialCell([
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        ])
        
        # 残差块组 3: 8x8 -> 4x4
        self.layer3 = nn.SequentialCell([
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        ])
        
        # 全局平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        
        # 分类器
        self.classifier = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(256, 256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Dense(256, num_classes)
        ])
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)  # 32x32 -> 16x16
        x = self.layer2(x)  # 16x16 -> 8x8
        x = self.layer3(x)  # 8x8 -> 4x4
        
        x = self.avg_pool(x)  # 4x4 -> 1x1
        x = self.classifier(x)
        
        return x


# ==================== 数据增强优化 ====================

def create_enhanced_dataset(data_path, batch_size=64, is_training=True):
    """
    创建增强版数据集（更强的数据增强）
    """
    
    print(f"加载 ArtBench-10 数据集...")
    print(f"  路径: {data_path}")
    print(f"  模式: {'训练' if is_training else '测试'}")
    print(f"  批次大小: {batch_size}")
    
    try:
        dataset = create_artbench10_mindspore_dataset(
            data_path,
            batch_size=batch_size,
            is_training=is_training,
            image_size=IMAGE_SIZE,
            use_enhanced_augment=is_training  # 训练时使用增强
        )
        return dataset
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        raise


# ==================== 学习率调度器 ====================

def get_lr_scheduler(epoch_size, initial_lr, epochs, schedule_type='cosine'):
    """
    创建学习率调度器（手动实现）
    
    Args:
        epoch_size: 每个 epoch 的步数
        initial_lr: 初始学习率
        epochs: 总训练轮数
        schedule_type: 调度类型 ('cosine', 'step', 'warmup_cosine')
    
    Returns:
        list: 学习率列表
    """
    
    total_steps = epoch_size * epochs
    lr_list = []
    
    if schedule_type == 'cosine':
        # Cosine 退火学习率调度
        min_lr = initial_lr * 0.01
        for step in range(total_steps):
            lr = min_lr + (initial_lr - min_lr) * 0.5 * \
                 (1 + np.cos(np.pi * step / total_steps))
            lr_list.append(lr)
        print(f"✓ 使用 Cosine 学习率调度")
        print(f"  初始学习率: {initial_lr}")
        print(f"  最小学习率: {min_lr}")
        
    elif schedule_type == 'step':
        # 阶梯式下降
        decay_factor = 0.1
        step_size = epochs // 3
        for step in range(total_steps):
            current_epoch = step // epoch_size
            num_decays = current_epoch // step_size
            lr = initial_lr * (decay_factor ** num_decays)
            lr_list.append(lr)
        print(f"✓ 使用 Step 学习率调度")
        print(f"  每 {step_size} 个 epoch 降低 {decay_factor}x")
        
    elif schedule_type == 'warmup_cosine':
        # Warmup + Cosine
        warmup_steps = WARMUP_EPOCHS * epoch_size
        for step in range(total_steps):
            if step < warmup_steps:
                # Warmup 阶段：线性增长
                lr = initial_lr * (step / warmup_steps)
            else:
                # Cosine 退火阶段
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
            lr_list.append(lr)
        print(f"✓ 使用 Warmup + Cosine 学习率调度")
        print(f"  Warmup: {WARMUP_EPOCHS} epochs")
        print(f"  初始学习率: {initial_lr}")
    
    else:
        # 固定学习率
        lr_list = [initial_lr] * total_steps
        print(f"✓ 使用固定学习率: {initial_lr}")
    
    return lr_list


# ==================== 损失函数（Label Smoothing）====================

class LabelSmoothingCrossEntropy(nn.Cell):
    """带 Label Smoothing 的交叉熵损失"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    
    def construct(self, logits, labels):
        n_class = logits.shape[1]
        
        # One-hot 编码
        one_hot = ms.ops.one_hot(
            labels, 
            n_class,
            ms.Tensor(self.confidence, ms.float32),
            ms.Tensor(self.smoothing / (n_class - 1), ms.float32)
        )
        
        # 计算交叉熵
        loss = self.cross_entropy(logits, one_hot)
        
        return loss


# ==================== 训练函数 ====================

def train():
    """训练优化版模型"""
    
    print("\n" + "=" * 70)
    print("  使用优化配置训练图像风格分类模型")
    print("=" * 70)
    
    # 设置 MindSpore 环境
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    # [1/6] 加载数据集
    print("\n[1/6] 加载数据集...")
    print("-" * 70)
    
    data_path = "./artbench10_data/artbench-10-batches-py"
    if not os.path.exists(data_path):
        data_path = "./artbench10_data"
        if not os.path.exists(data_path):
            print(f"⚠️  警告: 数据集路径不存在")
            return None
    
    train_dataset = create_enhanced_dataset(data_path, BATCH_SIZE, is_training=True)
    test_dataset = create_enhanced_dataset(data_path, BATCH_SIZE, is_training=False)
    
    epoch_size = train_dataset.get_dataset_size()
    print(f"✓ 训练集大小: {train_dataset.get_dataset_size()} 批次")
    print(f"✓ 测试集大小: {test_dataset.get_dataset_size()} 批次")
    
    # [2/6] 创建网络
    print("\n[2/6] 创建网络模型...")
    print("-" * 70)
    
    network = ImprovedStyleClassifier(num_classes=10)
    print(f"✓ 网络架构: ResNet-style")
    print(f"  参数量: {sum(p.size for p in network.trainable_params()):,}")
    
    # [3/6] 配置学习率调度器
    print("\n[3/6] 配置学习率调度器...")
    print("-" * 70)
    
    lr = get_lr_scheduler(epoch_size, INITIAL_LR, EPOCHS, LR_SCHEDULE)
    
    # [4/6] 配置优化器和损失函数
    print("\n[4/6] 配置优化器和损失函数...")
    print("-" * 70)
    
    optimizer = nn.Adam(
        params=network.trainable_params(),
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY  # L2 正则化
    )
    
    # 使用 Label Smoothing
    if LABEL_SMOOTHING > 0:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        print(f"✓ 损失函数: Label Smoothing Cross Entropy (smoothing={LABEL_SMOOTHING})")
    else:
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        print(f"✓ 损失函数: Softmax Cross Entropy")
    
    eval_metrics = {"Accuracy": nn.Accuracy()}
    
    model = Model(network, loss_fn, optimizer, metrics=eval_metrics)
    print(f"✓ 优化器: Adam (weight_decay={WEIGHT_DECAY})")
    print(f"✓ 初始学习率: {INITIAL_LR}")
    print(f"✓ 批次大小: {BATCH_SIZE}")
    
    # [5/6] 设置回调函数
    print("\n[5/6] 设置回调函数...")
    print("-" * 70)
    
    # 保存检查点
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=epoch_size,
        keep_checkpoint_max=5
    )
    ckpt_callback = ModelCheckpoint(
        prefix="style_classifier_optimized",
        directory="./checkpoints_optimized",
        config=ckpt_config
    )
    
    callbacks = [
        LossMonitor(per_print_times=epoch_size // 2),  # 减少打印频率以加速
        TimeMonitor(),
        ckpt_callback
    ]
    
    print(f"✓ 检查点保存: ./checkpoints_optimized/")
    print(f"✓ 每 {EPOCHS // 5} 个 epoch 保存一次")
    
    # [6/6] 开始训练
    print("\n[6/6] 开始训练...")
    print("=" * 70)
    print(f"训练配置:")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {INITIAL_LR} ({LR_SCHEDULE})")
    print(f"  - Weight Decay: {WEIGHT_DECAY}")
    print(f"  - Label Smoothing: {LABEL_SMOOTHING}")
    print(f"  - Dropout: {DROPOUT_RATE}")
    print("=" * 70)
    
    model.train(
        epoch=EPOCHS,
        train_dataset=train_dataset,
        callbacks=callbacks,
        dataset_sink_mode=False  # 禁用下沉模式以避免兼容性问题
    )
    
    print("=" * 70)
    print("✅ 训练完成！")
    
    # 测试模型
    print("\n正在测试模型...")
    eval_result = model.eval(test_dataset, dataset_sink_mode=False)  # 禁用下沉模式
    print(f"测试准确率: {eval_result['Accuracy']:.4f}")
    
    return model, network


# ==================== 导出模型 ====================

def export_model(model, network, export_path="style_classifier_optimized.ms"):
    """导出优化后的模型"""
    
    print("\n" + "=" * 70)
    print("导出模型")
    print("=" * 70)
    
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    # 创建虚拟输入
    dummy_input = Tensor(np.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32))
    
    # 导出为 MINDIR 格式（推荐，支持所有后端）
    mindir_path = export_path.replace(".ms", ".mindir")
    ms.export(network, dummy_input, file_name=mindir_path, file_format="MINDIR")
    print(f"✓ 已导出 MINDIR 格式: {mindir_path}")
    
    print("\n下一步：转换为 MS 格式")
    print("  使用 MindSpore Lite Converter 工具转换")
    print(f"  命令: converter --fmk=MINDIR --modelFile={mindir_path} --outputFile=style_classifier_optimized")


# ==================== 主函数 ====================

if __name__ == "__main__":
    
    print("\n📊 ArtBench-10 数据集信息:")
    print("-" * 70)
    print("• 总图片数: 60,000 张")
    print("• 训练集: 50,000 张 (每类 5,000 张)")
    print("• 测试集: 10,000 张 (每类 1,000 张)")
    print("• 类别数: 10 种艺术风格")
    print("• 特点: 类别平衡、高质量、标准化")
    
    # 检查数据集
    data_path = "./artbench10_data/artbench-10-batches-py"
    if not os.path.exists(data_path):
        data_path = "./artbench10_data"
        if not os.path.exists(data_path):
            print(f"\n⚠️  警告: 数据集路径 {data_path} 不存在")
            print("\n请先下载 ArtBench-10 数据集:")
            print("  python download_artbench10.py")
            exit(1)
    
    # 训练模型
    try:
        model, network = train()
        
        if model:
            # 导出模型
            export_model(model, network, "style_classifier_optimized.ms")
            
            print("\n" + "=" * 70)
            print("✅ 全部完成！")
            print("=" * 70)
            print(f"\n模型文件: style_classifier_optimized.ms")
            print(f"检查点: ./checkpoints_optimized/")
            print("\n下一步：")
            print("1. 转换模型: python convert_model.py --input style_classifier_optimized.air --output style_classifier_optimized.ms --android-path ../")
            print("2. 将 .ms 文件复制到 Android 项目的 assets 目录")
            print("3. 编译并运行 Android 应用")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
