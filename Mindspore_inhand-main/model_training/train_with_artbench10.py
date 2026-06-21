"""
使用 ArtBench-10 数据集训练风格分类模型

ArtBench-10 优势：
- ✅ 类别平衡（每类6000张）
- ✅ 高质量标注
- ✅ 标准化格式
- ✅ 自动下载
- ✅ 10种艺术风格完美匹配
"""

import os
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import nn, Tensor, Model, context
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import dtype as mstype
import mindspore.dataset.vision as vision
import mindspore.dataset as ds

# ==================== 配置参数 ====================

# 数据集配置
DATASET_NAME = "artbench10"
IMAGE_SIZE = 32  # ArtBench-10 原始图像尺寸为 32x32，无需 resize
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# ArtBench-10 的10种风格（与我们的定义完全匹配）
STYLE_LABELS = [
    "Abstract Expressionism",  # 抽象表现主义 → 抽象
    "Baroque",                  # 巴洛克 → 古典
    "Expressionism",            # 表现主义 → 现代艺术
    "Impressionism",            # 印象派
    "Minimalism",               # 极简主义
    "Pop Art",                  # 波普艺术
    "Realism",                  # 写实主义 → 写实
    "Rococo",                   # 洛可可 → 古典
    "Romanticism",              # 浪漫主义 → 古典
    "Renaissance"               # 文艺复兴 → 古典
]

# 映射到中文标签
CHINESE_LABELS = [
    "抽象", "古典", "现代艺术", "印象派", "极简主义",
    "波普艺术", "写实", "古典", "古典", "古典"
]


# ==================== 数据集加载 ====================

# 导入自定义加载器
from load_artbench_custom import create_artbench10_mindspore_dataset

def create_artbench_dataset(data_path, batch_size=32, is_training=True):
    """
    创建 ArtBench-10 数据集（使用自定义加载器）
    
    Args:
        data_path: 数据集根目录
        batch_size: 批次大小
        is_training: 是否为训练模式
    
    Returns:
        MindSpore Dataset
    """
    
    print(f"加载 ArtBench-10 数据集...")
    print(f"  路径: {data_path}")
    print(f"  模式: {'训练' if is_training else '测试'}")
    
    # 使用自定义加载器（绕过 Cifar10Dataset 的格式限制）
    try:
        dataset = create_artbench10_mindspore_dataset(
            data_path,
            batch_size=batch_size,
            is_training=is_training,
            image_size=IMAGE_SIZE
        )
        return dataset
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        print("\n请运行诊断脚本检查数据集结构:")
        print("  python check_dataset.py")
        raise


# ==================== 模型定义 ====================

class StyleClassifier(nn.Cell):
    """
    图像风格分类网络（适配 32x32 输入）
    """
    
    def __init__(self, num_classes=10):
        super(StyleClassifier, self).__init__()
        
        # 简化的卷积网络
        self.features = nn.SequentialCell([
            # 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 4x4 -> 2x2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])
        
        # 全局平均池化 (2x2 -> 1x1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        
        # 分类器
        self.classifier = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(256, 128),  # 256 = 256 * 1 * 1
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 使用 p 参数代替 keep_prob
            nn.Dense(128, num_classes)
        ])
    
    def construct(self, x):
        x = self.features(x)     # 32x32 -> 2x2x256
        x = self.avg_pool(x)     # 2x2 -> 1x1x256
        x = self.classifier(x)   # -> num_classes
        return x


# ==================== 训练函数 ====================

def train():
    """
    使用 ArtBench-10 训练风格分类模型
    """
    
    print("=" * 70)
    print("  使用 ArtBench-10 数据集训练图像风格分类模型")
    print("=" * 70)
    print()
    
    # 设置运行上下文
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",  # 可改为 "GPU" 或 "Ascend"
        save_graphs=False
    )
    
    # 数据集路径
    data_path = "./artbench10_data"
    
    # 创建数据集
    print("[1/5] 加载数据集...")
    print("-" * 70)
    
    try:
        train_dataset = create_artbench_dataset(
            data_path,
            batch_size=BATCH_SIZE,
            is_training=True
        )
        
        test_dataset = create_artbench_dataset(
            data_path,
            batch_size=BATCH_SIZE,
            is_training=False
        )
        
        train_steps = train_dataset.get_dataset_size()
        print(f"\n训练集: {train_steps} batches ({train_steps * BATCH_SIZE} 样本)")
        print(f"测试集: {test_dataset.get_dataset_size()} batches ({test_dataset.get_dataset_size() * BATCH_SIZE} 样本)")
        
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {str(e)}")
        print("\n请确保已下载 ArtBench-10 数据集")
        print("下载地址: https://artbench.eecs.berkeley.edu/")
        print("\n或使用 torchvision 自动下载:")
        print("  from torchvision.datasets import CIFAR10")
        print("  dataset = CIFAR10(root='./data', download=True)")
        return None
    
    # 创建网络
    print("\n[2/5] 创建网络模型...")
    print("-" * 70)
    network = StyleClassifier(num_classes=10)
    print("✓ 网络创建完成")
    
    # 定义损失函数和优化器
    print("\n[3/5] 配置训练参数...")
    print("-" * 70)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    optimizer = nn.Adam(
        params=network.trainable_params(),
        learning_rate=LEARNING_RATE
    )
    
    eval_metrics = {"Accuracy": nn.Accuracy()}
    
    model = Model(network, loss, optimizer, metrics=eval_metrics)
    print(f"✓ 学习率: {LEARNING_RATE}")
    print(f"✓ 批次大小: {BATCH_SIZE}")
    print(f"✓ 优化器: Adam")
    
    # 设置回调函数
    print("\n[4/5] 设置回调函数...")
    print("-" * 70)
    config_ck = CheckpointConfig(save_checkpoint_steps=train_steps, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="style_classifier_artbench", directory="./checkpoints", config=config_ck)
    
    loss_cb = LossMonitor(per_print_times=50)
    time_cb = TimeMonitor(data_size=train_steps)
    
    callbacks = [ckpt_cb, loss_cb, time_cb]
    print("✓ 检查点保存: 每轮保存")
    print("✓ 损失监控: 每50步打印")
    
    # 开始训练
    print("\n[5/5] 开始训练...")
    print("=" * 70)
    
    model.train(
        epoch=EPOCHS,
        train_dataset=train_dataset,
        callbacks=callbacks,
        dataset_sink_mode=False
    )
    
    print("=" * 70)
    print("✅ 训练完成！")
    
    # 测试模型
    print("\n正在测试模型...")
    eval_result = model.eval(test_dataset, dataset_sink_mode=False)
    print(f"测试准确率: {eval_result['Accuracy']:.4f}")
    
    return model, network


# ==================== 导出模型 ====================

def export_model(model, network, export_path="style_classifier_artbench.ms"):
    """
    导出模型为 MindSpore Lite 格式
    
    Args:
        model: MindSpore Model 对象
        network: 网络对象
        export_path: 导出路径
    """
    
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
    print("  命令: converter --fmk=MINDIR --modelFile=style_classifier_artbench.mindir --outputFile=style_classifier")


# ==================== 主函数 ====================

if __name__ == "__main__":
    
    print("\n📊 ArtBench-10 数据集信息:")
    print("-" * 70)
    print("• 总图片数: 60,000 张")
    print("• 训练集: 50,000 张 (每类 5,000 张)")
    print("• 测试集: 10,000 张 (每类 1,000 张)")
    print("• 类别数: 10 种艺术风格")
    print("• 特点: 类别平衡、高质量、标准化")
    print()
    print("艺术风格列表:")
    for i, (en, cn) in enumerate(zip(STYLE_LABELS, CHINESE_LABELS)):
        print(f"  {i+1}. {en:25s} → {cn}")
    print()
    
    # 检查数据集
    data_path = "./artbench10_data/artbench-10-batches-py"
    if not os.path.exists(data_path):
        print(f"⚠️  警告: 数据集路径 {data_path} 不存在")
        print("\n请下载 ArtBench-10 数据集:")
        print("  1. 访问: https://artbench.eecs.berkeley.edu/")
        print("  2. 下载 artbench-10-python.tar.gz")
        print("  3. 解压到 artbench10_data 目录")
        print("\n或使用以下代码自动下载:")
        print("""
from torchvision.datasets import CIFAR10

# 自动下载并加载
dataset = CIFAR10(
    root='./artbench10_data',
    download=True
)
        """)
        exit(1)
    
    # 训练模型
    try:
        model, network = train()
        
        if model:
            # 导出模型
            export_model(model, network, "style_classifier_artbench.ms")
            
            print("\n" + "=" * 70)
            print("✅ 全部完成！")
            print("=" * 70)
            print(f"\n模型文件: style_classifier_artbench.ms")
            print(f"检查点: ./checkpoints/")
            print("\n下一步：")
            print("1. 转换模型: python convert_model.py --input style_classifier_artbench.air --output style_classifier.ms --android-path ../")
            print("2. 将 .ms 文件复制到 Android 项目的 assets 目录")
            print("3. 编译并运行 Android 应用")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
