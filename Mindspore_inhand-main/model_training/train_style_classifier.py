"""
图像风格分类模型训练脚本
使用 MindSpore 训练一个可以识别10种艺术风格的分类模型

支持的风格：
- 油画 (Oil Painting)
- 水彩 (Watercolor)
- 素描 (Sketch)
- 写实 (Realistic)
- 抽象 (Abstract)
- 印象派 (Impressionism)
- 现代艺术 (Modern Art)
- 古典 (Classical)
- 波普艺术 (Pop Art)
- 极简主义 (Minimalism)
"""

import os
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import nn, Tensor, Model, context
from mindspore.nn import CrossEntropyLoss, SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import Compose
import mindspore.dataset.vision as vision
import mindspore.dataset as ds

# ==================== 配置参数 ====================

# 数据集路径（需要自己准备）
DATASET_PATH = "./style_dataset"

# 模型参数
NUM_CLASSES = 10  # 10种风格
IMAGE_SIZE = 224  # 输入图像尺寸
BATCH_SIZE = 32   # 批次大小
EPOCHS = 50       # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# 风格标签
STYLE_LABELS = [
    "oil_painting",    # 油画
    "watercolor",      # 水彩
    "sketch",          # 素描
    "realistic",       # 写实
    "abstract",        # 抽象
    "impressionism",   # 印象派
    "modern_art",      # 现代艺术
    "classical",       # 古典
    "pop_art",         # 波普艺术
    "minimalism"       # 极简主义
]

# ==================== 数据预处理 ====================

def create_dataset(dataset_path, batch_size=32, is_training=True):
    """
    创建数据集
    
    Args:
        dataset_path: 数据集根目录
        batch_size: 批次大小
        is_training: 是否为训练模式
    
    Returns:
        MindSpore Dataset
    """
    
    # 定义数据增强操作
    if is_training:
        # 训练时的数据增强
        transform = Compose([
            vision.Decode(),
            vision.RandomCropDecodeResize(size=IMAGE_SIZE, scale=(0.8, 1.0)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomRotation(degrees=15),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ])
    else:
        # 验证/测试时的处理
        transform = Compose([
            vision.Decode(),
            vision.Resize(IMAGE_SIZE),
            vision.CenterCrop(IMAGE_SIZE),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ])
    
    # 加载数据集
    dataset = ds.ImageFolderDataset(
        dataset_dir=dataset_path,
        num_parallel_workers=4,
        shuffle=is_training
    )
    
    # 应用变换
    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.map(operations=lambda x: Tensor(x, mstype.int32), input_columns="label")
    
    # 批处理和重复
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    
    return dataset


# ==================== 模型定义 ====================

class StyleClassifier(nn.Cell):
    """
    图像风格分类网络
    
    基于 MobileNetV2 架构，轻量级且适合移动端部署
    """
    
    def __init__(self, num_classes=10):
        super(StyleClassifier, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # 深度可分离卷积块 1
        self.dw_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, group=32, pad_mode='same')
        self.pw_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, pad_mode='same')
        self.bn2 = nn.BatchNorm2d(64)
        
        # 深度可分离卷积块 2
        self.dw_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, group=64, pad_mode='same')
        self.pw_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, pad_mode='same')
        self.bn3 = nn.BatchNorm2d(128)
        
        # 深度可分离卷积块 3
        self.dw_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, group=128, pad_mode='same')
        self.pw_conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, pad_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        
        # 深度可分离卷积块 4
        self.dw_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, group=128, pad_mode='same')
        self.pw_conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, pad_mode='same')
        self.bn5 = nn.BatchNorm2d(256)
        
        # 全局平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(256, 128)
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.fc2 = nn.Dense(128, num_classes)
    
    def construct(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 块 1
        x = self.dw_conv1(x)
        x = self.pw_conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 块 2
        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # 块 3
        x = self.dw_conv3(x)
        x = self.pw_conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # 块 4
        x = self.dw_conv4(x)
        x = self.pw_conv4(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        # 池化和分类
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ==================== 训练函数 ====================

def train():
    """
    训练风格分类模型
    """
    
    print("=" * 60)
    print("开始训练图像风格分类模型")
    print("=" * 60)
    
    # 设置运行上下文
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",  # 可以改为 "GPU" 或 "Ascend"
        save_graphs=False
    )
    
    # 创建数据集
    print("\n[1/5] 加载数据集...")
    train_dataset = create_dataset(
        os.path.join(DATASET_PATH, "train"),
        batch_size=BATCH_SIZE,
        is_training=True
    )
    
    val_dataset = create_dataset(
        os.path.join(DATASET_PATH, "val"),
        batch_size=BATCH_SIZE,
        is_training=False
    )
    
    train_steps = train_dataset.get_dataset_size()
    print(f"训练集大小: {train_steps} batches")
    print(f"验证集大小: {val_dataset.get_dataset_size()} batches")
    
    # 创建网络
    print("\n[2/5] 创建网络模型...")
    network = StyleClassifier(num_classes=NUM_CLASSES)
    
    # 定义损失函数
    print("\n[3/5] 配置训练参数...")
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 定义优化器
    optimizer = nn.Adam(
        params=network.trainable_params(),
        learning_rate=LEARNING_RATE
    )
    
    # 定义评估指标
    eval_metrics = {"Accuracy": nn.Accuracy()}
    
    # 创建 Model
    model = Model(network, loss, optimizer, metrics=eval_metrics)
    
    # 设置回调函数
    print("\n[4/5] 设置回调函数...")
    config_ck = CheckpointConfig(save_checkpoint_steps=train_steps, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="style_classifier", directory="./checkpoints", config=config_ck)
    
    loss_cb = LossMonitor(per_print_times=50)
    time_cb = TimeMonitor(data_size=train_steps)
    
    callbacks = [ckpt_cb, loss_cb, time_cb]
    
    # 开始训练
    print("\n[5/5] 开始训练...")
    print("-" * 60)
    
    model.train(
        epoch=EPOCHS,
        train_dataset=train_dataset,
        callbacks=callbacks,
        dataset_sink_mode=False
    )
    
    print("-" * 60)
    print("训练完成！")
    
    # 验证模型
    print("\n正在验证模型...")
    eval_result = model.eval(val_dataset, dataset_sink_mode=False)
    print(f"验证准确率: {eval_result['Accuracy']:.4f}")
    
    return model


# ==================== 导出模型 ====================

def export_model(model, export_path="style_classifier.ms"):
    """
    导出模型为 MindSpore Lite 格式 (.ms)
    
    Args:
        model: 训练好的模型
        export_path: 导出路径
    """
    
    print("\n" + "=" * 60)
    print("导出模型为 MindSpore Lite 格式")
    print("=" * 60)
    
    # 设置导出上下文
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    # 创建虚拟输入
    dummy_input = Tensor(np.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32))
    
    # 导出为 AIR 格式（中间格式）
    air_path = export_path.replace(".ms", ".air")
    ms.export(model.network, dummy_input, file_name=air_path, file_format="AIR")
    print(f"✓ 已导出 AIR 格式: {air_path}")
    
    # 转换为 MS 格式（需要使用 MindSpore Lite 转换工具）
    print("\n注意：需要使用 MindSpore Lite 转换工具将 AIR 转换为 MS 格式")
    print("命令示例：")
    print(f"converter --fmk=AIR --modelFile={air_path} --outputFile={export_path}")
    print("\n或者使用 Python API:")
    print("""
    from mindspore_lite import ModelConvertor
    
    convertor = ModelConvertor()
    convertor.convert(
        model_type="AIR",
        model_file=air_path,
        output_file=export_path,
        fp16=False,
        quant_type="POST_TRAINING_AWARE"
    )
    """)


# ==================== 数据集准备说明 ====================

def prepare_dataset_instructions():
    """
    打印数据集准备说明
    """
    
    print("\n" + "=" * 60)
    print("数据集准备说明")
    print("=" * 60)
    print("""
您需要准备以下结构的数据集：

style_dataset/
├── train/
│   ├── oil_painting/      # 油画图片
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── watercolor/        # 水彩图片
│   ├── sketch/            # 素描图片
│   ├── realistic/         # 写实图片
│   ├── abstract/          # 抽象图片
│   ├── impressionism/     # 印象派图片
│   ├── modern_art/        # 现代艺术图片
│   ├── classical/         # 古典图片
│   ├── pop_art/           # 波普艺术图片
│   └── minimalism/        # 极简主义图片
└── val/
    ├── oil_painting/
    ├── watercolor/
    └── ... (同上)

建议：
1. 每个类别至少准备 500-1000 张图片
2. 图片格式支持 JPG、PNG
3. 图片尺寸不限，会自动调整为 224x224
4. 训练集和验证集比例建议为 8:2

数据来源：
- WikiArt 数据集
- Kaggle 艺术风格数据集
- 自行收集和标注
    """)


# ==================== 主函数 ====================

if __name__ == "__main__":
    
    # 打印数据集准备说明
    prepare_dataset_instructions()
    
    # 检查数据集是否存在
    if not os.path.exists(DATASET_PATH):
        print(f"\n⚠️  警告：数据集路径 {DATASET_PATH} 不存在")
        print("请先准备数据集，然后再次运行此脚本")
        exit(1)
    
    # 训练模型
    try:
        model = train()
        
        # 导出模型
        export_model(model, "style_classifier.ms")
        
        print("\n" + "=" * 60)
        print("✅ 全部完成！")
        print("=" * 60)
        print(f"模型文件: style_classifier.ms")
        print(f"检查点文件: ./checkpoints/")
        print("\n下一步：")
        print("1. 将 style_classifier.ms 复制到 Android 项目的 assets 目录")
        print("2. 在 Android 代码中加载模型")
        print("3. 编译并运行应用")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
