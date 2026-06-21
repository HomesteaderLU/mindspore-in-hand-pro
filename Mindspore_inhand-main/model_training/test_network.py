"""
测试网络架构，检查各层的输出形状
"""

import numpy as np
from mindspore import Tensor, nn
import mindspore as ms

class StyleClassifier(nn.Cell):
    """图像风格分类网络（适配 32x32 输入）"""
    
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
            nn.Dense(256, 128),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(128, num_classes)
        ])
    
    def construct(self, x):
        print(f"输入形状: {x.shape}")
        x = self.features(x)
        print(f"features 输出: {x.shape}")
        x = self.avg_pool(x)
        print(f"avg_pool 输出: {x.shape}")
        x = self.classifier(x)
        print(f"classifier 输出: {x.shape}")
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("  测试网络架构")
    print("=" * 70)
    
    # 设置 MindSpore 模式
    ms.set_context(mode=ms.GRAPH_MODE)
    
    # 创建网络
    network = StyleClassifier(num_classes=10)
    
    # 创建测试数据 (batch_size=1, channels=3, height=32, width=32)
    test_input = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
    
    print("\n前向传播测试:")
    print("-" * 70)
    
    try:
        output = network(test_input)
        print("\n✓ 测试成功!")
        print(f"最终输出形状: {output.shape}")
        print(f"预期输出: (1, 10)")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
