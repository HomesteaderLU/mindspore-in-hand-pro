"""
检查 ArtBench-10 批次文件的实际结构
"""

import pickle
import os

def inspect_batch_file(filepath):
    """检查单个批次文件的结构"""
    
    print(f"\n检查文件: {filepath}")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        print(f"  ❌ 文件不存在")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        
        print(f"  ✓ 文件加载成功")
        print(f"  类型: {type(data)}")
        print(f"\n  字典键列表:")
        
        if isinstance(data, dict):
            for key in data.keys():
                value = data[key]
                if isinstance(value, (list, tuple)):
                    print(f"    - {key}: {type(value)}, 长度={len(value)}")
                    if len(value) > 0:
                        print(f"      第一个元素: {value[0]} (类型: {type(value[0])})")
                elif isinstance(value, bytes):
                    print(f"    - {key}: bytes, 长度={len(value)}")
                elif isinstance(value, (int, float, str)):
                    print(f"    - {key}: {type(value).__name__}, 值={value}")
                else:
                    print(f"    - {key}: {type(value)}")
        else:
            print(f"  数据不是字典类型: {type(data)}")
            print(f"  数据内容预览: {str(data)[:200]}")
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("  ArtBench-10 批次文件结构检查")
    print("=" * 70)
    
    # 检查训练批次
    batch_files = [
        "./artbench10_data/artbench-10-batches-py/data_batch_1",
        "./artbench10_data/artbench-10-batches-py/test_batch",
        "./artbench10_data/artbench-10-batches-py/meta",
    ]
    
    for filepath in batch_files:
        inspect_batch_file(filepath)
    
    print("\n" + "=" * 70)
    print("  建议")
    print("=" * 70)
    print("\n根据输出结果，修改 load_artbench_custom.py 中的键名")
    print("如果键名不是 b'data' 和 b'labels'，需要相应调整代码")
