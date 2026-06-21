"""
检查 ArtBench-10 数据集结构的诊断脚本
"""

import os
import sys

def check_dataset_structure():
    """检查数据集目录结构"""
    
    print("=" * 70)
    print("  ArtBench-10 数据集结构诊断")
    print("=" * 70)
    print()
    
    # 检查可能的路径
    paths_to_check = [
        "./artbench10_data",
        "./artbench10_data/artbench-10-batches-py",
    ]
    
    for path in paths_to_check:
        print(f"检查路径: {path}")
        print("-" * 70)
        
        if not os.path.exists(path):
            print(f"  ❌ 路径不存在\n")
            continue
        
        # 列出目录内容
        try:
            files = os.listdir(path)
            print(f"  ✓ 目录存在，包含 {len(files)} 个项目:\n")
            
            for item in sorted(files):
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"    📄 {item:30s} ({size:6.1f} MB)")
                else:
                    print(f"    📁 {item:30s} (目录)")
            
            print()
            
            # 检查是否是 CIFAR 格式
            cifar_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 
                          'data_batch_4', 'data_batch_5', 'test_batch', 'meta']
            
            found_files = [f for f in cifar_files if f in files]
            
            if len(found_files) == len(cifar_files):
                print(f"  ✅ 找到所有必需的 CIFAR 格式文件！")
                print(f"  这个路径可以用于 MindSpore Cifar10Dataset")
            elif len(found_files) > 0:
                print(f"  ⚠️  找到部分 CIFAR 文件 ({len(found_files)}/{len(cifar_files)}):")
                for f in found_files:
                    print(f"      - {f}")
                missing = [f for f in cifar_files if f not in files]
                print(f"  缺失的文件:")
                for f in missing:
                    print(f"      - {f}")
            else:
                print(f"  ❌ 未找到 CIFAR 格式文件")
                print(f"  这可能不是标准的 CIFAR-10 格式数据集")
            
        except Exception as e:
            print(f"  ❌ 读取目录失败: {e}\n")
        
        print()
    
    # 递归检查
    print("=" * 70)
    print("  完整目录树")
    print("=" * 70)
    print()
    
    def print_tree(path, prefix="", level=0):
        if level > 3:  # 限制深度
            return
        
        try:
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                full_path = os.path.join(path, item)
                is_last = (i == len(items) - 1)
                
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{item}")
                
                if os.path.isdir(full_path):
                    extension = "    " if is_last else "│   "
                    print_tree(full_path, prefix + extension, level + 1)
        except Exception as e:
            print(f"{prefix}└── [无法访问: {e}]")
    
    if os.path.exists("./artbench10_data"):
        print_tree("./artbench10_data")
    else:
        print("❌ artbench10_data 目录不存在")
    
    print()
    print("=" * 70)
    print("  建议")
    print("=" * 70)
    print()
    
    # 给出具体建议
    if os.path.exists("./artbench10_data/artbench-10-batches-py"):
        subdir = "./artbench10_data/artbench-10-batches-py"
        files = os.listdir(subdir)
        cifar_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 
                      'data_batch_4', 'data_batch_5', 'test_batch', 'meta']
        
        if all(f in files for f in cifar_files):
            print("✅ 数据集结构正确！")
            print(f"\n在代码中使用:")
            print(f'  dataset_dir = "./artbench10_data/artbench-10-batches-py"')
        else:
            print("⚠️  目录存在但文件格式可能不正确")
            print("\n请检查文件是否为 Python pickle 格式")
    
    elif os.path.exists("./artbench10_data"):
        files = os.listdir("./artbench10_data")
        cifar_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 
                      'data_batch_4', 'data_batch_5', 'test_batch', 'meta']
        
        if all(f in files for f in cifar_files):
            print("✅ 文件在根目录，结构正确！")
            print(f"\n在代码中使用:")
            print(f'  dataset_dir = "./artbench10_data"')
        else:
            print("❌ 数据结构不正确")
            print("\n可能需要重新下载或解压数据集")
    
    else:
        print("❌ 数据集目录不存在")
        print("\n请下载数据集:")
        print("  python download_artbench10.py")
    
    print()


if __name__ == "__main__":
    check_dataset_structure()
