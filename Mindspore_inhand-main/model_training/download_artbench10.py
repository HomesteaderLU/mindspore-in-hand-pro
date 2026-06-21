"""
ArtBench-10 数据集下载和准备工具

自动下载、解压和验证 ArtBench-10 数据集
"""

import os
import sys
import tarfile
import hashlib
from pathlib import Path


# ==================== 配置 ====================

DATASET_URL = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
DATASET_FILENAME = "artbench-10-python.tar.gz"
DATASET_DIR = "./artbench10_data"
EXPECTED_MD5 = "9df1e998ee026aae36ec60ca7b44960e"


def download_dataset():
    """
    下载 ArtBench-10 数据集
    """
    
    print("=" * 70)
    print("  ArtBench-10 数据集下载工具")
    print("=" * 70)
    print()
    
    # 检查是否已存在
    if os.path.exists(DATASET_DIR):
        print(f"✓ 数据集目录已存在: {DATASET_DIR}")
        print("  如需重新下载，请删除该目录后重试")
        return True
    
    print(f"📥 下载 ArtBench-10 数据集...")
    print(f"  URL: {DATASET_URL}")
    print(f"  目标: {DATASET_DIR}")
    print()
    
    try:
        import urllib.request
        
        # 创建临时文件
        temp_file = DATASET_FILENAME
        
        # 定义下载进度回调
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                sys.stdout.write(f"\r  下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
        
        # 下载文件
        print("  开始下载...")
        urllib.request.urlretrieve(DATASET_URL, temp_file, reporthook=progress_hook)
        print("\n  ✓ 下载完成")
        
        # 验证 MD5
        print("\n  验证文件完整性...")
        md5 = calculate_md5(temp_file)
        print(f"  计算 MD5: {md5}")
        print(f"  期望 MD5: {EXPECTED_MD5}")
        
        if md5 != EXPECTED_MD5:
            print("  ❌ MD5 校验失败！文件可能损坏")
            return False
        
        print("  ✓ MD5 校验通过")
        
        # 解压文件
        print("\n  解压文件...")
        extract_dataset(temp_file)
        
        # 清理临时文件
        print("  清理临时文件...")
        os.remove(temp_file)
        
        print("\n✅ 数据集下载完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        print("\n请尝试手动下载:")
        print(f"  1. 访问: {DATASET_URL}")
        print(f"  2. 下载 {DATASET_FILENAME}")
        print(f"  3. 解压到 {DATASET_DIR} 目录")
        return False


def calculate_md5(filepath, chunk_size=8192):
    """
    计算文件的 MD5 哈希值
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def extract_dataset(tar_path):
    """
    解压数据集
    """
    
    print(f"  解压 {tar_path}...")
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # 获取所有成员
            members = tar.getmembers()
            total = len(members)
            
            # 检查是否需要调整目录结构
            # ArtBench-10 解压后应该是 artbench-10-batches-py 文件夹
            first_member = members[0].name if members else ""
            
            for i, member in enumerate(members):
                tar.extract(member, DATASET_DIR)
                
                # 显示进度
                if i % 100 == 0 or i == total - 1:
                    percent = (i + 1) / total * 100
                    sys.stdout.write(f"\r  解压进度: {percent:.1f}% ({i+1}/{total})")
                    sys.stdout.flush()
        
        print("\n  ✓ 解压完成")
        
        # 验证目录结构
        expected_dir = os.path.join(DATASET_DIR, "artbench-10-batches-py")
        if os.path.exists(expected_dir):
            print(f"  ✓ 目录结构正确: {expected_dir}")
        else:
            # 检查是否直接解压到了 DATASET_DIR
            if os.path.exists(os.path.join(DATASET_DIR, "data_batch_1")):
                print(f"  ⚠️  文件直接解压到 {DATASET_DIR}")
                print(f"  建议移动到子目录以获得更好的组织")
            else:
                print(f"  ❌ 找不到预期的数据文件")
                print(f"  请检查解压后的目录结构")
        
    except Exception as e:
        print(f"\n  ❌ 解压失败: {str(e)}")
        raise


def verify_dataset():
    """
    验证数据集完整性
    """
    DATASET_DIR = "./artbench10_data/artbench-10-batches-py"

    print("\n" + "=" * 70)
    print("  验证数据集")
    print("=" * 70)
    print()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录不存在: {DATASET_DIR}")
        return False
    
    # 检查必需的文件
    required_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
        "meta"
    ]
    
    print("检查文件...")
    all_exist = True
    
    for filename in required_files:
        filepath = os.path.join(DATASET_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename:20s} ({size:.1f} MB)")
        else:
            print(f"  ✗ {filename:20s} (缺失)")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 数据集不完整")
        return False
    
    # 统计信息
    print("\n数据集统计:")
    print(f"  训练批次: 5 个 (data_batch_1-5)")
    print(f"  测试批次: 1 个 (test_batch)")
    print(f"  元数据: meta")
    print(f"  每类训练样本: 5,000 张")
    print(f"  每类测试样本: 1,000 张")
    print(f"  总样本数: 60,000 张")
    
    print("\n✅ 数据集验证通过！")
    return True


def print_usage():
    """
    打印使用说明
    """
    
    print("\n" + "=" * 70)
    print("  使用方法")
    print("=" * 70)
    print()
    print("方式 1: 自动下载（推荐）")
    print("  python download_artbench10.py")
    print()
    print("方式 2: 仅验证已有数据集")
    print("  python download_artbench10.py --verify")
    print()
    print("方式 3: 手动下载")
    print(f"  1. 访问: {DATASET_URL}")
    print(f"  2. 下载: {DATASET_FILENAME}")
    print(f"  3. 解压到: {DATASET_DIR}")
    print(f"  4. 运行: python download_artbench10.py --verify")
    print()


def main():
    """
    主函数
    """
    
    # 解析命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # 仅验证
        success = verify_dataset()
        sys.exit(0 if success else 1)
    
    # 打印使用说明
    print_usage()
    
    # 下载数据集
    success = download_dataset()
    
    if success:
        # 验证数据集
        verify_dataset()
        
        print("\n" + "=" * 70)
        print("  ✅ 准备完成！")
        print("=" * 70)
        print()
        print("下一步：训练模型")
        print("  python train_with_artbench10.py")
        print()
    else:
        print("\n❌ 准备失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
