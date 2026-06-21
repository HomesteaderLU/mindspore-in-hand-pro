"""
使用 Python API 转换 MINDIR 到 MS 格式
"""

import sys
import os

try:
    import mindspore_lite as mslite
    print("✓ MindSpore Lite 已安装")
except ImportError:
    print("❌ 错误: 未安装 mindspore_lite")
    print("\n请安装:")
    print("  pip install mindspore-lite")
    sys.exit(1)

def convert_model(input_path, output_path):
    """
    转换 MINDIR 模型为 MS 格式
    
    Args:
        input_path: 输入的 MINDIR 文件路径
        output_path: 输出的 MS 文件路径
    """
    
    print(f"\n输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    
    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件 {input_path}")
        return False
    
    try:
        # 创建 Model 对象
        print("\n[1/3] 创建 Model 对象...")
        model = mslite.Model()
        
        # 从 MINDIR 文件加载
        print("[2/3] 加载 MINDIR 模型...")
        model.build_from_file(input_path, mslite.ModelType.MINDIR)
        
        # 导出为 MS 格式
        print("[3/3] 导出为 MS 格式...")
        model.export(output_path, mslite.ModelType.MINDIR)
        
        print(f"\n✓ 转换成功!")
        print(f"输出文件: {output_path}")
        
        # 检查文件大小
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"文件大小: {size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("MindSpore Lite 模型转换工具 (Python API)")
    print("=" * 70)
    
    input_model = "style_classifier_artbench.mindir"
    output_model = "style_classifier.ms"
    
    success = convert_model(input_model, output_model)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ 完成！")
        print("=" * 70)
        print(f"\n下一步：")
        print(f"  1. 将 {output_model} 复制到 Android 项目的 assets 目录")
        print(f"  2. 编译并运行 Android 应用")
    else:
        print("\n" + "=" * 70)
        print("❌ 失败！")
        print("=" * 70)
        print("\n建议：")
        print("  1. 检查 mindspore_lite 是否正确安装")
        print("  2. 检查 MINDIR 文件是否完整")
        print("  3. 查看上面的错误信息")
