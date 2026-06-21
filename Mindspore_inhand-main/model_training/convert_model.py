"""
模型转换工具
将 MindSpore 训练好的模型转换为 MindSpore Lite 格式 (.ms)
"""

import os
import argparse


def convert_air_to_ms(air_path, output_path=None, quantize=False):
    """
    将 AIR 格式模型转换为 MS 格式
    
    Args:
        air_path: AIR 模型文件路径
        output_path: 输出 MS 文件路径（可选）
        quantize: 是否进行量化
    """
    
    if output_path is None:
        output_path = air_path.replace(".air", ".ms")
    
    print("=" * 60)
    print("模型转换工具")
    print("=" * 60)
    print(f"\n输入文件: {air_path}")
    print(f"输出文件: {output_path}")
    print(f"量化: {'是' if quantize else '否'}\n")
    
    # 检查输入文件
    if not os.path.exists(air_path):
        print(f"❌ 错误: 文件不存在 - {air_path}")
        return False
    
    try:
        # 方法 1: 使用 mindspore_lite 库（推荐）
        print("方法 1: 使用 MindSpore Lite Converter API")
        print("-" * 60)
        
        try:
            from mindspore_lite import ModelConvertor
            
            convertor = ModelConvertor()
            
            # 配置转换参数
            config = {
                "model_type": "AIR",
                "model_file": air_path,
                "output_file": output_path,
                "fp16": False,
            }
            
            if quantize:
                config["quant_type"] = "POST_TRAINING_AWARE"
                print("启用后训练感知量化...")
            
            # 执行转换
            result = convertor.convert(**config)
            
            if result:
                print(f"✅ 转换成功！")
                print(f"输出文件: {output_path}")
                
                # 显示文件大小
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"文件大小: {file_size:.2f} MB")
                
                return True
            else:
                print("❌ 转换失败")
                return False
                
        except ImportError:
            print("⚠️  mindspore_lite 未安装，尝试方法 2...\n")
        
        # 方法 2: 使用命令行工具
        print("方法 2: 使用 converter 命令行工具")
        print("-" * 60)
        
        cmd = f"converter --fmk=AIR --modelFile={air_path} --outputFile={output_path}"
        
        if quantize:
            cmd += " --quantType=POST_TRAINING_AWARE"
        
        print(f"执行命令: {cmd}\n")
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 转换成功！")
            print(f"输出文件: {output_path}")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"文件大小: {file_size:.2f} MB")
            
            return True
        else:
            print(f"❌ 转换失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 转换过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def optimize_model(ms_path, output_path=None):
    """
    优化 MS 模型（可选）
    
    Args:
        ms_path: MS 模型文件路径
        output_path: 输出路径（可选）
    """
    
    if output_path is None:
        output_path = ms_path.replace(".ms", "_optimized.ms")
    
    print("\n" + "=" * 60)
    print("模型优化")
    print("=" * 60)
    print(f"\n输入文件: {ms_path}")
    print(f"输出文件: {output_path}\n")
    
    try:
        from mindspore_lite import ModelOptimizer
        
        optimizer = ModelOptimizer()
        
        # 配置优化参数
        config = {
            "model_file": ms_path,
            "output_file": output_path,
            "optimize_level": 2,  # 优化级别 0-3
        }
        
        result = optimizer.optimize(**config)
        
        if result:
            print(f"✅ 优化成功！")
            
            original_size = os.path.getsize(ms_path) / (1024 * 1024)
            optimized_size = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"原始大小: {original_size:.2f} MB")
            print(f"优化后: {optimized_size:.2f} MB")
            print(f"减少: {(1 - optimized_size/original_size) * 100:.1f}%")
            
            return True
        else:
            print("❌ 优化失败")
            return False
            
    except Exception as e:
        print(f"⚠️  优化跳过: {str(e)}")
        return False


def verify_model(ms_path):
    """
    验证 MS 模型文件
    
    Args:
        ms_path: MS 模型文件路径
    """
    
    print("\n" + "=" * 60)
    print("模型验证")
    print("=" * 60)
    print(f"\n模型文件: {ms_path}\n")
    
    if not os.path.exists(ms_path):
        print(f"❌ 文件不存在: {ms_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(ms_path)
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
    
    # 尝试验证模型
    try:
        from mindspore_lite import Model
        
        model = Model()
        ret = model.load_model(ms_path)
        
        if ret:
            print("✅ 模型验证通过")
            
            # 获取模型信息
            inputs = model.get_inputs()
            outputs = model.get_outputs()
            
            print(f"\n输入张量数量: {len(inputs)}")
            for i, tensor in enumerate(inputs):
                print(f"  输入 {i}: shape={tensor.shape()}, dtype={tensor.data_type()}")
            
            print(f"\n输出张量数量: {len(outputs)}")
            for i, tensor in enumerate(outputs):
                print(f"  输出 {i}: shape={tensor.shape()}, dtype={tensor.data_type()}")
            
            return True
        else:
            print("❌ 模型加载失败")
            return False
            
    except Exception as e:
        print(f"⚠️  无法验证: {str(e)}")
        print("但这不影响使用，可以直接部署到 Android")
        return True


def deploy_to_android(ms_path, android_project_path):
    """
    部署模型到 Android 项目
    
    Args:
        ms_path: MS 模型文件路径
        android_project_path: Android 项目路径
    """
    
    print("\n" + "=" * 60)
    print("部署到 Android 项目")
    print("=" * 60)
    
    assets_dir = os.path.join(android_project_path, "custommodel/src/main/assets")
    
    if not os.path.exists(assets_dir):
        print(f"❌ Android assets 目录不存在: {assets_dir}")
        return False
    
    # 复制模型文件
    dest_path = os.path.join(assets_dir, "style_classifier.ms")
    
    import shutil
    shutil.copy2(ms_path, dest_path)
    
    print(f"✅ 模型已复制到: {dest_path}")
    
    file_size = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")
    
    print("\n下一步：")
    print("1. 在 Android Studio 中打开项目")
    print("2. 修改 StyleClassifierExecutor.java 中的 TODO 部分")
    print("3. 编译并运行应用")
    
    return True


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="MindSpore 模型转换工具")
    parser.add_argument("--input", type=str, required=True, help="输入 AIR 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 MS 文件路径")
    parser.add_argument("--quantize", action="store_true", help="启用量化")
    parser.add_argument("--android-path", type=str, default=None, help="Android 项目路径")
    
    args = parser.parse_args()
    
    # 步骤 1: 转换模型
    success = convert_air_to_ms(args.input, args.output, args.quantize)
    
    if not success:
        print("\n❌ 转换失败，退出")
        return
    
    ms_path = args.output if args.output else args.input.replace(".air", ".ms")
    
    # 步骤 2: 验证模型
    verify_model(ms_path)
    
    # 步骤 3: 部署到 Android（如果指定了路径）
    if args.android_path:
        deploy_to_android(ms_path, args.android_path)
    
    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
