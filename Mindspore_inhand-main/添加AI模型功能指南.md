# 添加新 AI 模型功能指南

## 📋 概述

本指南展示了如何在 MindSpore 掌中宝项目中添加新的 AI 模型功能。我们以"图像风格分类"为例，演示完整的实现流程。

## ✅ 已完成的工作

### 1. 创建的文件

#### 核心代码文件
- `custommodel/src/main/java/com/mindspore/custommodel/StyleClassifierExecutor.java`
  - 模型执行器，负责加载模型、预处理、推理和后处理
  
- `custommodel/src/main/java/com/mindspore/custommodel/StyleClassifierFragment.java`
  - UI 界面，提供图片选择和结果显示功能

#### 布局文件
- `custommodel/src/main/res/layout/fragment_style_classifier.xml`
  - Fragment 的主布局
  
- `custommodel/src/main/res/layout/item_style_probability.xml`
  - 概率列表项布局

### 2. 修改的文件

- `app/src/main/java/com/mindspore/himindspore/comment/FragmentFactory.java`
  - 添加了 StyleClassifierFragment 的工厂方法

- `app/src/main/java/com/mindspore/himindspore/ui/main/MainActivity.java`
  - 在 ViewPager 中添加了新的 Fragment

- `app/src/main/res/values/strings.xml`
- `app/src/main/res/values-zh/strings.xml`
- `app/src/main/res/values-en/strings.xml`
  - 添加了"风格分类"的 Tab 标题

- `app/src/main/res/values/arrays.xml`
  - 添加了新的 Tab 图标配置

## 🚀 如何使用

### 方式一：使用模拟数据（当前状态）

当前代码使用模拟数据进行演示，可以直接运行查看效果：

1. **编译项目**
   ```bash
   ./gradlew assembleDebug
   ```

2. **安装到设备**
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

3. **运行应用**
   - 打开应用后，底部导航栏会显示 4 个 Tab
   - 点击"风格分类"Tab
   - 点击"选择图片"按钮
   - 从相册选择一张图片
   - 查看识别结果（当前为模拟数据）

### 方式二：集成真实的 MindSpore 模型

要使用真实的模型，需要以下步骤：

#### 步骤 1：准备模型文件

1. 使用 MindSpore 训练风格分类模型
2. 导出为 `.ms` 格式（MindSpore Lite 格式）
3. 将模型文件放到 `custommodel/src/main/assets/` 目录

```
custommodel/src/main/assets/
└── style_classifier.ms
```

#### 步骤 2：修改 StyleClassifierExecutor.java

取消注释并完善以下代码：

```java
// 在 loadModel() 方法中
public boolean loadModel(String modelPath) {
    try {
        // 初始化 MSContext
        MSContext msContext = new MSContext();
        msContext.init();
        
        // 配置 CPU 信息
        DeviceInfo cpuDeviceInfo = new DeviceInfo();
        cpuDeviceInfo.setDeviceType(DeviceType.DT_CPU);
        cpuDeviceInfo.setThreadNum(NUM_THREADS);
        msContext.addDeviceInfo(cpuDeviceInfo);
        
        // 加载模型
        model = new Model();
        boolean ret = model.loadModel(modelPath, msContext);
        
        if (ret) {
            isModelLoaded = true;
            Log.i(TAG, "Model loaded successfully: " + modelPath);
        }
        
        return ret;
    } catch (Exception e) {
        Log.e(TAG, "Error loading model: " + e.getMessage());
        return false;
    }
}

// 在 runInference() 方法中
private float[] runInference(ByteBuffer inputBuffer) {
    // 获取输入张量
    List<MSTensor> inputs = model.getInputs();
    MSTensor inputTensor = inputs.get(0);
    
    // 设置输入数据
    inputTensor.setData(inputBuffer.array());
    
    // 执行推理
    model.run();
    
    // 获取输出张量
    List<MSTensor> outputs = model.getOutputs();
    MSTensor outputTensor = outputs.get(0);
    
    // 返回输出数据
    return outputTensor.getFloatData();
}
```

#### 步骤 3：修改 StyleClassifierFragment.java

在 `initModel()` 方法中加载模型：

```java
private void initModel() {
    executor = new StyleClassifierExecutor(requireContext());
    
    // 从 assets 加载模型
    try {
        String modelPath = copyAssetToCache("style_classifier.ms");
        boolean loaded = executor.loadModel(modelPath);
        if (!loaded) {
            Toast.makeText(getContext(), "模型加载失败", Toast.LENGTH_SHORT).show();
        } else {
            Log.i(TAG, "模型加载成功");
        }
    } catch (Exception e) {
        Log.e(TAG, "Error loading model: " + e.getMessage(), e);
    }
}

// 辅助方法：将 assets 中的模型复制到缓存目录
private String copyAssetToCache(String assetName) throws IOException {
    InputStream inputStream = requireContext().getAssets().open(assetName);
    File cacheFile = new File(requireContext().getCacheDir(), assetName);
    
    FileOutputStream outputStream = new FileOutputStream(cacheFile);
    byte[] buffer = new byte[4096];
    int bytesRead;
    while ((bytesRead = inputStream.read(buffer)) != -1) {
        outputStream.write(buffer, 0, bytesRead);
    }
    
    outputStream.close();
    inputStream.close();
    
    return cacheFile.getAbsolutePath();
}
```

#### 步骤 4：添加依赖

确保 `custommodel/build.gradle` 中包含 MindSpore Lite 依赖：

```gradle
dependencies {
    // ... 其他依赖
    implementation project(':mindsporelibrary')
}
```

## 🎨 自定义新功能

如果你想添加其他类型的 AI 模型功能，可以参照以下步骤：

### 1. 确定模型类型

常见的模型类型：
- **图像分类**：识别图像类别
- **目标检测**：检测图像中的物体位置
- **语义分割**：像素级分类
- **风格迁移**：图像风格转换
- **姿态估计**：人体关键点检测
- **文本分类**：文本情感分析等
- **语音识别**：语音转文字

### 2. 创建执行器类

参考 `StyleClassifierExecutor.java` 的结构：

```java
public class YourModelExecutor {
    private static final String TAG = "YourModel";
    
    // 1. 初始化
    public void init() { }
    
    // 2. 加载模型
    public boolean loadModel(String modelPath) { }
    
    // 3. 预处理
    private ByteBuffer preprocessImage(Bitmap bitmap) { }
    
    // 4. 推理
    private float[] runInference(ByteBuffer inputBuffer) { }
    
    // 5. 后处理
    private YourResult postprocess(float[] outputData) { }
    
    // 6. 执行入口
    public YourResult execute(Bitmap inputBitmap) { }
    
    // 7. 释放资源
    public void release() { }
}
```

### 3. 创建 UI 界面

- 设计布局文件（XML）
- 创建 Fragment 或 Activity
- 实现用户交互逻辑

### 4. 注册到系统

- 在 `FragmentFactory` 中添加工厂方法
- 在 `MainActivity` 的 `initViewPager()` 中添加 Fragment
- 在资源文件中添加 Tab 配置

## 📝 注意事项

### 1. 模型兼容性

- MindSpore Lite 仅支持 ARM 架构设备
- x86 模拟器可能无法正常运行
- 确保模型输入输出格式与代码匹配

### 2. 性能优化

- 使用合适的线程数（通常 2-4 个）
- 避免在主线程执行推理
- 及时释放 Bitmap 和模型资源
- 考虑使用模型量化减小体积

### 3. 内存管理

```java
@Override
public void onDestroyView() {
    super.onDestroyView();
    if (executor != null) {
        executor.release();  // 释放模型
    }
    if (currentBitmap != null && !currentBitmap.isRecycled()) {
        currentBitmap.recycle();  // 回收 Bitmap
    }
}
```

### 4. 错误处理

- 始终检查模型是否加载成功
- 捕获推理过程中的异常
- 提供友好的错误提示

## 🔧 常见问题

### Q1: 模型加载失败？

**检查清单：**
- 模型文件格式是否正确（.ms）
- 模型路径是否正确
- MindSpore Lite 库是否正确引入
- 设备架构是否支持（ARM）

### Q2: 推理结果不正确？

**排查步骤：**
- 检查预处理是否与训练时一致
- 验证输入尺寸是否符合模型要求
- 确认归一化参数是否正确
- 检查输出解析逻辑

### Q3: 应用崩溃？

**常见原因：**
- 内存不足（大图片或大模型）
- 空指针异常（未初始化）
- 线程问题（UI 操作必须在主线程）

## 📚 参考资料

- [MindSpore 官方文档](https://www.mindspore.cn/)
- [MindSpore Lite 教程](https://www.mindspore.cn/lite)
- [Android NDK 开发指南](https://developer.android.com/ndk)

## 🎯 下一步

1. **测试功能**：在真实设备上测试
2. **优化性能**：调整模型参数和线程数
3. **美化 UI**：根据设计风格调整界面
4. **添加更多功能**：如历史记录、分享等

---

**祝你开发顺利！** 🚀
