# 📱 Android 部署完整指南

## ✅ 当前状态

- ✅ 模型训练完成
- ✅ 已导出 MINDIR 格式
- ✅ 已转换为 MS 格式
- ⏳ 需要部署到 Android

---

## 🚀 快速部署（3步完成）

### 步骤 1: 运行自动部署脚本

```powershell
cd model_training
.\deploy_to_android.bat
```

这会自动将 `style_classifier.ms` 复制到：
```
custommodel/src/main/assets/style_classifier.ms
```

---

### 步骤 2: 更新 Android 代码

已完成！我已经更新了 `StyleClassifierExecutor.java`：

```java
private static final int INPUT_IMAGE_SIZE = 32; // ArtBench-10: 32x32
```

**如果需要启用真实的 MindSpore Lite 推理**：

在 `StyleClassifierExecutor.java` 中取消注释以下代码：

```java
// 在文件顶部添加导入
import com.mindspore.lite.MSContext;
import com.mindspore.lite.Model;
import com.mindspore.lite.DeviceInfo;
import com.mindspore.lite.DeviceType;

// 在 loadModel() 方法中取消注释真实加载代码
public boolean loadModel(String modelPath) {
    try {
        Log.i(TAG, "Loading model from: " + modelPath);
        
        // 初始化 MSContext
        MSContext msContext = new MSContext();
        msContext.init();
        
        // 配置 CPU 信息
        DeviceInfo cpuDeviceInfo = new DeviceInfo();
        cpuDeviceInfo.setDeviceType(DeviceType.DT_CPU);
        cpuDeviceInfo.setThreadNum(4);
        msContext.addDeviceInfo(cpuDeviceInfo);
        
        // 创建并加载模型
        Model model = new Model();
        boolean ret = model.loadModel(modelPath, msContext);
        
        if (ret) {
            this.model = model;
            isModelLoaded = true;
            Log.i(TAG, "Model loaded successfully");
        }
        
        return ret;
    } catch (Exception e) {
        Log.e(TAG, "Error loading model: " + e.getMessage(), e);
        return false;
    }
}
```

---

### 步骤 3: 编译并运行

#### 方法 A: 使用 Android Studio（推荐）

1. 打开 Android Studio
2. File → Open → 选择项目根目录
3. 等待 Gradle 同步完成
4. 点击 Run 按钮（绿色三角形）
5. 选择设备或模拟器

#### 方法 B: 使用命令行

```powershell
# 返回项目根目录
cd ..

# 编译 Debug 版本
gradlew assembleDebug

# 安装到连接的设备
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## 🔍 验证部署

### 1. 检查文件是否正确复制

```powershell
dir custommodel\src\main\assets\style_classifier.ms
```

应该显示文件大小（通常几百 KB 到几 MB）。

### 2. 检查 Logcat 日志

运行应用后，在 Android Studio 的 Logcat 中查看：

```
I/StyleClassifier: Loading model from: file:///android_asset/style_classifier.ms
I/StyleClassifier: Model loaded successfully
```

### 3. 测试分类功能

1. 打开应用
2. 进入"图像风格分类"功能
3. 选择一张图片
4. 查看分类结果

---

## 🎨 使用示例

### 在 Activity 中使用

```java
// 创建执行器
StyleClassifierExecutor executor = new StyleClassifierExecutor(this);

// 加载模型
String modelPath = "file:///android_asset/style_classifier.ms";
boolean success = executor.loadModel(modelPath);

if (success) {
    Log.i(TAG, "模型加载成功");
    
    // 分类图片
    Bitmap image = ...; // 你的图片
    StyleResult result = executor.classify(image);
    
    Log.i(TAG, "风格: " + result.getStyle());
    Log.i(TAG, "置信度: " + result.getConfidence());
} else {
    Log.e(TAG, "模型加载失败");
}
```

---

## ⚠️ 常见问题

### Q1: 找不到 assets 目录

**解决**：手动创建
```powershell
mkdir custommodel\src\main\assets
```

### Q2: 模型加载失败

**可能原因**：
1. 文件路径错误
2. 模型文件损坏
3. MindSpore Lite 库未正确集成

**解决**：
```java
// 检查文件是否存在
AssetManager assetManager = context.getAssets();
try {
    InputStream is = assetManager.open("style_classifier.ms");
    Log.i(TAG, "文件存在，大小: " + is.available() + " bytes");
    is.close();
} catch (IOException e) {
    Log.e(TAG, "文件不存在", e);
}
```

### Q3: 分类结果不准确

**可能原因**：
1. 输入图像预处理不正确
2. 图像尺寸不匹配（应该是 32x32）
3. 归一化参数错误

**解决**：检查预处理代码
```java
// 确保图像 resize 到 32x32
Bitmap resized = Bitmap.createScaledBitmap(image, 32, 32, true);

// 确保正确的归一化
float[] mean = {0.485f, 0.456f, 0.406f};
float[] std = {0.229f, 0.224f, 0.225f};
```

### Q4: 应用崩溃

**检查**：
1. Logcat 中的错误信息
2. 是否添加了必要的权限
3. MindSpore Lite native 库是否正确加载

---

## 📊 性能优化

### 1. 使用 GPU 加速（如果支持）

```java
DeviceInfo gpuDeviceInfo = new DeviceInfo();
gpuDeviceInfo.setDeviceType(DeviceType.DT_GPU);
msContext.addDeviceInfo(gpuDeviceInfo);
```

### 2. 调整线程数

```java
cpuDeviceInfo.setThreadNum(4); // 根据设备 CPU 核心数调整
```

### 3. 量化模型

如果模型太大，可以使用量化版本：
```bash
# 转换时添加量化
converter --fmk=MINDIR \
          --modelFile=style_classifier.mindir \
          --outputFile=style_classifier_quant \
          --quantType=PostTrainingQuantization
```

---

## 🎯 完整流程总结

```
1. 训练模型
   ↓
   python train_with_artbench10.py
   
2. 转换为 MINDIR
   ↓
   （自动完成）
   
3. 转换为 MS
   ↓
   python convert_with_python.py
   
4. 部署到 Android
   ↓
   .\deploy_to_android.bat
   
5. 编译运行
   ↓
   gradlew assembleDebug
   
6. 测试验证
   ↓
   在设备上运行应用
```

---

## 📚 相关文档

- [MindSpore Lite Android 教程](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/android_quick.html)
- [模型转换指南](模型导出和转换指南.md)
- [优化方案](优化方案_提升准确率.md)

---

## ✅ 检查清单

部署前确认：

- [ ] `.ms` 文件已生成
- [ ] 文件已复制到 `assets` 目录
- [ ] `INPUT_IMAGE_SIZE` 已更新为 32
- [ ] MindSpore Lite 库已集成
- [ ] Android Studio 可以正常编译
- [ ] 设备已连接或模拟器已启动

---

## 🚀 立即开始

```powershell
# 1. 部署模型
.\deploy_to_android.bat

# 2. 编译项目
cd ..
gradlew assembleDebug

# 3. 安装到设备
adb install app/build/outputs/apk/debug/app-debug.apk

# 4. 运行并测试
```

**祝你好运！** 🎉
