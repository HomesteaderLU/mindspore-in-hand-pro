# 🎨 图像风格检测功能

> **项目状态**: ✅ 已集成  
> **最后更新**: 2026-06-22  
> **模块**: `custommodel`

---

## ✨ 功能简介

选择一张图片，AI 自动识别其所属的艺术风格（10 种风格），并显示置信度。

### 🎯 识别的 10 种风格（ArtBench-10）

| # | 风格 | 说明 |
|---|------|------|
| 1 | 🎭 **抽象表现主义** | Abstract Expressionism |
| 2 | 🏛️ **巴洛克** | Baroque |
| 3 | 🎨 **表现主义** | Expressionism |
| 4 | 🌅 **印象派** | Impressionism |
| 5 | ⬜ **极简主义** | Minimalism |
| 6 | 🎪 **波普艺术** | Pop Art |
| 7 | 📷 **写实主义** | Realism |
| 8 | 👑 **洛可可** | Rococo |
| 9 | 🌹 **浪漫主义** | Romanticism |
| 10 | ✝️ **文艺复兴** | Renaissance |

---

## 📁 相关文件

| 文件 | 说明 |
|------|------|
| `custommodel/.../StyleClassifierExecutor.java` | 模型执行器（预处理→推理→后处理） |
| `custommodel/.../StyleClassifierFragment.java` | 选择图片并显示结果的 UI 界面 |
| `custommodel/.../StyleClassifierActivity.java` | 包装 Fragment 的 Activity（ARouter 入口） |
| `custommodel/.../res/layout/fragment_style_classifier.xml` | 风格检测布局 |
| `model_training/train_with_artbench10.py` | 训练脚本 |

---

## 🧠 模型架构

### 网络结构（简化 CNN）

```
输入: 32×32×3 (RGB)
    ↓
Conv2D(3→32, 3×3, stride=2) + BatchNorm + ReLU   → 16×16×32
    ↓
Conv2D(32→64, 3×3, stride=2) + BatchNorm + ReLU   → 8×8×64
    ↓
Conv2D(64→128, 3×3, stride=2) + BatchNorm + ReLU  → 4×4×128
    ↓
GlobalAveragePooling                                → 128
    ↓
FullyConnected(128→10) + Softmax                   → 10 类概率
```

### 技术规格

| 属性 | 值 |
|------|-----|
| 输入尺寸 | **32 × 32 × 3**（适配 ArtBench-10 原始尺寸） |
| 输出类别 | 10 种艺术风格 |
| 参数体积 | ~200K（轻量级，适合移动端） |
| 推理模式 | **模拟模式**（当前返回随机结果，可用于 UI 体验） |
| 推理时间 | ~100-300ms（模拟模式） |
| 训练框架 | MindSpore / PyTorch |

> 💡 **切换到真实模型**: 去掉 `StyleClassifierExecutor.java` 中的 `TODO` 注释，加载 `.ms` 模型文件即可。

---

## 📱 界面说明

1. 在 **体验 → 高级区** 点击 **"图像风格检测"**
2. 点击 **"选择图片"** 从相册选取图片
3. 自动推理并显示：
   - 🏷️ 预测风格名称
   - 📊 置信度百分比
   - ⏱️ 推理耗时

---

## 🏗️ 代码架构

```
StyleClassifierActivity          ← ARouter 入口 /custommodel/StyleClassifierActivity
    └── StyleClassifierFragment   ← UI 布局 + 图片选择 + 结果展示
            └── StyleClassifierExecutor  ← 模型加载 + 预处理 + 推理 + 后处理
                    ├── loadModel(path)      ← 加载 .ms 模型
                    ├── classify(bitmap)     ← 执行完整推理流程
                    │     ├── preprocess()   ← Resize 32×32 + 归一化
                    │     ├── runInference() ← 调用 MindSpore Lite API
                    │     └── postprocess()  ← Softmax + 取最大值
                    └── release()            ← 释放资源
```

### 数据流

```
Bitmap → [Resize 32×32] → [Normalize] → [FloatBuffer] 
    → [MindSpore Lite Inference] → [float[10]] 
    → [ArgMax] → 风格名称 + 置信度
```

---

## 🔧 训练模型（可选）

如需训练真实模型，参考 `model_training/` 目录：

```bash
cd model_training
pip install -r requirements.txt
python train_with_artbench10.py
```

生成 `.ms` 文件后放入 Android 项目的 `assets/` 目录，并取消注释 `StyleClassifierExecutor.java` 中的真实推理代码。

---

## 🔍 常见问题

### Q1: 为什么显示随机结果？

**A**: 当前为**模拟模式**，用于 UI 体验。要使用真实模型：
1. 训练模型并转换为 `.ms` 格式
2. 放入 `app/src/main/assets/` 
3. 取消注释 `StyleClassifierExecutor.java` 中 `TODO` 部分的代码

### Q2: 输入图片尺寸要求？

**A**: 任何尺寸均可，代码会自动 Resize 到 32×32。

### Q3: 模型太大怎么办？

**A**: 当前简化 CNN 仅 ~200K 参数，远小于 MobileNetV2，非常适合移动端部署。

