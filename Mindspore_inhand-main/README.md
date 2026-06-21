# MindSpore 掌中宝

基于 MindSpore Lite 的 Android AI 演示应用，集成多种 AI 模型能力，提供直观的交互体验。

## 🚀 功能总览

### 📸 体验区 · 视觉图像技术

| 功能 | 说明 |
|------|------|
| 🦴 **骨骼检测** | 识别摄像头中人体面部五官与肢体姿势 |
| 👤 **人像分割** | 检测并分割出图片中的人像区域 |
| 🏷️ **图像分类** | 推测图片中物体的所属类别 |
| ✋ **手势识别** | 识别摄像头中的手势动作 |

### 🎤 体验区 · 语音语义技术

| 功能 | 说明 |
|------|------|
| 📝 **文字识别** | 识别图片中的文字信息 |
| 🌐 **文本翻译** | 多语言文本互译 |

### ⚡ 高级区 · 视觉图像技术

| 功能 | 说明 |
|------|------|
| 🖼️ **对象检测(图片)** | 识别图中物体类别并标示位置 |
| 🎥 **对象检测(实时)** | 实时识别摄像头中物体 |
| 🔄 **背景替换** | 基于人像分割替换图片背景 |
| 🎨 **图像风格检测** | 识别图像所属的艺术风格 |
| 🎭 **风格迁移** | 将图片转换为不同艺术风格 |
| 🆔 **证件照换底色** | 一键更换证件照为红底/蓝底/白底 |

## 📚 功能文档

| 文档 | 说明 |
|------|------|
| 📄 [`README_图像风格分类.md`](./README_图像风格分类.md) | 🎨 图像风格检测详细说明（模型架构、代码结构、训练指南） |
| 📄 [`README_换底功能.md`](./README_换底功能.md) | 🔄 换底功能详细说明（背景替换 + 证件照换底色） |

## 🏗️ 项目结构

```
app/              — 主应用模块（导航、UI）
common/           — 公共基础组件
custommodel/      — 自定义模型加载与推理
customView/       — 自定义 UI 组件
dance/            — 舞蹈相关功能
hms/              — 华为 HMS ML Kit 集成
imageObject/      — 图像目标检测
mindsporelibrary/ — MindSpore Lite 运行时库
styletransfer/    — 风格迁移模型执行
model_training/   — 模型训练脚本（Python）
```

## 🛠️ 构建指南

1. 安装 Android Studio 并打开项目根目录
2. 确保已配置 Android SDK 30 和 NDK 21.3.6528147
3. 连接 Android 设备或启动模拟器（需 ARM 架构）
4. 运行：
   ```bash
   .\gradlew assembleDebug
   ```
5. APK 生成路径：`app/build/outputs/apk/debug/app-debug.apk`

## ⚠️ 注意事项

- 模型推理仅支持 **ARM 架构** 的 Android 设备，x86 模拟器无法加载
- 部分功能依赖华为 HMS ML Kit，需华为设备或安装 HMS Core
- 舞蹈相关功能（横屏适配）存在显示问题
- Quick Start 内置网页链接已失效

## 📦 技术栈

- **AI 框架**: MindSpore Lite / Huawei ML Kit
- **路由**: ARouter
- **网络**: Retrofit2 + OkHttp3 + RxJava2
- **图片**: Glide
- **事件**: EventBus
- **构建**: Gradle 7.5 + Android Gradle Plugin 7.4.2


