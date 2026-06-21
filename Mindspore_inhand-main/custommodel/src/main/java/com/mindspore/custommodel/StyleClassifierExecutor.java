/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.custommodel;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * 图像风格分类模型执行器
 * 用于识别图像的艺术风格（如：油画、水彩、素描、写实等）
 */
public class StyleClassifierExecutor {

    private static final String TAG = "StyleClassifier";
    private static final int INPUT_IMAGE_SIZE = 32; // 模型输入尺寸（ArtBench-10: 32x32）

    private Context mContext;
    private boolean isModelLoaded = false;
    
    // MindSpore Lite 模型实例
    private Model model;
    private LiteSession session;
    private MSConfig msConfig;
    
    private final int NUM_THREADS = 4;
    private long fullExecutionTime;
    private long preProcessTime;
    private long inferenceTime;
    private long postProcessTime;

    // 风格标签（ArtBench-10 数据集）
    private static final String[] STYLE_LABELS = {
        "抽象表现主义",  // Abstract Expressionism
        "巴洛克",        // Baroque
        "表现主义",      // Expressionism
        "印象派",        // Impressionism
        "极简主义",      // Minimalism
        "波普艺术",      // Pop Art
        "写实主义",      // Realism
        "洛可可",        // Rococo
        "浪漫主义",      // Romanticism
        "文艺复兴"       // Renaissance
    };

    public StyleClassifierExecutor(Context context) {
        mContext = context;
        init();
    }

    /**
     * 初始化执行器
     */
    public void init() {
        Log.i(TAG, "StyleClassifierExecutor initialized");
    }

    /**
     * 加载模型文件
     * 先从 assets 复制到内部存储，再从文件路径加载，提高兼容性
     * @param assetPath assets 中的模型文件名（如 style_classifier_artbench.ms）
     * @return 是否加载成功
     */
    public boolean loadModel(String assetPath) {
        try {
            Log.i(TAG, "Loading model from assets: " + assetPath);

            // 将模型从 assets 复制到内部存储
            File modelFile = new File(mContext.getFilesDir(), assetPath);
            if (!modelFile.exists()) {
                Log.i(TAG, "Copying model from assets to: " + modelFile.getAbsolutePath());
                try (InputStream is = mContext.getAssets().open(assetPath);
                     FileOutputStream os = new FileOutputStream(modelFile)) {
                    byte[] buffer = new byte[8192];
                    int len;
                    while ((len = is.read(buffer)) != -1) {
                        os.write(buffer, 0, len);
                    }
                }
                Log.i(TAG, "Model copied to internal storage");
            }

            // 从文件路径加载模型
            model = new Model();
            boolean ret = model.loadModel(modelFile.getAbsolutePath());
            if (!ret) {
                Log.e(TAG, "Failed to load model: " + modelFile.getAbsolutePath());
                return false;
            }

            // 创建配置
            msConfig = new MSConfig();
            if (!msConfig.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU)) {
                Log.e(TAG, "MSConfig init failed");
                return false;
            }

            // 创建会话
            session = new LiteSession();
            if (!session.init(msConfig)) {
                Log.e(TAG, "LiteSession init failed");
                msConfig.free();
                return false;
            }
            msConfig.free();

            // 编译图
            if (!session.compileGraph(model)) {
                Log.e(TAG, "Compile graph failed");
                model.freeBuffer();
                return false;
            }
            model.freeBuffer();

            isModelLoaded = true;
            Log.i(TAG, "Model loaded successfully: " + assetPath);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error loading model: " + e.getMessage(), e);
            return false;
        }
    }

    /**
     * 执行风格分类推理
     * @param inputBitmap 输入图像
     * @return 分类结果
     */
    public StyleClassificationResult classify(Bitmap inputBitmap) {
        if (!isModelLoaded) {
            Log.e(TAG, "Model not loaded");
            return null;
        }

        Log.i(TAG, "Running style classification");
        fullExecutionTime = SystemClock.uptimeMillis();
        
        try {
            // 1. 预处理
            preProcessTime = SystemClock.uptimeMillis();
            ByteBuffer inputBuffer = preprocessImage(inputBitmap);
            preProcessTime = SystemClock.uptimeMillis() - preProcessTime;

            // 2. 推理
            inferenceTime = SystemClock.uptimeMillis();
            float[] outputData = runInference(inputBuffer);
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

            // 3. 后处理
            postProcessTime = SystemClock.uptimeMillis();
            StyleClassificationResult result = postprocess(outputData);
            postProcessTime = SystemClock.uptimeMillis() - postProcessTime;

            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime;

            Log.d(TAG, String.format("Classification completed in %dms (Pre: %dms, Inference: %dms, Post: %dms)",
                    fullExecutionTime, preProcessTime, inferenceTime, postProcessTime));

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error during classification: " + e.getMessage(), e);
            return null;
        }
    }

    /**
     * 运行推理（调用 MindSpore Lite API）
     */
    private float[] runInference(ByteBuffer inputBuffer) {
        // 获取输入张量
        List<MSTensor> inputs = session.getInputs();
        if (inputs == null || inputs.isEmpty()) {
            Log.e(TAG, "No input tensors found");
            return null;
        }
        
        MSTensor inputTensor = inputs.get(0);
        
        // 设置输入数据
        inputTensor.setData(inputBuffer);
        
        // 执行推理
        if (!session.runGraph()) {
            Log.e(TAG, "Inference failed");
            return null;
        }
        
        // 获取输出张量
        List<String> tensorNames = session.getOutputTensorNames();
        Map<String, MSTensor> outputs = session.getOutputMapByTensor();
        if (outputs == null || outputs.isEmpty()) {
            Log.e(TAG, "No output tensors found");
            return null;
        }
        
        float[] outputData = null;
        for (String tensorName : tensorNames) {
            MSTensor output = outputs.get(tensorName);
            if (output != null) {
                outputData = output.getFloatData();
                break;
            }
        }
        
        Log.d(TAG, "Inference completed, output size: " + outputData.length);
        return outputData;
    }

    /**
     * 图像预处理
     */
    private ByteBuffer preprocessImage(Bitmap bitmap) {
        // 调整图像尺寸
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        
        // 转换为 ByteBuffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();
        
        int[] intValues = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        resizedBitmap.getPixels(intValues, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        
        int pixel = 0;
        for (int y = 0; y < INPUT_IMAGE_SIZE; y++) {
            for (int x = 0; x < INPUT_IMAGE_SIZE; x++) {
                int value = intValues[pixel++];
                
                // 提取 RGB 通道并归一化到 [0, 1]
                float r = ((float) (value >> 16 & 255)) / 255.0f;
                float g = ((float) (value >> 8 & 255)) / 255.0f;
                float b = ((float) (value & 255)) / 255.0f;
                
                // 根据模型要求可能需要标准化（如减去均值）
                // r = (r - 0.485f) / 0.229f;
                // g = (g - 0.456f) / 0.224f;
                // b = (b - 0.406f) / 0.225f;
                
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
        }
        
        inputBuffer.rewind();
        return inputBuffer;
    }

    /**
     * 后处理：解析输出结果
     */
    private StyleClassificationResult postprocess(float[] outputData) {
        StyleClassificationResult result = new StyleClassificationResult();
        
        // 找到最高概率的类别
        int maxIndex = 0;
        float maxProb = outputData[0];
        for (int i = 1; i < outputData.length; i++) {
            if (outputData[i] > maxProb) {
                maxProb = outputData[i];
                maxIndex = i;
            }
        }
        
        // 设置主要结果
        result.setPredictedStyle(STYLE_LABELS[maxIndex]);
        result.setConfidence(maxProb);
        result.setExecutionTime(fullExecutionTime);
        
        // 设置所有类别的概率
        List<StyleProbability> probabilities = new ArrayList<>();
        for (int i = 0; i < outputData.length && i < STYLE_LABELS.length; i++) {
            StyleProbability sp = new StyleProbability();
            sp.setStyle(STYLE_LABELS[i]);
            sp.setProbability(outputData[i]);
            probabilities.add(sp);
        }
        result.setProbabilities(probabilities);
        
        return result;
    }

    /**
     * 释放资源
     */
    public void release() {
        isModelLoaded = false;
        if (session != null) {
            session.free();
            session = null;
        }
        if (model != null) {
            model.free();
            model = null;
        }
        Log.i(TAG, "StyleClassifierExecutor released");
    }

    public boolean isModelLoaded() {
        return isModelLoaded;
    }

    // ==================== 结果类 ====================

    /**
     * 风格分类结果
     */
    public static class StyleClassificationResult {
        private String predictedStyle;      // 预测的风格
        private float confidence;           // 置信度
        private long executionTime;         // 执行时间
        private List<StyleProbability> probabilities; // 所有风格的概率

        public String getPredictedStyle() {
            return predictedStyle;
        }

        public void setPredictedStyle(String predictedStyle) {
            this.predictedStyle = predictedStyle;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }

        public long getExecutionTime() {
            return executionTime;
        }

        public void setExecutionTime(long executionTime) {
            this.executionTime = executionTime;
        }

        public List<StyleProbability> getProbabilities() {
            return probabilities;
        }

        public void setProbabilities(List<StyleProbability> probabilities) {
            this.probabilities = probabilities;
        }
    }

    /**
     * 风格概率
     */
    public static class StyleProbability {
        private String style;
        private float probability;

        public String getStyle() {
            return style;
        }

        public void setStyle(String style) {
            this.style = style;
        }

        public float getProbability() {
            return probability;
        }

        public void setProbability(float probability) {
            this.probability = probability;
        }
    }
}
