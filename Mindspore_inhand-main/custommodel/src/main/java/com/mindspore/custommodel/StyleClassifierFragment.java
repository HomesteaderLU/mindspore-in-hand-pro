package com.mindspore.custommodel;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class StyleClassifierFragment extends Fragment {

    private static final String TAG = "StyleClassifierFragment";
    private static final String MODEL_NAME = "style_classifier_artbench.ms";
    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int PERMISSION_REQUEST = 100;

    private ImageView ivPreview;
    private TextView tvPlaceholder;
    private TextView textResult;
    private TextView tvPredictedStyle;
    private TextView tvConfidence;
    private TextView tvExecutionTime;
    private CardView cardResult;
    private Button btnSelectImage;

    private StyleClassifierExecutor classifierExecutor;
    private boolean isModelReady = false;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_style_classifier, container, false);

        ivPreview = view.findViewById(R.id.iv_preview);
        tvPlaceholder = view.findViewById(R.id.tv_placeholder);
        textResult = view.findViewById(R.id.text_result);
        tvPredictedStyle = view.findViewById(R.id.tv_predicted_style);
        tvConfidence = view.findViewById(R.id.tv_confidence);
        tvExecutionTime = view.findViewById(R.id.tv_execution_time);
        cardResult = view.findViewById(R.id.card_result);
        btnSelectImage = view.findViewById(R.id.btn_select_image);

        btnSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isModelReady) {
                    Toast.makeText(getContext(), "模型正在加载中，请稍候...", Toast.LENGTH_SHORT).show();
                    return;
                }
                checkPermissionAndSelectImage();
            }
        });

        // 初始化模型执行器并加载模型
        initModel();

        return view;
    }

    private void initModel() {
        textResult.setText("正在加载模型...");
        classifierExecutor = new StyleClassifierExecutor(getContext());
        boolean loaded = classifierExecutor.loadModel(MODEL_NAME);
        if (loaded) {
            isModelReady = true;
            textResult.setText("模型已就绪，请选择图片");
            Log.i(TAG, "Model loaded from assets: " + MODEL_NAME);
        } else {
            isModelReady = false;
            textResult.setText("模型加载失败，请确认模型文件存在");
            Toast.makeText(getContext(), "模型加载失败，请确认 assets 中存在 " + MODEL_NAME, Toast.LENGTH_LONG).show();
        }
    }

    private void checkPermissionAndSelectImage() {
        if (ContextCompat.checkSelfPermission(getContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(getActivity(),
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST);
        } else {
            selectImage();
        }
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && data != null && data.getData() != null) {
            Uri imageUri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getActivity().getContentResolver(), imageUri);
                ivPreview.setImageBitmap(bitmap);
                tvPlaceholder.setVisibility(View.GONE);
                cardResult.setVisibility(View.VISIBLE);
                textResult.setText("正在推理...");
                performInference(bitmap);
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(getContext(), "加载图片失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                selectImage();
            } else {
                Toast.makeText(getContext(), "需要存储权限才能选择图片", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void performInference(Bitmap bitmap) {
        try {
            if (!isModelReady || classifierExecutor == null) {
                textResult.setText("模型未就绪");
                return;
            }

            StyleClassifierExecutor.StyleClassificationResult result = classifierExecutor.classify(bitmap);
            if (result != null) {
                tvPredictedStyle.setText(result.getPredictedStyle());
                tvConfidence.setText(String.format("%.1f%%", result.getConfidence() * 100));
                tvExecutionTime.setText("推理耗时: " + result.getExecutionTime() + "ms");
                textResult.setText("推理完成！");
            } else {
                textResult.setText("推理失败：模型返回空结果");
                Toast.makeText(getContext(), "推理失败", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            e.printStackTrace();
            textResult.setText("推理失败: " + e.getMessage());
            Toast.makeText(getContext(), "推理失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (classifierExecutor != null) {
            classifierExecutor.release();
        }
    }
}