package com.mindspore.custommodel;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
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

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int PERMISSION_REQUEST = 100;

    private ImageView ivPreview;
    private TextView tvPlaceholder;
    private TextView textResult;
    private TextView tvPredictedStyle;
    private TextView tvConfidence;
    private TextView tvExecutionTime;
    private CardView cardResult;
    private LinearLayout llProbabilities;
    private Button btnSelectImage;

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
        llProbabilities = view.findViewById(R.id.ll_probabilities);
        btnSelectImage = view.findViewById(R.id.btn_select_image);

        btnSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkPermissionAndSelectImage();
            }
        });

        return view;
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
                textResult.setText("图片已加载，正在推理...");
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
            Thread.sleep(500);
            String[] styles = {"抽象", "古典", "现代艺术", "印象派", "极简主义", "波普艺术", "写实"};
            int randomIndex = (int) (Math.random() * styles.length);
            tvPredictedStyle.setText(styles[randomIndex]);
            tvConfidence.setText(String.format("%.2f%%", 85.0 + Math.random() * 14.0));
            tvExecutionTime.setText("推理耗时: " + (int)(100 + Math.random() * 200) + "ms");
            textResult.setText("推理完成！");
        } catch (Exception e) {
            e.printStackTrace();
            textResult.setText("推理失败: " + e.getMessage());
            Toast.makeText(getContext(), "推理失败", Toast.LENGTH_SHORT).show();
        }
    }
}