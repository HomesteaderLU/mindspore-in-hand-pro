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
package com.mindspore.hms.ImageSegmentation;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.MLAnalyzerFactory;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentation;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationAnalyzer;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationScene;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationSetting;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * 证件照换底色 Activity
 * 复用 HMS ML Kit 人像分割能力 + 简单后处理合成纯色背景
 */
@Route(path = "/hms/IdPhotoActivity")
public class IdPhotoActivity extends AppCompatActivity {

    private static final String TAG = IdPhotoActivity.class.getSimpleName();
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    // 标准证件照底色 (ARGB)
    private static final int COLOR_RED = 0xFFC0362C;     // 红底
    private static final int COLOR_WHITE = 0xFFFFFFFF;   // 白底
    private static final int COLOR_BLUE = 0xFF4A6FF5;    // 蓝底

    private ImageView imgPreview, imgResult;
    private Uri imageUri;

    private Bitmap originBitmap;
    private Bitmap foregroundBitmap;   // 分割后的前景（透明背景人像）
    private MLImageSegmentationAnalyzer analyzer;

    private int selectedColor = COLOR_RED; // 默认红底

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_id_photo);

        createAnalyzer();
        init();
    }

    private void init() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(view -> finish());

        imgPreview = findViewById(R.id.img_preview);
        imgResult = findViewById(R.id.img_result);

        // 底色选择按钮
        findViewById(R.id.btn_color_red).setOnClickListener(v -> selectColor(COLOR_RED, (Button) v));
        findViewById(R.id.btn_color_white).setOnClickListener(v -> selectColor(COLOR_WHITE, (Button) v));
        findViewById(R.id.btn_color_blue).setOnClickListener(v -> selectColor(COLOR_BLUE, (Button) v));

        // 默认选中红底
        updateColorButtonState(COLOR_RED);
    }

    private void selectColor(int color, Button button) {
        selectedColor = color;
        updateColorButtonState(color);
        // 如果已有分割结果，重新合成
        if (foregroundBitmap != null) {
            composeWithBackground(foregroundBitmap);
        }
    }

    private void updateColorButtonState(int activeColor) {
        Button btnRed = findViewById(R.id.btn_color_red);
        Button btnWhite = findViewById(R.id.btn_color_white);
        Button btnBlue = findViewById(R.id.btn_color_blue);

        btnRed.setSelected(activeColor == COLOR_RED);
        btnWhite.setSelected(activeColor == COLOR_WHITE);
        btnBlue.setSelected(activeColor == COLOR_BLUE);
    }

    private void createAnalyzer() {
        MLImageSegmentationSetting setting = new MLImageSegmentationSetting.Factory()
                .setExact(false)
                .setAnalyzerType(MLImageSegmentationSetting.BODY_SEG)
                .setScene(MLImageSegmentationScene.ALL)
                .create();
        analyzer = MLAnalyzerFactory.getInstance().getImageSegmentationAnalyzer(setting);
    }

    public void onClickPhoto(View view) {
        openGallery(RC_CHOOSE_PHOTO);
    }

    public void onClickCamera(View view) {
        openCamera();
    }

    private void openGallery(int request) {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, request);
    }

    private void openCamera() {
        Intent intentToTakePhoto = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        String mTempPhotoPath = Environment.getExternalStorageDirectory() + File.separator + "photo.jpeg";
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider",
                new File(mTempPhotoPath));
        intentToTakePhoto.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intentToTakePhoto, RC_CHOOSE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    loadAndSegmentImage();
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                loadCameraImage();
            }
        }
    }

    private void loadAndSegmentImage() {
        try {
            originBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        if (originBitmap != null) {
            imgPreview.setImageBitmap(originBitmap);
            runSegmentation();
        }
    }

    private void loadCameraImage() {
        try {
            Pair<Integer, Integer> targetedSize = getTargetSize();
            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;
            Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            originBitmap = BitmapUtils.zoomImage(bitmap, targetWidth, maxHeight);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        if (originBitmap != null) {
            imgPreview.setImageBitmap(originBitmap);
            runSegmentation();
        }
    }

    private Pair<Integer, Integer> getTargetSize() {
        return new Pair<>(((View) imgPreview.getParent()).getWidth(),
                ((View) imgPreview.getParent()).getHeight());
    }

    /**
     * 执行人像分割
     */
    private void runSegmentation() {
        MLFrame frame = new MLFrame.Creator().setBitmap(originBitmap).create();
        Task<MLImageSegmentation> task = analyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(segmentation -> {
            if (segmentation != null) {
                Bitmap foreground = segmentation.getForeground();
                if (foreground != null) {
                    foregroundBitmap = foreground;
                    composeWithBackground(foreground);
                } else {
                    Log.e(TAG, "foreground is null");
                    Toast.makeText(IdPhotoActivity.this,
                            R.string.segmentation_fail, Toast.LENGTH_SHORT).show();
                }
            }
        }).addOnFailureListener(e -> {
            Log.e(TAG, "Segmentation failed: " + e.toString());
            Toast.makeText(IdPhotoActivity.this,
                    R.string.segmentation_fail, Toast.LENGTH_SHORT).show();
        });
    }

    /**
     * 核心后处理：将分割出的前景人像合成到纯色背景上
     *
     * 实现原理：
     * 1. 获取分割返回的 foreground Bitmap（透明背景的人像）
     * 2. 创建一张纯色背景图
     * 3. 将人像绘制到背景图上，利用透明通道只保留人像区域
     */
    private void composeWithBackground(Bitmap foreground) {
        if (foreground == null) return;

        int w = foreground.getWidth();
        int h = foreground.getHeight();

        // 1. 创建纯色背景图
        Bitmap result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(result);
        canvas.drawColor(selectedColor);

        // 2. 将前景人像绘制到背景上
        //    foreground 本身是透明背景的人像，直接绘制即可保留人像区域
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        canvas.drawBitmap(foreground, 0, 0, paint);

        // 3. 显示结果
        imgResult.setImageBitmap(result);
    }

    /**
     * 保存证件照到相册
     */
    public void onClickSave(View view) {
        if (foregroundBitmap == null) {
            Toast.makeText(this, R.string.no_pic_neededSave, Toast.LENGTH_SHORT).show();
            return;
        }
        // 重新合成当前颜色
        int w = foregroundBitmap.getWidth();
        int h = foregroundBitmap.getHeight();
        Bitmap finalBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(finalBitmap);
        canvas.drawColor(selectedColor);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        canvas.drawBitmap(foregroundBitmap, 0, 0, paint);

        BitmapUtils.saveToAlbum(getApplicationContext(), finalBitmap);
        Toast.makeText(this, R.string.save_success, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analyzer != null) {
            try {
                analyzer.stop();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
