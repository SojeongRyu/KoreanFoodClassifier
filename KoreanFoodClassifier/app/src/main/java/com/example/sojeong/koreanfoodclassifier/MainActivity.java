package com.example.sojeong.koreanfoodclassifier;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.sojeong.koreanfoodclassifier.util.ImageResizeUtils;
import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.TedPermission;
import com.soundcloud.android.crop.Crop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "KFC";
    public static final int REQUEST_CODE_ANOTHER = 1001;

    private Button btnTakePicture;
    private Button btnScrapped;
    private Button btnGallery;
    private Button btnInfo;

    private ImageView ivPicture;
    private String imagePath;

    private Boolean isPermission = true;

    private static final int PICK_FROM_ALBUM = 1;
    private static final int PICK_FROM_CAMERA = 2;

    private Boolean isCamera = false;
    private File tempFile;

    static final String[] LIST_MENU = {"LIST1", "LIST2", "LIST3"};

    @SuppressLint("WrongViewCast")
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tedPermission();
    }

    public void onTakePictureBtnClicked(View v) {
        // 권한 허용에 동의하지 않았을 경우 토스트를 띄움
        if(isPermission) takePhoto();
        else Toast.makeText(v.getContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
    }

    public void onGalleryBtnClicked(View v) {
        // 권한 허용에 동의하지 않았을 경우 토스트를 띄움
        if(isPermission) goToAlbum();
        else Toast.makeText(v.getContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
    }

    public void onScrapBtnClicked(View v){
       Intent intent = new Intent(getApplicationContext(), ScrappedActivity.class);
       startActivityForResult(intent, REQUEST_CODE_ANOTHER);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if(requestCode==REQUEST_CODE_ANOTHER){
            Toast toast = Toast.makeText(getBaseContext(),"onActivityResult 메소드가 호출됨. 요청코드 : "+requestCode + "결과코드 : "+resultCode, Toast.LENGTH_LONG);
            toast.show();

            if(resultCode == RESULT_OK){
                String name = intent.getExtras().getString("name");
                toast = Toast.makeText(getBaseContext(), "응답으로 전달된 name : "+name, Toast.LENGTH_LONG);
                toast.show();

                if(tempFile != null) {
                    if(tempFile.exists()) {
                        if(tempFile.delete()) {
                            Log.e(TAG, tempFile.getAbsolutePath() + "삭제 성공");
                            tempFile = null;
                        }
                    }
                }
                return;
            }
        }

        switch (requestCode) {
            case PICK_FROM_ALBUM: {
                Uri photoUri = intent.getData();
                Log.d(TAG, "PICK_FROM_ALBUM photoUri : " + photoUri);

                cropImage(photoUri);

                break;
            }
            case PICK_FROM_CAMERA: {
                Uri photoUri = Uri.fromFile(tempFile);
                Log.d(TAG, "takePhoto photoUri : " + photoUri);

                cropImage(photoUri);

                break;
            }
            case Crop.REQUEST_CROP: {
                /*
                    tempFile 에 크롭한 이미지를 이미 저장했기 때문에
                    tempFile을 크롭한 이미지라고 생각하고 사용하면 된다.
                 */
                sendImageToServer();
            }
        }
    }

    /*
       앨범에서 이미지 가져오기
    */
    private void goToAlbum() {
        isCamera = false;

        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, PICK_FROM_ALBUM);
    }

    /*
        카메라에서 이미지 가져오기
     */
    private void takePhoto() {
        isCamera = false;

        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        try {
            tempFile = createImageFile();
        } catch (IOException e) {
            Toast.makeText(this, "이미지 처리 오류! 다시 시도해주세요.", Toast.LENGTH_SHORT).show();
            finish();
            e.printStackTrace();
        }
        if(tempFile != null) {

            if(android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                Uri photoUri = FileProvider.getUriForFile(this, "{package name}.provider", tempFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(intent, PICK_FROM_CAMERA);
            } else {
                Uri photoUri = Uri.fromFile(tempFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(intent, PICK_FROM_CAMERA);
            }
        }
    }

    /*
        crop 기능
     */
    private void cropImage(Uri photoUri) {
        Log.d(TAG, "tempFile: " + tempFile);

        /*
            갤러리에서 선택한 경우에는 tempFile이 없으므로 새로 생성해 준다.
         */
        if(tempFile == null) {
            try {
                tempFile = createImageFile();
            } catch (IOException e) {
                Toast.makeText(this, "이미지 처리 오류! 다시 시도해주세요.", Toast.LENGTH_SHORT).show();
                finish();
                e.printStackTrace();
            }
        }

        // 크롭 후 저장할 Uri
        Uri savingUri = Uri.fromFile(tempFile);

        Crop.of(photoUri, savingUri).asSquare().start(this);
    }

    /*
        폴더 및 파일 만들기
     */
    private File createImageFile() throws IOException {
        // 이미지 파일 이름 (KFC_(시간)_)
        String timeStamp = new SimpleDateFormat("HHmmss").format(new Date());
        String imageFileName = TAG + "_" + timeStamp + "_";

        // 이미지가 저장될 폴더 이름 (KFC)
        File storageDir = new File(Environment.getExternalStorageDirectory() + "/" + TAG + "/");
        if(!storageDir.exists())    storageDir.mkdirs();

        // 파일 생성
        File image = File.createTempFile(imageFileName, "jpg", storageDir);
        Log.d(TAG, "createImageFile : " + image.getAbsolutePath());

        return image;
    }

    /*
        tempFile 을 bitmap 으로 변환 후 TCP 연결을 통해 서버로 보낸다
     */
    private void sendImageToServer() {
        ImageButton imageButton = findViewById(R.id.btnTakePicture);

        ImageResizeUtils.resizeFile(tempFile, tempFile, 1280, isCamera);

        BitmapFactory.Options options = new BitmapFactory.Options();
        Bitmap originalBm = BitmapFactory.decodeFile(tempFile.getAbsolutePath(), options);
        Log.d(TAG, "setImage: " + tempFile.getAbsolutePath());

        imageButton.setImageResource(R.drawable.full_star);

        tempFile = null;
    }

    /*
        권한 설정
     */
    private void tedPermission() {
        PermissionListener permissionListener = new PermissionListener() {
            @Override
            public void onPermissionGranted() {

            }

            @Override
            public void onPermissionDenied(List<String> deniedPermissions) {

            }
        };

        TedPermission.with(this)
                .setPermissionListener(permissionListener)
                .setRationaleMessage(getResources().getString(R.string.permission_2))
                .setDeniedMessage(getResources().getString(R.string.permission_1))
                .setPermissions(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA)
                .check();
    }

}
