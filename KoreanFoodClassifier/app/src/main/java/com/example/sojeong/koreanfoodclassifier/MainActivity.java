package com.example.sojeong.koreanfoodclassifier;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.TedPermission;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public SharedPreferences prefs;
    private static final String TAG = "KFC";
    public static final int REQUEST_CODE_ANOTHER = 1001;

    private Button btnTakePicture;
    private Button btnScrapped;
    private Button btnGallery;
    private Button btnInfo;

    private ImageView ivPicture;
    private String imagePath;

    static final String[] LIST_MENU = {"LIST1", "LIST2", "LIST3"};

    private Boolean isPermission = true;
    private static final int REQUEST_TAKE_PHOTO = 1111;
    private static final int REQUEST_TAKE_ALBUM = 2222;
    private static final int REQUEST_IMAGE_CROP = 3333;
    private String mCurrentPhotoPath;
    private Uri photoURI, albumURI;
    private boolean isAlbum = false;

    @SuppressLint("WrongViewCast")
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prefs = getSharedPreferences("Pref",MODE_PRIVATE);
        checkFirstRun();
        tedPermission();
    }

    public void checkFirstRun(){
        boolean isFirstRun = prefs.getBoolean("isFirstRun",true);
        if(isFirstRun){
            final AlertDialog.Builder first_dialog = new AlertDialog.Builder(this);
            first_dialog.setCancelable(false);
            first_dialog.setTitle("Terms of service");
            first_dialog.setMessage("We will use your pictures for noncommercial purposes.\nIf you do not agree with our policy, please stop using this application");
            first_dialog.setPositiveButton("I got it", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                }
            });
            first_dialog.show();
            prefs.edit().putBoolean("isFirstRun",false).apply();
        }
    }

    public void onTakePictureBtnClicked(View v) {
        // 권한 허용에 동의하지 않았을 경우 토스트를 띄움
        if (isPermission) takePhoto();
        else Toast.makeText(v.getContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
    }

    public void onGalleryBtnClicked(View v) {
        // 권한 허용에 동의하지 않았을 경우 토스트를 띄움
        if (isPermission) goToAlbum();
        else Toast.makeText(v.getContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
    }

    public void onScrapBtnClicked(View v){
        Intent intent = new Intent(getApplicationContext(), ScrappedActivity.class);
        startActivityForResult(intent, REQUEST_CODE_ANOTHER);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case REQUEST_TAKE_PHOTO: {
                    try {
                        galleryAddPic();
                    } catch (Exception e) {
                        Log.e("Photo Save Error", e.toString());
                    }
                    cropImage(photoURI);
                    break;
                }
                case REQUEST_TAKE_ALBUM: {
                    isAlbum = true;
                    albumURI = intent.getData();
                    cropImage(albumURI);
                    break;
                }
                case REQUEST_IMAGE_CROP: {
                    if (isAlbum)
                        sendImageToServer(albumURI);
                    else
                        sendImageToServer(photoURI);
                    break;
                }
            }

            if (requestCode == REQUEST_CODE_ANOTHER) {
                Toast toast = Toast.makeText(getBaseContext(), "onActivityResult method called. Request code : " + requestCode + "Result code : " + resultCode, Toast.LENGTH_LONG);
                toast.show();

                if (resultCode == RESULT_OK) {
                    String name = intent.getExtras().getString("name");
                    toast = Toast.makeText(getBaseContext(), "name passed in response : " + name, Toast.LENGTH_LONG);
                    toast.show();

                    return;
                }
            }
        }
    }

    private void takePhoto() {  // 카메라에서 이미지 가져오기
        String state = Environment.getExternalStorageState();
        // 외장 메모리 검사
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                File photoFile = null;
                try {
                    photoFile = createImageFile();
                } catch (IOException e) {
                    Log.e("captureCamera Error", e.toString());
                }
                if (photoFile != null) {
                    Log.i("getPackageName", getPackageName());
                    Uri providerURI = FileProvider.getUriForFile(this, getPackageName(), photoFile);
                    photoURI = providerURI;
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, providerURI);
                    startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
                }
            }
        } else {
            Toast.makeText(this, "Cannot access the storage space of this device.", Toast.LENGTH_SHORT).show();
            return;
        }
    }

    private File createImageFile() throws IOException { // 폴더 및 파일 만들기
        // 이미지 파일 이름 (KFC_(시간)_)
        String timeStamp = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date());
        String imageFileName = TAG + "_" + timeStamp + ".jpg";
        File storageDir = new File(Environment.getExternalStorageDirectory() + "/Pictures", "KFC");

        if (!storageDir.exists()) {
            Log.i("mCurrentPhotoPath", storageDir.toString());
            storageDir.mkdirs();
        }

        File imageFile = new File(storageDir, imageFileName);
        mCurrentPhotoPath = imageFile.getAbsolutePath();

        return imageFile;
    }

    private void goToAlbum() {  // 앨범에서 이미지 가져오기
        Log.i("getAlbum", "Call");
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, REQUEST_TAKE_ALBUM);
    }

    private void galleryAddPic() {
        Log.i("galleryAddPic", "Call");
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        // 해당 경로에 있는 파일을 객체화
        File f = new File(mCurrentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        sendBroadcast(mediaScanIntent);
        Toast.makeText(this, "Picture saved.", Toast.LENGTH_SHORT).show();
    }

    private void cropImage(Uri uri) {  // crop 기능
        Log.i("cropImage", "Call");
        Log.i("cropImage", "photoURI: " + photoURI + " / albumURI: " + albumURI);

        Intent cropIntent = new Intent("com.android.camera.action.CROP");

        // 50 * 50 픽셀 미만은 편집할 수 없다는 문구 처리 + 갤러리, 포토 둘 다 호환하는 방법
        cropIntent.setDataAndType(uri, "image/*");
        cropIntent.putExtra("scale", true);
        if (isAlbum)    cropIntent.putExtra("output", albumURI);
        else            cropIntent.putExtra("output", photoURI);
        startActivityForResult(cropIntent.setData(uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION), REQUEST_IMAGE_CROP);
    }

    private void sendImageToServer(Uri img) {  //이미지를 TCP 연결을 통해 서버로 보내고 결과 화면 Activity로 전환한다.
        File imgFile = null;
        if (isAlbum) {
            Cursor cursor = getContentResolver().query(img, null, null, null, null);
            cursor.moveToNext();
            imgFile = new File(cursor.getString(cursor.getColumnIndex("_data")));
        }
        else
            imgFile = new File(Environment.getExternalStorageDirectory() + img.getPath());

        TCP_client tcp_client = new TCP_client("203.153.146.10", 16161, imgFile);
        tcp_client.startTCP(false, 0);

        albumURI = null;
        photoURI = null;
        mCurrentPhotoPath = null;
        isAlbum = false;

        Intent intent = new Intent(getApplicationContext(), OutputActivity.class);
        startActivity(intent);
    }

    private void tedPermission() {  // 권한 설정
        PermissionListener permissionListener = new PermissionListener() {
            @Override
            public void onPermissionGranted() { }

            @Override
            public void onPermissionDenied(List<String> deniedPermissions) { }
        };

        TedPermission.with(this)
                .setPermissionListener(permissionListener)
                .setRationaleMessage(getResources().getString(R.string.permission_2))
                .setDeniedMessage(getResources().getString(R.string.permission_1))
                .setPermissions(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA)
                .check();
    }

}
