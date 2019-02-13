package com.example.sojeong.koreanfoodclassifier;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    public static final int REQUEST_CODE_ANOTHER = 1001;
    private Button btnTakePicture;
    private Button btnScrapped;
    private Button btnGallery;
    private Button btnInfo;

    private ImageView ivPicture;
    private String imagePath;

    static final String[] LIST_MENU = {"LIST1", "LIST2", "LIST3"};

    @SuppressLint("WrongViewCast")
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
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
            }
        }
    }


}
