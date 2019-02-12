package com.example.sojeong.hello;

import android.annotation.SuppressLint;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;

public class MainActivity extends AppCompatActivity {

    private Button btnTakePicture;
    private Button btnScrapped;
    private Button btnGallery;
    private Button btnInfo;

    private ImageView ivPicture;
    private String imagePath;

    @SuppressLint("WrongViewCast")
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void onScrapBtnClicked(View v){
        setContentView(R.layout.scrap_display);
    }
    public void onBackBtn1Clicked(View v){
        setContentView(R.layout.activity_main);
    }
    public void onBackBtn2Clicked(View v){
        setContentView(R.layout.scrap_display);
    }





}
