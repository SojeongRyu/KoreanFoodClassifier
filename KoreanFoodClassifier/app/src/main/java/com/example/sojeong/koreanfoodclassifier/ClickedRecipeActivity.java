package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.CheckBox;
import android.widget.TextView;

public class ClickedRecipeActivity extends Activity {
    Intent intent;
    String menu_name;
    String test;
    TextView edit;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrapped_recipe);
        intent = getIntent();
        menu_name = intent.getStringExtra("menu");
        edit = (TextView) this.findViewById(R.id.textView1);
        edit.append(menu_name);

        CheckBox liked = (CheckBox)findViewById(R.id.btn_selector);


    }


    public void onBackBtn2Clicked(View v) {
        finish();
    }

}
