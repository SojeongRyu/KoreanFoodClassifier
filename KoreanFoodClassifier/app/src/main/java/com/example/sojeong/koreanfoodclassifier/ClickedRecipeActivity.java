package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.TextView;

public class ClickedRecipeActivity extends Activity {
    Intent intent;
    String menu_name;
    TextView edit;
    private CheckBox liked;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrapped_recipe);

        intent = getIntent();
        menu_name = intent.getStringExtra("menu");
        edit = (TextView) this.findViewById(R.id.textView1);
        edit.append(menu_name);

        liked = (CheckBox)findViewById(R.id.btn_selector);
        ImageButton btnBack = (ImageButton)findViewById(R.id.btn_back2);
        btnBack.setOnClickListener(new ImageButton.OnClickListener(){
            public void onClick(View v){
                if(!liked.isChecked()){
                    Intent resultIntent = new Intent();
                    resultIntent.putExtra("menu_name", menu_name);
                    setResult(RESULT_OK, resultIntent);
                }
                finish();
            }
        });


    }


}
