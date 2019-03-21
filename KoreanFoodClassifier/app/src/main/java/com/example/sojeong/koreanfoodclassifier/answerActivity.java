package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.app.AppComponentFactory;
import android.os.Bundle;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.TextView;

import java.util.HashMap;

public class answerActivity extends Activity {
    HashMap<String ,String> recipe_ko = TCP_client.recipe_ko;
    HashMap<String ,String> recipe_en = TCP_client.recipe_en;
    private ImageView img;
    private int no_cnt = 0;
    private CheckBox liked;
    private DbOpenHelper mDbOpenHelper;
    private String foodId, foodName, foodIngredients, foodPreparation, foodCooking;
    String[] tokens = {"food name", "food ingredients", "food preparation", "food cooking", "food krName", "food id", "predict percentage"};

    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.answer_dialog);
        img = (ImageView)findViewById(R.id.dialog_img);
        String dialog_foodID = recipe_ko.get(tokens[5]);
        RadioButton rb1 = (RadioButton)findViewById(R.id.radio1);
        RadioButton rb2 = (RadioButton)findViewById(R.id.radio2);


    }
}
