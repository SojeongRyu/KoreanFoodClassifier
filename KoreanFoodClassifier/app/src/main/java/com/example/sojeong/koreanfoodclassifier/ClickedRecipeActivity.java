package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.TextView;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.ImageView;
import android.util.Log;

public class ClickedRecipeActivity extends Activity {
    Intent intent;
    String foodName, foodIngredients, foodPreparation, foodCooking;
    String foodImg_byteArray;
    TextView foodNameEdit, foodIngredientsEdit, foodPreparationEdit, foodCookingEdit;
    private CheckBox liked;
    private DbOpenHelper mDbOpenHelper;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrapped_recipe);

        intent = getIntent();
        foodName = intent.getStringExtra("foodName");
        foodNameEdit = (TextView) this.findViewById(R.id.foodName);
        foodNameEdit.append(foodName);

        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();

        Cursor iCursor = mDbOpenHelper.selectColumns();
        while(iCursor.moveToNext()) {
            String tempName = iCursor.getString(iCursor.getColumnIndex("foodName"));
            if (tempName.equals(foodName)) {
                foodIngredients = iCursor.getString(iCursor.getColumnIndex("foodIngredients"));
                foodIngredientsEdit = (TextView) this.findViewById(R.id.foodIngredients);
                foodIngredientsEdit.setText("Food Ingredients\n" + foodIngredients);
                foodPreparation = iCursor.getString(iCursor.getColumnIndex("foodPreparation"));
                foodPreparationEdit = (TextView) this.findViewById(R.id.foodPreparation);
                foodPreparationEdit.setText("Food Preparation\n" + foodPreparation);
                foodCooking = iCursor.getString(iCursor.getColumnIndex("foodCooking"));
                foodCookingEdit = (TextView) this.findViewById(R.id.foodCooking);
                foodCookingEdit.setText("Food Cooking\n" + foodCooking);
                /*
                foodImg_byteArray = iCursor.getString(iCursor.getColumnIndex("foodImg"));
                Bitmap foodImg = byteArrayToBitmap(foodImg_byteArray);
                ImageView imageView = (ImageView)findViewById(R.id.foodImg);
                imageView.setImageBitmap(foodImg);
                */
            }
        }

        liked = (CheckBox)findViewById(R.id.btn_selector);
        ImageButton btnBack = (ImageButton)findViewById(R.id.btn_back2);
        btnBack.setOnClickListener(new ImageButton.OnClickListener(){
            public void onClick(View v){
                if(!liked.isChecked()){
                    Intent resultIntent = new Intent();
                    resultIntent.putExtra("menu_name", foodName);
                    setResult(RESULT_OK, resultIntent);
                    Log.e("menu_name",foodName);
                }
                finish();
            }
        });


    }

    public Bitmap byteArrayToBitmap(byte[] byteArray) {
        return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
    }


}
