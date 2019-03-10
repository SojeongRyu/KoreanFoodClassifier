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
import android.widget.ImageView;
import android.util.Log;

public class ClickedRecipeActivity extends Activity {
    Intent intent;
    String foodId, foodName, foodIngredients, foodPreparation, foodCooking;
    TextView foodNameEdit, foodIngredientsEdit, foodPreparationEdit, foodCookingEdit;
    ImageView imageView;
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
                foodId = iCursor.getString(iCursor.getColumnIndex("foodId"));
                ImageView imageView = (ImageView)findViewById(R.id.foodImg);
                String foodImgName = "food00" + foodId;
                imageView.setImageResource(getResources().getIdentifier(foodImgName.trim(),"drawable",getPackageName()));
                foodIngredients = iCursor.getString(iCursor.getColumnIndex("foodIngredients"));
                foodIngredientsEdit = (TextView) this.findViewById(R.id.foodIngredients);
                foodIngredientsEdit.setText(foodIngredients);
                foodPreparation = iCursor.getString(iCursor.getColumnIndex("foodPreparation"));
                foodPreparationEdit = (TextView) this.findViewById(R.id.foodPreparation);
                foodPreparationEdit.setText(foodPreparation);
                foodCooking = iCursor.getString(iCursor.getColumnIndex("foodCooking"));
                foodCookingEdit = (TextView) this.findViewById(R.id.foodCooking);
                foodCookingEdit.setText(foodCooking);
            }
        }

    }

    public void onBackBtn2Clicked(View v) {
        checkLikedAndPassFoodName();
        finish();
    }

    @Override
    public void onBackPressed() {
        checkLikedAndPassFoodName();
        super.onBackPressed();
    }

    public void checkLikedAndPassFoodName() {
        liked = (CheckBox)findViewById(R.id.btn_selector);
        if(!liked.isChecked()) {
            Intent resultIntent = new Intent();
            resultIntent.putExtra("menu_name", foodName);
            setResult(RESULT_OK, resultIntent);
            Log.e("menu_name", foodName);
        }
    }
}
