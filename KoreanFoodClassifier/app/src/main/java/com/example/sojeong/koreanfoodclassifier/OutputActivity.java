package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.media.Image;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.Toast;

public class OutputActivity extends Activity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recipe_output_display);

        ImageButton ok_button = (ImageButton)findViewById(R.id.ok_button);
        ok_button.setOnClickListener(new ImageButton.OnClickListener(){
            @Override
            public void onClick(View v) {
                dialog_show();

            }
        });
    }

    void dialog_show(){
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        //builder.setTitle("dd");
        builder.setMessage("Are you satisfy the recipe?");
        builder.setPositiveButton("NO", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent intent_No = new Intent();
                intent_No.putExtra("answer_code","1");
                setResult(100,intent_No);
                Toast.makeText(getApplicationContext(),"Thank you", Toast.LENGTH_SHORT).show();
                finish();
            }
        });
        builder.setNegativeButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent intent_Yes = new Intent();
                intent_Yes.putExtra("answer_code","0");
                setResult(101,intent_Yes);
                Toast.makeText(getApplicationContext(),"Thank you", Toast.LENGTH_SHORT).show();
                finish();
            }
        });
        builder.show();
    }

    public void onBackBtnClicked(View v){
        finish();
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);
        if(resultCode==100){
            Toast.makeText(getApplicationContext(),"resultCode: "+resultCode,Toast.LENGTH_LONG).show();
        }
        else if(resultCode==101){
            Toast.makeText(getApplicationContext(),"resultCode: "+resultCode,Toast.LENGTH_LONG).show();
        }
    }
}
