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

import android.os.Handler;
import android.os.Message;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;
import android.widget.TextView;

import android.util.Log;

public class OutputActivity extends Activity {

    private String response;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        response = getIntent().getStringExtra("response");
        setContentView(R.layout.recipe_output_display);
        foodInfo_show();
        ImageButton ok_button = (ImageButton)findViewById(R.id.ok_button);
        ok_button.setOnClickListener(new ImageButton.OnClickListener(){
            @Override
            public void onClick(View v) {
                dialog_show();

            }
        });
    }


    void foodInfo_show() {
        final String systemLanguage = Locale.getDefault().getLanguage();

        if (systemLanguage == "ko") {
            TextView textView = (TextView)findViewById(R.id.foodName);
            textView.setText("김치"); // textView.setText(foodName);
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            textView2.setText("김치재료"); //textView2.setText(foodIngredients);
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            textView3.setText("김치만들기 준비"); //textView3.setText(foodPreparation);
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            textView4.setText("김치만들기"); //textView4.setText(foodCooking);
        }
        else if (systemLanguage == "en") {
            TextView textView = (TextView)findViewById(R.id.foodName);
            textView.setText("This is the name of Kimchi." + response); // textView.setText(foodName);
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            textView2.setText("This is the ingredients of Kimchi."); //textView2.setText(foodIngredients);
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            textView3.setText("This is the preparation of Kimchi."); //textView3.setText(foodPreparation);
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            textView4.setText("This is the cooking of Kimchi."); //textView4.setText(foodCooking);
        }
        else {
            new Thread() {
                public void run() {
                    String originFoodName = "This is the name of Kimchi.";
                    String foodName = getTranslatedString("en", systemLanguage, originFoodName);
                    String originFoodIngredients = "This is the ingredients of Kimchi.";
                    String foodIngredients = getTranslatedString("en", systemLanguage, originFoodIngredients);
                    //Log.d("test", "번역결과2: "+foodName+"   ///    "+foodIngredients);
                    String originFoodPreparation = "This is the preparation of Kimchi.";
                    String foodPreparation= getTranslatedString("en", systemLanguage, originFoodPreparation);
                    String originFoodCooking = "This is the cooking of Kimchi.";
                    String foodCooking = getTranslatedString("en", systemLanguage, originFoodCooking);

                    Bundle bun = new Bundle();
                    bun.putString("foodName_DATA",foodName);
                    bun.putString("foodIngredients_DATA", foodIngredients);
                    bun.putString("foodPreparation_DATA", foodPreparation);
                    bun.putString("foodCooking_DATA", foodCooking);

                    Message msg = handler.obtainMessage();
                    msg.setData(bun);
                    handler.sendMessage(msg);
                }
            }.start();
        }
    }

    Handler handler = new Handler() {
        public void handleMessage(Message msg) {
            Bundle bun = msg.getData();
            String foodName = bun.getString("foodName_DATA");
            TextView textView = (TextView)findViewById(R.id.foodName);
            textView.setText(foodName);
            String foodIngredients = bun.getString("foodIngredients_DATA");
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            textView2.setText(foodIngredients);
            //Log.d("test", "번역결과3: "+foodName+"   ///    "+foodIngredients);
            String foodPreparation = bun.getString("foodPreparation_DATA");
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            textView3.setText(foodPreparation);
            String foodCooking = bun.getString("foodCooking_DATA");
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            textView4.setText(foodCooking);
        }
    };

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


    private String getTranslatedString(String sourceLanguage, String targetLanguage, String originString) { //APITranslateNMT
        String translatedString = "Text for Translated String";

        String clientId = "obY_tGKsObVUX_AY7b9u";//애플리케이션 클라이언트 아이디값";
        String clientSecret = "nEAjxFllry";//애플리케이션 클라이언트 시크릿값";
        try {
            String text = URLEncoder.encode(originString, "UTF-8");
            String apiURL = "https://openapi.naver.com/v1/papago/n2mt";
            URL url = new URL(apiURL);
            HttpURLConnection con = (HttpURLConnection)url.openConnection();
            con.setRequestMethod("POST");
            con.setRequestProperty("X-Naver-Client-Id", clientId);
            con.setRequestProperty("X-Naver-Client-Secret", clientSecret);
            // post request
            String postParams = "source="+sourceLanguage+"&target="+targetLanguage+"&text=" + text;
            con.setDoOutput(true);
            DataOutputStream wr = new DataOutputStream(con.getOutputStream());
            wr.writeBytes(postParams);
            wr.flush();
            wr.close();
            int responseCode = con.getResponseCode();
            BufferedReader br;
            if(responseCode==200) { // 정상 호출
                br = new BufferedReader(new InputStreamReader(con.getInputStream()));
            } else {  // 에러 발생
                br = new BufferedReader(new InputStreamReader(con.getErrorStream()));
            }
            String inputLine;
            StringBuffer response = new StringBuffer();
            while ((inputLine = br.readLine()) != null) {
                response.append(inputLine);
            }
            br.close();
            //System.out.println(response.toString());
            //translatedString = response.toString();
            String responseString = response.toString();
            Log.d("test", "응답메세지"+responseString);
            String target = "\"translatedText\":";
            int target_num = responseString.indexOf(target);
            translatedString = responseString.substring(target_num+target.length()+1,responseString.length()-4);
            Log.d("test", "번역결과"+translatedString);
        } catch (Exception e) {
            System.out.println(e+"error");
            translatedString = e.toString();
        }
        return translatedString;
    }
}
