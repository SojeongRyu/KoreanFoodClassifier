package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import android.os.Handler;
import android.os.Message;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.channels.Selector;
import java.util.*;
import android.widget.TextView;

import android.util.Log;

public class OutputActivity extends Activity {
    /*
    HashMap<String ,String> recipe_ko = TCP_client.recipe_ko;
    HashMap<String ,String> recipe_en = TCP_client.recipe_en;
    */
    private ImageView img;
    private int no_cnt = 0;
    private CheckBox liked;
    private DbOpenHelper mDbOpenHelper;
    private String foodId, foodName, foodIngredients, foodPreparation, foodCooking;
    String[] tokens = {"food name", "food ingredients", "food preparation", "food cooking", "food krName", "food id", "predict percentage"};

    HashMap<String ,String> recipe_ko;
    HashMap<String ,String> recipe_en;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recipe_output_display);

        Intent intent = getIntent();

        recipe_ko = (HashMap<String,String>)intent.getSerializableExtra("hash_ko");
        recipe_en = (HashMap<String,String>)intent.getSerializableExtra("hash_en");
        String dialog_foodID = recipe_ko.get(tokens[5]);

        foodInfo_show(dialog_foodID);


        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();

    }


    void foodInfo_show(String foodId) {
        final String systemLanguage = Locale.getDefault().getLanguage();
        ImageView imageView = (ImageView)findViewById(R.id.foodImg);
        String foodImgName = "food00" + foodId;
        Log.e("foodImgName", getPackageName());
        imageView.setImageResource(getResources().getIdentifier(foodImgName.trim(),"drawable",getPackageName()));
        Log.e("foodImg","food00"+foodId);

        this.foodId = foodId;
        if (systemLanguage == "ko") {
            TextView textView = (TextView)findViewById(R.id.foodName);
            this.foodName = recipe_ko.get(tokens[0]);
            textView.append(foodName);
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            this.foodIngredients = recipe_ko.get(tokens[1]);
            textView2.append(foodIngredients);
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            this.foodPreparation = recipe_ko.get(tokens[2]);
            textView3.append(foodPreparation);
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            this.foodCooking = recipe_ko.get(tokens[3]);
            textView4.append(foodCooking);
        }
        else if (systemLanguage == "en") {
            TextView textView = (TextView)findViewById(R.id.foodName);
            this.foodName = recipe_en.get(tokens[4]) + "(" + recipe_en.get(tokens[0]).trim() + ")";
            textView.append(foodName);
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            this.foodIngredients = recipe_en.get(tokens[1]);
            textView2.append(foodIngredients);
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            this.foodPreparation = recipe_en.get(tokens[2]);
            textView3.append(foodPreparation);
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            this.foodCooking = recipe_en.get(tokens[3]);
            textView4.append(foodCooking);
        }
        else {
            new Thread() {
                public void run() {
                    String originFoodName = recipe_en.get(tokens[0]);
                    foodName = getTranslatedString("en", systemLanguage, originFoodName);
                    foodName =  recipe_en.get(tokens[4]) + "(" + foodName.trim() + ")";
                    //foodName = foodName.replace("\r\n","\\n");
                    String originFoodIngredients = recipe_en.get(tokens[1]);
                    foodIngredients = getTranslatedString("en", systemLanguage, originFoodIngredients);
                    //foodIngredients = foodIngredients.replace("\r\n","\\n");
                    //Log.d("test", "번역결과2: "+foodName+"   ///    "+foodIngredients);
                    String originFoodPreparation = recipe_en.get(tokens[2]);
                    foodPreparation= getTranslatedString("en", systemLanguage, originFoodPreparation);
                    //foodPreparation = foodPreparation.replace("\r\n","\\n");
                    String originFoodCooking = recipe_en.get(tokens[3]);
                    foodCooking = getTranslatedString("en", systemLanguage, originFoodCooking);
                    //foodCooking = foodCooking.replace("\r\n","\\n");

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
            String tmpfoodName = bun.getString("foodName_DATA");
            TextView textView = (TextView)findViewById(R.id.foodName);
            textView.append(tmpfoodName);
            String tmpfoodIngredients = bun.getString("foodIngredients_DATA");
            TextView textView2 = (TextView)findViewById(R.id.foodIngredients);
            textView2.append(tmpfoodIngredients);
            //Log.d("test", "번역결과3: "+foodName+"   ///    "+foodIngredients);
            String tmpfoodPreparation = bun.getString("foodPreparation_DATA");
            TextView textView3 = (TextView)findViewById(R.id.foodPreparation);
            textView3.append(tmpfoodPreparation);
            String tmpfoodCooking = bun.getString("foodCooking_DATA");
            TextView textView4 = (TextView)findViewById(R.id.foodCooking);
            textView4.append(tmpfoodCooking);
        }
    };

    public Bitmap byteArrayToBitmap(byte[] byteArray) {
        return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
    }

    void sorry_dialog(){
        AlertDialog.Builder sorry_dialog = new AlertDialog.Builder(this);
        sorry_dialog.setMessage("I can not find you any other recipes.\nSorry for the inconvenience\n");
        sorry_dialog.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
                finish();
            }
        });
        sorry_dialog.show();
    }

    void dialog_show(final String dialog_foodID){
        Log.e("img", img.toString());
        try {
            if (img.getParent() != null)
                ((ViewGroup) img.getParent()).removeView(img);
        } catch (Exception e) {
            Log.e("img.getParent()", e.toString());
        }
        AlertDialog.Builder builder = new AlertDialog.Builder(this).setView(img);
        //setContentView(R.layout.check_img);
        builder.setCancelable(false);
        builder.setOnDismissListener(new DialogInterface.OnDismissListener() {
            @Override
            public void onDismiss(DialogInterface dialog) {
                Toast.makeText(getApplicationContext(),"Thank you", Toast.LENGTH_LONG).show();
            }
        });
        String firstImageName = "food00"+dialog_foodID;
        img.setImageResource(getResources().getIdentifier(firstImageName.trim(),"drawable",getPackageName()));
        builder.setMessage("I think your picture is " + TCP_client.recipe_ko.get(tokens[6]).trim() + " this food. Is it right?");
        builder.setPositiveButton("NO", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if(no_cnt == 0){
                    no_cnt++;
                    sendResponse(false, no_cnt);
                    dialog.dismiss();
                    recipe_ko = TCP_client.recipe_ko;
                    recipe_en = TCP_client.recipe_en;
                    String new_dialog_foodID = recipe_ko.get(tokens[5]);
                    dialog_show(new_dialog_foodID);
                }
                else {
                    no_cnt++;
                    sendResponse(false, no_cnt);
                    sorry_dialog();
                }
            }
        });
        builder.setNegativeButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                sendResponse(true, no_cnt);
                dialog.dismiss();
                foodInfo_show(dialog_foodID);
            }
        });
        AlertDialog alert = builder.create();
        alert.show();
        //builder.show();
    }

    private void sendResponse(boolean isYes, int noCnt) {
        TCP_client tcp_client = new TCP_client("203.153.146.10", 16161, null);
        tcp_client.startTCP(isYes, noCnt);
    }

    public void onBackBtnClicked(View v){
        ifChecked_scrap();
        finish();
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        ifChecked_scrap();
    }

    public void ifChecked_scrap() {
        liked = (CheckBox)findViewById(R.id.btn_selector);
        if(liked.isChecked()) {
            mDbOpenHelper.deleteColumn(foodName);
            mDbOpenHelper.insertColumn(foodId, foodName,foodIngredients, foodPreparation,foodCooking);
        }
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
        originString = originString.replace("\r\n","$");

        String clientId = "obY_tGKsObVUX_AY7b9u";//애플리케이션 수현클라이언트 아이디값";
        String clientSecret = "nEAjxFllry";//애플리케이션 수현클라이언트 시크릿값";

        //String clientId = "ZKSqbDudRKbISBGmZm1k";//애플리케이션 소정클라이언트 아이디값";
        //String clientSecret = "ZtEH03SRb0";//애플리케이션 소정클라이언트 시크릿값";

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
        translatedString = translatedString.replace("$","\n");
        return translatedString;
    }
}
