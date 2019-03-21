package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

public class listviewTestActivity extends Activity {
    private ArrayList<listviewItem_test> list;
    ListView listView;
    HashMap<String ,String> recipe_ko1 = TCP_client.recipe_ko;
    HashMap<String ,String> recipe_en1 = TCP_client.recipe_en;
    HashMap<String ,String> recipe_ko2;
    HashMap<String ,String> recipe_en2;

    private DbOpenHelper mDbOpenHelper;
    private String foodId, foodName, foodIngredients, foodPreparation, foodCooking;
    String[] tokens = {"food name", "food ingredients", "food preparation", "food cooking", "food krName", "food id", "predict percentage"};
    private int no_cnt=0;

    private String firstImageName;
    private String secondImageName;
    private String FoodName1;
    private String FoodName2;
    int current_position=0;

    private ListviewAdapter adapter;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.listview_test);

        adapter = new ListviewAdapter();

        listView = (ListView) findViewById(R.id.lv_test);
        listView.setAdapter(adapter);

        final String dialog_foodID_top1 = recipe_ko1.get(tokens[5]);

        FoodName1 = recipe_en1.get(tokens[0]);
        no_cnt++;
        sendResponse(false, no_cnt);
        recipe_ko2 = TCP_client.recipe_ko;
        recipe_en2 = TCP_client.recipe_en;
        final String dialog_foodID_top2 = recipe_ko2.get(tokens[5]);
        FoodName2 = recipe_en2.get(tokens[0]);
        firstImageName = "food00"+dialog_foodID_top1;
        secondImageName = "food00"+dialog_foodID_top2;

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                current_position=position;
            }
        });
        initList();

        Button btn = (Button)findViewById(R.id.OK_btn);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(current_position==0){
                    Intent intent = new Intent(getApplicationContext(), OutputActivity.class);
                    intent.putExtra("fooId", dialog_foodID_top1);
                    intent.putExtra("hash_ko",recipe_ko1);
                    intent.putExtra("hash_en",recipe_en1);
                    startActivity(intent);
                }
                else if(current_position==1){
                    Intent intent = new Intent(getApplicationContext(), OutputActivity.class);
                    intent.putExtra("foodId", dialog_foodID_top2);
                    intent.putExtra("hash_ko",recipe_ko2);
                    intent.putExtra("hash_en",recipe_en2);
                    startActivity(intent);
                }
                else{

                }
            }
        });

        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();

    }

    public void initList(){
        adapter.clearItem();
        adapter.notifyDataSetChanged();
        adapter.addItem(FoodName1, getResources().getIdentifier(firstImageName.trim(),"drawable",getPackageName()));
        adapter.addItem(FoodName2, getResources().getIdentifier(secondImageName.trim(),"drawable",getPackageName()));
    }

    private void sendResponse(boolean isYes, int noCnt) {
        TCP_client tcp_client = new TCP_client("203.153.146.10", 16161, null);
        tcp_client.startTCP(isYes, noCnt);
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
