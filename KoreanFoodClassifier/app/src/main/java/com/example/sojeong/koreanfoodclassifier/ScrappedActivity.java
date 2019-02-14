package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Adapter;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import java.util.ArrayList;

public class ScrappedActivity extends Activity {

    private ArrayAdapter<String> Adapter;
    public static final int REQUEST_CODE_ANOTHER = 1002;
    private ArrayList<String> datalist = new ArrayList<>();
    private ListView listView;

    public ArrayList<String> getData() {
        return datalist;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        listView = (ListView)findViewById(R.id.recipe_list);
        String menu1 = "김치볶음밥";
        String menu2 = "계란말이";
        String menu3 = "치킨";
        String menu4 = "삼계탕";
        String menu5 = "김밥";
        String menu6 = "비빔밥";
        String menu7 = "짜장면";
        String menu8 = "짬뽕";
        String menu9 = "짬짜면";
        String menu10 = "파스타";
        String menu11 = "피자";
        String menu12 = "쌀국수";
        String menu13 = "돈까스";
        String menu14 = "햄버거";
        String menu15 = "삼겹살";
        datalist.add(menu1);
        datalist.add(menu2);
        datalist.add(menu3);
        datalist.add(menu4);
        datalist.add(menu5);
        datalist.add(menu6);
        datalist.add(menu7);
        datalist.add(menu8);
        datalist.add(menu9);
        datalist.add(menu10);
        datalist.add(menu11);
        datalist.add(menu12);
        datalist.add(menu13);
        datalist.add(menu14);
        datalist.add(menu15);


        Adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, datalist);
        listView.setAdapter(Adapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent intent = new Intent(getApplicationContext(), ClickedRecipeActivity.class);
                intent.putExtra("menu",datalist.get(position));
                startActivityForResult(intent,REQUEST_CODE_ANOTHER);
            }
        });

    }

    public void onBackBtn1Clicked(View v){
        Intent resultIntent = new Intent();
        resultIntent.putExtra("name", "mike");

        setResult(RESULT_OK, resultIntent);
        finish();
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if(resultCode == RESULT_OK){
            if(requestCode==REQUEST_CODE_ANOTHER){
                String menu_name = intent.getExtras().getString("menu_name");
                datalist.remove(menu_name);
                Adapter.notifyDataSetChanged();
            }
        }
    }
}

