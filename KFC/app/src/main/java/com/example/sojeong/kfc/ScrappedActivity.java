package com.example.sojeong.kfc;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import java.util.ArrayList;

public class ScrappedActivity extends Activity {


    private ArrayList<String> data = new ArrayList<>();

    public ArrayList<String> getData() {
        return data;
    }

    private ListView listView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        listView = (ListView)findViewById(R.id.recipe_list);
        String menu1 = "김치볶음밥";
        String menu2 = "계란말이";
        data.add(menu1);
        data.add(menu2);


        ArrayAdapter<String> Adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, data);
        listView.setAdapter(Adapter);

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent intent = new Intent(getApplicationContext(), ClickedRecipeActivity.class);
                intent.putExtra("menu",data.get(position));
                startActivity(intent);
            }
        });

    }

    public void onBackBtn1Clicked(View v){
        Intent resultIntent = new Intent();
        resultIntent.putExtra("name", "mike");

        setResult(RESULT_OK, resultIntent);
        finish();
    }
}

