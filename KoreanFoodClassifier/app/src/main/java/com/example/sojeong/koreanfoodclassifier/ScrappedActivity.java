package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.database.Cursor;
import android.util.Log;

import java.util.ArrayList;

public class ScrappedActivity extends Activity {

    private ArrayAdapter<String> arrayAdapter;
    public static final int REQUEST_CODE_ANOTHER = 1002;

    static ArrayList<String> arrayIndex = new ArrayList<String>();
    static ArrayList<String> arrayData = new ArrayList<String>();
    private ListView listView;
    private DbOpenHelper mDbOpenHelper;

    //public ArrayList<String> getData() {
    //    return arrayData;
    //}

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1);
        //arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, arrayData);
        listView = (ListView) findViewById(R.id.recipe_list);
        listView.setAdapter(arrayAdapter);
        listView.setOnItemClickListener(onClickListener);

        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();
        mDbOpenHelper.insertColumn("김치","재료","준비","만들기");
        mDbOpenHelper.insertColumn("김밥","재료","준비","만들기");
        mDbOpenHelper.insertColumn("삼계탕","재료","준비","만들기");
        showDatabase();

    }

    public void showDatabase() {
        Cursor iCursor = mDbOpenHelper.sortColumn();
        arrayData.clear();
        arrayIndex.clear();
        while(iCursor.moveToNext()) {
            String tempIndex = iCursor.getString(iCursor.getColumnIndex("_id"));
            String tempName = iCursor.getString(iCursor.getColumnIndex("foodName"));

            String result = tempName;
            arrayData.add(result);
            arrayIndex.add(tempIndex);
        }
        arrayAdapter.clear();
        arrayAdapter.addAll(arrayData);
        arrayAdapter.notifyDataSetChanged();
    }

    private AdapterView.OnItemClickListener onClickListener = new AdapterView.OnItemClickListener() {
        @Override
        public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
            Intent intent = new Intent(getApplicationContext(), ClickedRecipeActivity.class);
            intent.putExtra("menu",arrayData.get(position));
            startActivityForResult(intent,REQUEST_CODE_ANOTHER);
        }
    };

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
                arrayData.remove(menu_name);
                arrayAdapter.notifyDataSetChanged();
            }
        }
    }
}


