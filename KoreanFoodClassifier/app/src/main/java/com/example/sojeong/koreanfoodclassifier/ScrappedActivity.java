package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.database.Cursor;

import java.util.ArrayList;
import java.util.List;
import android.util.Log;

public class ScrappedActivity extends Activity {

    private SearchAdapter searchAdapter;
    public static final int REQUEST_CODE_ANOTHER = 1002;

    static ArrayList<String> arrayData = new ArrayList<String>();
    private ListView listView;
    private DbOpenHelper mDbOpenHelper;
    private EditText editSearch;
    private List<String> list;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        list = new ArrayList<String>();
        editSearch = (EditText) findViewById(R.id.editSearch);
        listView = (ListView) findViewById(R.id.recipe_list);
        listView.setOnItemClickListener(onClickListener);
        searchAdapter = new SearchAdapter(list, this);
        listView.setAdapter(searchAdapter);

        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();

        initializeArrayData();

        editSearch.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

            }

            @Override
            public void afterTextChanged(Editable s) {
                String text = editSearch.getText().toString();
                search(text);
            }
        });

    }

    public void search(String charText) {
        list.clear();

        if (charText.length() == 0) {
            list.addAll(arrayData);
        }
        else {
            for(int i=0; i<arrayData.size(); i++) {
                if (arrayData.get(i).toLowerCase().contains(charText)) {
                    list.add(arrayData.get(i));
                }
            }
        }
        searchAdapter.notifyDataSetChanged();
    }

    public void initializeArrayData() {
        Cursor iCursor = mDbOpenHelper.sortColumn();
        arrayData.clear();
        while(iCursor.moveToNext()) {
            String tempName = iCursor.getString(iCursor.getColumnIndex("foodName"));
            arrayData.add(tempName);
        }
        list.clear();
        list.addAll(arrayData);
        searchAdapter.notifyDataSetChanged();
    }

    private AdapterView.OnItemClickListener onClickListener = new AdapterView.OnItemClickListener() {
        @Override
        public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
            Intent intent = new Intent(getApplicationContext(), ClickedRecipeActivity.class);
            intent.putExtra("foodName",list.get(position));
            Log.e("click_foodName",list.get(position));
            startActivityForResult(intent,REQUEST_CODE_ANOTHER);
        }
    };

    public void onBackBtn1Clicked(View v){

        finish();
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if(resultCode == RESULT_OK){
            if(requestCode==REQUEST_CODE_ANOTHER){
                String menu_name = intent.getExtras().getString("menu_name");
                Log.e("result_menu", menu_name);
                mDbOpenHelper.deleteColumn(menu_name);

                Log.e("delete ","완료");
                initializeArrayData();
                Log.e("showDB ","완료");
            }
        }
    }
}


