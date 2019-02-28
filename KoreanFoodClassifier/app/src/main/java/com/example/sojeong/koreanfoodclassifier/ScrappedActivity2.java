package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;

import java.util.ArrayList;

public class ScrappedActivity2 extends Activity {

    private ListView listView;
    private DbOpenHelper mDbOpenHelper;
    ArrayList<Listviewitem> data = new ArrayList<>();
    ListviewAdapter adapter;
    public static final int REQUEST_CODE_ANOTHER = 1002;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        listView = (ListView) findViewById(R.id.recipe_list);

        Listviewitem food1 = new Listviewitem(R.drawable.full_star,"떡볶이");
        Listviewitem food2 = new Listviewitem(R.drawable.empty_star,"food2");

        data.add(food1);
        data.add(food2);

        adapter = new ListviewAdapter(this, R.layout.scrapped_item, data);
        listView.setAdapter(adapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Log.e("onclick","클릭!!!!!");

                Listviewitem item = (Listviewitem) parent.getItemAtPosition(position);

                String food_name = item.getFood_name();
                Integer food_image = item.getFood_image();

                Intent intent = new Intent(getApplicationContext(), ClickedRecipeActivity.class);
                intent.putExtra("foodName", food_name);
                intent.putExtra("food_image", food_image);
                startActivityForResult(intent,REQUEST_CODE_ANOTHER);
            }
        });

        mDbOpenHelper = new DbOpenHelper(this);
        mDbOpenHelper.open();
        mDbOpenHelper.create();
        showDatabase();

    }

    public void showDatabase() {
        Cursor iCursor = mDbOpenHelper.sortColumn();
        data.clear();
        while(iCursor.moveToNext()) {
            int tempImage = R.drawable.full_star;
            String tempName = iCursor.getString(iCursor.getColumnIndex("foodName"));
            if(tempName=="떡볶이") {
                tempImage = R.drawable.tteokbokki;
            }

            Listviewitem temp = new Listviewitem(tempImage, tempName);
            data.add(temp);
        }
        adapter.clear();
        adapter.addAllItem(data);
       // adapter.addAll(data);
        adapter.notifyDataSetChanged();
    }

    public class ListviewAdapter extends BaseAdapter{

        private LayoutInflater inflater;
        private ArrayList<Listviewitem> data;
        private int layout;

        public ListviewAdapter(Context context, int layout, ArrayList<Listviewitem> data){
            this.inflater = (LayoutInflater)context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            this.data = data;
            this.layout=layout;
        }

        public int getCount(){return data.size();}

        public String getItem(int position){return data.get(position).getFood_name();}

        public long getItemId(int position){return position;}

        public View getView(int position, View convertView, ViewGroup parent){
            if(convertView==null){
                convertView = inflater.inflate(layout,parent,false);
            }

            Listviewitem listviewitem = data.get(position);
            ImageView food_image = (ImageView)convertView.findViewById(R.id.imageview);

            food_image.setImageResource(listviewitem.getFood_image());

            TextView name = (TextView)convertView.findViewById(R.id.textview);
            name.setText(listviewitem.getFood_name());

            return convertView;
        }

        /////////////////////////////
        public void addAllItem(ArrayList data){
            ArrayList<Listviewitem> item = data;
            data.add(item);
        }

        public void clear(){
            data.clear();
        }


    }
    public void onBackBtn1Clicked(View v){
        Intent resultIntent = new Intent();

        setResult(RESULT_OK, resultIntent);
        finish();
    }
    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if(resultCode == RESULT_OK){
            if(requestCode==REQUEST_CODE_ANOTHER){
                String menu_name = intent.getExtras().getString("menu_name");
                Log.e("result_menu", menu_name);
                mDbOpenHelper.deleteColumn(menu_name);

                //mDbOpenHelper.execSQL("delete from foodTable where foodName = \'" + menu_name + "\';");

                Log.e("delete ","완료");
                showDatabase();
                Log.e("showDB ","완료");
            }
        }
    }



}
