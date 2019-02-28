package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.EditText;
import android.widget.Filter;
import android.widget.Filterable;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

public class ScrappedActivity2 extends Activity {

    private ListView listView;
    private DbOpenHelper mDbOpenHelper;
    ArrayList<Listviewitem> data = new ArrayList<>();
    ListviewAdapter adapter;
    public static final int REQUEST_CODE_ANOTHER = 1002;

    private EditText editSearch;
    List<Listviewitem> list = new ArrayList<>();

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scrap_display);

        listView = (ListView) findViewById(R.id.recipe_list);

        //list = new ArrayList<Listviewitem>();
        editSearch = (EditText) findViewById(R.id.editSearch);


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
                if(text.length()>0){
                    listView.setFilterText(text);
                }
                else{
                    listView.clearTextFilter();
                }
            }
        });

    }

    public void initializeArrayData() {
        Cursor iCursor = mDbOpenHelper.sortColumn();
        data.clear();
        while(iCursor.moveToNext()) {
            int tempImage = R.drawable.full_star;
            String tempName = iCursor.getString(iCursor.getColumnIndex("foodName"));
            if (tempName == "떡볶이") {
                tempImage = R.drawable.tteokbokki;
            }

            Listviewitem temp = new Listviewitem(tempImage, tempName);
            data.add(temp);
        }

        adapter.notifyDataSetChanged();
    }

    /*
    public void search(String charText) {
        list.clear();

        if (charText.length() == 0) {
            list.addAll(data);
        }
        else {
            for(int i=0; i<data.size(); i++) {
                if (data.get(i).getFood_name().toLowerCase().contains(charText)) {
                    list.add(data.get(i));
                }
            }
        }
        adapter.notifyDataSetChanged();
    }*/

    public class ListviewAdapter extends BaseAdapter implements Filterable {

        private LayoutInflater inflater;
        private ArrayList<Listviewitem> data;
        private int layout;

        private ArrayList<Listviewitem> listViewItemList = new ArrayList<Listviewitem>();
        private ArrayList<Listviewitem> filteredItemList = listViewItemList;

        Filter listFilter;

        @Override
        public Filter getFilter() {
            if(listFilter == null){
                listFilter = new ListFilter();
            }

            return listFilter;
        }

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

        private class ListFilter extends Filter {

            @Override
            protected FilterResults performFiltering(CharSequence constraint) {
                FilterResults results = new FilterResults() ;

                if (constraint == null || constraint.length() == 0) {
                    results.values = listViewItemList ;
                    results.count = listViewItemList.size() ;
                } else {
                    ArrayList<Listviewitem> itemList = new ArrayList<Listviewitem>() ;

                    for (Listviewitem item : listViewItemList) {
                        if (item.getFood_name().toUpperCase().contains(constraint.toString().toUpperCase()))
                        {
                            itemList.add(item) ;
                        }
                    }

                    results.values = itemList ;
                    results.count = itemList.size() ;
                }
                return results;

        }
            @Override
            protected void publishResults(CharSequence constraint, FilterResults results) {

                // update listview by filtered data list.
                filteredItemList = (ArrayList<Listviewitem>) results.values ;

                // notify
                if (results.count > 0) {
                    notifyDataSetChanged() ;
                } else {
                    notifyDataSetInvalidated() ;
                }
            }
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
                initializeArrayData();
                Log.e("showDB ","완료");
            }
        }
    }



}
