package com.example.sojeong.koreanfoodclassifier;

import android.content.Context;
import android.graphics.Color;
import android.text.Spannable;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.AbsoluteSizeSpan;
import android.text.style.ForegroundColorSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

public class ListviewAdapter extends BaseAdapter {

    private ArrayList<listviewItem_test> arrayList;

    public ListviewAdapter() {
        arrayList = new ArrayList<listviewItem_test>();
    }

    public View getView(int i, View view, ViewGroup viewGroup){
        final Context context = viewGroup.getContext();

        listviewItem_test listItem = arrayList.get(i);

        if(view == null){
            LayoutInflater inflater = (LayoutInflater)context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            view = inflater.inflate(R.layout.list_test, viewGroup, false);
        }

        ImageView img = (ImageView)view.findViewById(R.id.img_test);
        TextView txt = (TextView)view.findViewById(R.id.text_test);


        String str = String.valueOf(listItem.getFoodName());
        String acc = String.valueOf(listItem.getAccuracy());
        String msg = str+"\n"+acc+" certainty.";
        SpannableStringBuilder ssb1 = new SpannableStringBuilder(msg);
        int size = str.length();
        ssb1.setSpan(new AbsoluteSizeSpan(70),0,size,Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        ssb1.setSpan(new ForegroundColorSpan(Color.parseColor("#5F00FF")),0,size,Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        ssb1.setSpan(new ForegroundColorSpan(Color.parseColor("#5F00FF")),size+1,size+1+acc.length(),Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);


        txt.setText(ssb1);
        img.setImageResource(Integer.valueOf(listItem.getFoodImg()));



        return view;
    }

    public void addItem(String name, String accuracy, Integer img){
        listviewItem_test item = new listviewItem_test();

        item.setAccuracy(accuracy);
        item.setFoodName(name);
        item.setFoodImg(img);

        arrayList.add(item);
    }

    public void clearItem(){
        arrayList.clear();
    }

    public Object getItem(int i){
        return arrayList.get(i);
    }

    public long getItemId(int i){
        return i;
    }
    public int getCount(){
        return arrayList.size();
    }
}
