package com.example.sojeong.koreanfoodclassifier;

import android.media.Image;

public class Listviewitem {
    private int food_image;
    private String food_name;

    public int getFood_image(){return food_image;}

    //public int setFood_image(int food_image){this.food_image = food_image;}

    public String getFood_name(){return food_name;}

    //public String setFood_name(String food_name){this.food_name = food_name;}


    public Listviewitem(int food_image, String food_name){
        this.food_image=food_image;
        this.food_name=food_name;
    }
}
