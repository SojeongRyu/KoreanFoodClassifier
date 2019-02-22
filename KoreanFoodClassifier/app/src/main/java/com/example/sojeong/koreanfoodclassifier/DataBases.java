package com.example.sojeong.koreanfoodclassifier;

import android.provider.BaseColumns;

public final class DataBases {

    public static final class CreateDB implements  BaseColumns {
        public static final String foodName  = "foodName";
        public static final String foodIngredients = "foodIngredients";
        public static final String foodPreparation  = "foodPreparation";
        public static final String foodCooking  = "foodCooking";
        public static final String foodImg = "foodImg";

        public static final String _TABLENAME0 = "foodTable";
        public static final String _CREATE0 = "create table if not exists " + _TABLENAME0
                + "(" + _ID + " integer primary key autoincrement, "
                + foodName + " text not null , "
                + foodIngredients + " text not null , "
                + foodPreparation + " text not null , "
                + foodCooking + " text not null , "
                + foodImg + " blob );" ;
    }
}