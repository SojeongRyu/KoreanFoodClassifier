package com.example.sojeong.koreanfoodclassifier;

public class listviewItem_test {

    private String foodName;
    private int foodImg;
    private String accuracy;

    public String getFoodName(){
        return foodName;
    }

    public void setFoodName(String foodName){
        this.foodName = foodName;
    }

    public int getFoodImg(){
        return foodImg;
    }

    public void setFoodImg(int foodImg) {
        this.foodImg = foodImg;
    }

    public void setAccuracy(String accuracy){this.accuracy = accuracy;}

    public String getAccuracy(){ return accuracy; }
}
