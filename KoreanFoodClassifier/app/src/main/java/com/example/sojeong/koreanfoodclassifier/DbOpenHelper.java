package com.example.sojeong.koreanfoodclassifier;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDatabase.CursorFactory;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

public class DbOpenHelper {

    private static final String DATABASE_NAME = "InnerDatabase(SQLite).db";
    private static final int DATABASE_VERSION = 1;
    public static SQLiteDatabase mDB;
    private DatabaseHelper mDBHelper;
    private Context mCtx;

    private class DatabaseHelper extends SQLiteOpenHelper{

        public DatabaseHelper(Context context, String name, CursorFactory factory, int version) {
            super(context, name, factory, version);
        }

        @Override
        public void onCreate(SQLiteDatabase db){
            db.execSQL(DataBases.CreateDB._CREATE0);
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            db.execSQL("DROP TABLE IF EXISTS "+DataBases.CreateDB._TABLENAME0);
            onCreate(db);
        }
    }

    public DbOpenHelper(Context context){
        this.mCtx = context;
    }

    public DbOpenHelper open() throws SQLException{
        mDBHelper = new DatabaseHelper(mCtx, DATABASE_NAME, null, DATABASE_VERSION);
        mDB = mDBHelper.getWritableDatabase();
        return this;
    }

    public void create(){
        mDBHelper.onCreate(mDB);
    }

    public void close(){
        mDB.close();
    }

    // Insert DB (not include img)
    public long insertColumn(String foodId, String foodName, String foodIngredients, String foodPreparation , String foodCooking){
        ContentValues values = new ContentValues();
        values.put(DataBases.CreateDB.foodId, foodId);
        values.put(DataBases.CreateDB.foodName, foodName);
        values.put(DataBases.CreateDB.foodIngredients, foodIngredients);
        values.put(DataBases.CreateDB.foodPreparation, foodPreparation);
        values.put(DataBases.CreateDB.foodCooking, foodCooking);
        return mDB.insert(DataBases.CreateDB._TABLENAME0, null, values);
    }

    // Delete DB
    public boolean deleteColumn(String name){
        return mDB.delete(DataBases.CreateDB._TABLENAME0, " foodName = \'"+ name + "\'", null) > 0;
    }
    public void execSQL(String sql){
        mDB.execSQL(sql);
    }
    // Select DB
    public Cursor selectColumns(){
        return mDB.query(DataBases.CreateDB._TABLENAME0, null, null, null, null, null, null);
    }

    public Cursor sortColumn(){
        Cursor c = mDB.rawQuery( "SELECT * FROM foodTable ;" , null);
        return c;
    }
}