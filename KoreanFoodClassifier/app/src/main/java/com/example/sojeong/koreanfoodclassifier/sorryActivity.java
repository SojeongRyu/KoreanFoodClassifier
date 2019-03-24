package com.example.sojeong.koreanfoodclassifier;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;

public class sorryActivity extends Activity {
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.sorry_answer);

        sorry_dialog();
    }

    void sorry_dialog(){
        AlertDialog.Builder sorry_dialog = new AlertDialog.Builder(this);
        sorry_dialog.setMessage("I can not find you any other recipes.\nSorry for the inconvenience\n");
        sorry_dialog.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
                Intent intent2 = new Intent(getApplicationContext(), MainActivity.class);
                intent2.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent2);
            }
        });
        sorry_dialog.show();
    }
}
