package com.example.sojeong.koreanfoodclassifier;

import android.content.Context;
import android.os.Environment;
import android.provider.ContactsContract;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.Buffer;
import java.nio.BufferOverflowException;
import java.util.HashMap;

public class TCP_client implements Runnable{
    private Socket socket = null;
    private DataOutputStream dataOutput = null;
    private DataInputStream dataInput = null;
    private int BUF_SIZE = 1024;
    private String serverIp;
    private int serverPort;
    private File img;

    private Thread thread = new Thread(this);

    static HashMap<String, String> recipe_ko, recipe_en;

    public TCP_client(String serverIp, int serverPort, File img) {
        super();
        this.serverIp = serverIp;
        this.serverPort = serverPort;
        this.img = img;
    }

    public void startTCP() {
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {}
    }

    public void run() {
        try {
            InetAddress serverAddr = InetAddress.getByName(serverIp);
            socket = new Socket(serverAddr, serverPort);
            try {
                dataOutput = new DataOutputStream(socket.getOutputStream());
                dataInput = new DataInputStream(new FileInputStream(img));
                byte[] buf = new byte[BUF_SIZE];
                byte[] header = new byte[32];
                byte[] lengthInfo = Integer.toString((int)img.length()).getBytes();

                // 이미지 전송
                for (int i = 0; i < lengthInfo.length; i++)
                    header[32 - lengthInfo.length + i] = lengthInfo[i];
                Log.e("len", Integer.toString((int)img.length()) + "/" + Integer.toString(lengthInfo.length));
                dataOutput.write(header, 0, 32);
                int dataLen;
                while ((dataLen = dataInput.read(buf)) > 0) {
                    dataOutput.write(buf, 0, dataLen);
                    dataOutput.flush();
                }

                // recipe 전송 받기
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream(), "CP949"));
                String recv;
                recipe_ko = new HashMap<String, String>();
                recipe_en = new HashMap<String, String>();
                String[] tokens = {"food id", "food name", "food ingredients", "food preparation", "food cooking", "food name", "food krName", "food ingredients", "food preparation", "food cooking"};
                for (int i = 0; i < 5; i++)
                    recipe_ko.put(tokens[i], "");
                for (int i = 5; i < tokens.length; i++)
                    recipe_en.put(tokens[i], "");
                int i = 0;
                while(!(recv = bufferedReader.readLine()).equals("recipe_en")) {
                    if (recv.equals(tokens[i])) {
                        Log.e(tokens[i], recipe_ko.get(tokens[i]));
                        i++;
                        continue;
                    }
                    String tmp = recipe_ko.get(tokens[i]);
                    recipe_ko.put(tokens[i], tmp + recv.trim() + "\r\n");
                }
                while((recv = bufferedReader.readLine()) != null) {
                    if (recv.equals(tokens[i])) {
                        Log.e(tokens[i] + "_en", recipe_en.get(tokens[i]));
                        i++;
                        continue;
                    }
                    String tmp = recipe_en.get(tokens[i]);
                    recipe_en.put(tokens[i], tmp + recv.trim() + "\r\n");
                }

                dataInput.close();
                dataOutput.close();
                socket.close();

            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                e.printStackTrace(new PrintWriter(sw));
                String exceptionAsString = sw.toString();
                Log.e("StackTrace", exceptionAsString);
            }
        } catch (IOException e) {
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String exceptionAsString = sw.toString();
            Log.e("StackTrace", exceptionAsString);
        }

    }
}
