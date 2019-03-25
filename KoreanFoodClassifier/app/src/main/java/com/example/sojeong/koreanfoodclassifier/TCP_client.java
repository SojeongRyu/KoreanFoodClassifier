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
    private static Socket socket = null;
    private DataOutputStream[] dataOutput = new DataOutputStream[2];
    private DataInputStream dataInput = null;
    private  BufferedReader bufferedReader = null;
    private int BUF_SIZE = 1024;
    private String serverIp;
    private int serverPort;
    private File img;
    private String response;

    static HashMap<String, String> recipe_ko1, recipe_ko2, recipe_en1, recipe_en2;

    public TCP_client(String serverIp, int serverPort, File img) {
        super();
        this.serverIp = serverIp;
        this.serverPort = serverPort;
        this.img = img;
    }

    public void startTCP(String response) {
        this.response = response;

        Thread thread = new Thread(this);
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {}
    }

    public void run() {
        try {
            InetAddress serverAddr = InetAddress.getByName(serverIp);
            if (socket == null)
                socket = new Socket(serverAddr, serverPort);
            try {
                if (response.length() == 0) {
                    sendImg();
                    recvRecipe();
                }
                else {
                    sendUserResponse(response);
                    finishConnection();
                }

            } catch (Exception e) {
                Log.e("sending receiving", e.toString());
            }
        } catch (IOException e) {
            Log.e("socket", e.toString());
        }

    }

    private void sendUserResponse(String response) {
        try {
            dataOutput[1] = new DataOutputStream(socket.getOutputStream());
            byte[] buf = response.getBytes();
            dataOutput[1].write(buf, 0, response.length());
        } catch (Exception e) {
            Log.e("sendUserResponse", e.toString());
        }
    }

    private void sendImg() {
        try {
            dataOutput[0] = new DataOutputStream(socket.getOutputStream());
            dataInput = new DataInputStream(new FileInputStream(img));
            byte[] buf = new byte[BUF_SIZE];
            byte[] header = new byte[32];
            byte[] lengthInfo = Integer.toString((int) img.length()).getBytes();

            // 이미지 전송
            for (int i = 0; i < lengthInfo.length; i++)
                header[32 - lengthInfo.length + i] = lengthInfo[i];
            Log.e("len", Integer.toString((int) img.length()) + "/" + Integer.toString(lengthInfo.length));
            dataOutput[0].write(header, 0, 32);
            int dataLen;
            while ((dataLen = dataInput.read(buf)) > 0) {
                dataOutput[0].write(buf, 0, dataLen);
                dataOutput[0].flush();
            }
        } catch (Exception e) {
            Log.e("sendImg", e.toString());
        }

    }

    private void recvRecipe() {
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream(), "CP949"));
            String recv;
            recipe_ko1 = new HashMap<String, String>();
            recipe_ko2 = new HashMap<String, String>();
            recipe_en1 = new HashMap<String, String>();
            recipe_en2 = new HashMap<String, String>();
            String[] tokens = {"food id", "predict percentage", "food name", "food ingredients", "food preparation", "food cooking", "food name", "food krName", "food ingredients", "food preparation", "food cooking"};
            for (int i = 0; i < 6; i++) {
                recipe_ko1.put(tokens[i], "");
                recipe_ko2.put(tokens[i], "");
            }
            for (int i = 6; i < tokens.length; i++) {
                recipe_en1.put(tokens[i], "");
                recipe_en2.put(tokens[i], "");
            }
            int i = 0;
            while(!(recv = bufferedReader.readLine().trim()).equals("recipe_en")) {
                if (recv.equals(tokens[i])) {
                    i++;
                    continue;
                }
                String tmp = recipe_ko1.get(tokens[i]);
                recipe_ko1.put(tokens[i], tmp + recv + "\r\n");
            }
            while(!(recv = bufferedReader.readLine().trim()).equals("recipe_done")) {
                if (recv.equals(tokens[i])) {
                    i++;
                    continue;
                }
                String tmp = recipe_en1.get(tokens[i]);
                recipe_en1.put(tokens[i], tmp + recv + "\r\n");
            }

            i = 0;
            while(!(recv = bufferedReader.readLine().trim()).equals("recipe_en")) {
                if (recv.equals(tokens[i])) {
                    i++;
                    continue;
                }
                String tmp = recipe_ko2.get(tokens[i]);
                recipe_ko2.put(tokens[i], tmp + recv + "\r\n");
            }
            while(!(recv = bufferedReader.readLine().trim()).equals("recipe_done")) {
                if (recv.equals(tokens[i])) {
                    i++;
                    continue;
                }
                String tmp = recipe_en2.get(tokens[i]);
                recipe_en2.put(tokens[i], tmp + recv + "\r\n");
            }
        } catch (Exception e) {
            Log.e("recvImg", e.toString());
        }
    }

    private void finishConnection() {
        try {
            if (dataInput != null)
                dataInput.close();
            for (int i = 0; i < 2; i++) {
                if (dataOutput[i] != null)
                    dataOutput[i].close();
            }
            if (bufferedReader != null)
                bufferedReader.close();
            if (socket != null) {
                socket.close();
                socket = null;
            }
        } catch (Exception e) {

        }
    }
}
