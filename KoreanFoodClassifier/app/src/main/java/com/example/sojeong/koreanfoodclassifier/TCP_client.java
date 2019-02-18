package com.example.sojeong.koreanfoodclassifier;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.InetAddress;
import java.net.Socket;

public class TCP_client implements Runnable{
    private Socket socket = null;
    private DataOutputStream dataOutput = null;
    private DataInputStream dataInput = null;
    private int BUF_SIZE = 1024;
    private String serverIp;
    private int serverPort;
    private File img;

    private Thread thread = new Thread(this);

    private String response;

    public TCP_client(String serverIp, int serverPort, File img) {
        super();
        this.serverIp = serverIp;
        this.serverPort = serverPort;
        this.img = img;
    }

    public String startTCP() {
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {}
        return response;
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
                for (int i = 0; i < lengthInfo.length; i++)
                    header[32 - lengthInfo.length + i] = lengthInfo[i];
                Log.e("len", Integer.toString((int)img.length()) + "/" + Integer.toString(lengthInfo.length));
                dataOutput.write(header, 0, 32);
                int dataLen;
                while ((dataLen = dataInput.read(buf)) > 0) {
                    dataOutput.write(buf, 0, dataLen);
                    dataOutput.flush();
                }

                BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                response = reader.readLine();

                reader.close();
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
