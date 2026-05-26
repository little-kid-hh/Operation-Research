package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class dataprocess {

    public static Map<Integer, Integer> processCSV1(String filename) {
        Map<Integer,Integer> IDs = new HashMap<>();
        try (BufferedReader file = Files.newBufferedReader(Paths.get(filename))) {
            String line;



            // 读取CSV文件的第一行（表头）
            file.readLine();  // 跳过表头行


            while ((line = file.readLine()) != null) {
                // 分割CSV行（假设使用逗号分隔）
                String[] columns = line.split(",");

                // 确保行有足够多的列
                if (columns.length >= 4) {
                    try {
                        // 第一列作为key（转换为整数）
                        int key = Integer.parseInt(columns[0].trim());
                        // 第四列作为value（转换为整数）
                        int value = Integer.parseInt(columns[3].trim());
                        // 存入HashMap
                        IDs.put(key, value);
                    } catch (NumberFormatException e) {
                        // 处理数字格式异常
                        System.err.println("无法解析数字: " + line);
                    }
                }
            }

            return IDs;

        } catch (IOException e) {
            System.err.println("Failed to open the file!");
            e.printStackTrace();
        }
        return IDs;
    }

    public static void processCSV2(String filename1, String filename2) {
        Map<Integer, Integer> IDs1 = processCSV1(filename1);
        try (BufferedReader file2 = new BufferedReader(new InputStreamReader(
                Files.newInputStream(Paths.get(filename2)), "GBK"))) {
            List<String> outputLines = new ArrayList<>();

            // 读取并保留表头
            String header = file2.readLine();
            if (header != null) {
                outputLines.add(header);  // 将表头添加到输出列表
            }

            String line;
            while ((line = file2.readLine()) != null) {
                String[] columns = line.split(",");

                if (columns.length >= 3) {
                    try {
                        int secondCol = Integer.parseInt(columns[1].trim());

                        // 检查第2列是否在Map中
                        if (IDs1.containsKey(secondCol)) {
                            // 根据Map中的value设置第3列的值
                            int newValue = (IDs1.get(secondCol) == 1) ? 1 : 0;
                            columns[2] = String.valueOf(newValue);

                            // 重新组装行
                            line = String.join(",", columns);
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("无法解析数字: " + line);
                    }
                }
                outputLines.add(line);
            }

            // 使用GBK编码写入文件，包含表头
            Files.write(Paths.get("output22222.csv"), outputLines, Charset.forName("GBK"));

        } catch (IOException e) {
            System.err.println("Failed to open the second file!");
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        processCSV2("milpLabeled/output0_20w_3_10s_3.csv", "milpLabeled/training_feature_engineered&ZXY_dblf_plus_tabu_selected_200000_newSampled.csv");
    }



}
