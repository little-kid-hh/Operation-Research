package org.example;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class Test0_4w {

    public static void processCSV(String filename, List<List<Double>> parcel_p, List<List<Double>> parcel_q, List<List<Double>> parcel_r, List<Double> id) {
        try (BufferedReader file = Files.newBufferedReader(Paths.get(filename))) {
            String line;
            Map<Double, List<Double>> length = new LinkedHashMap<>();
            Map<Double, List<Double>> wide = new LinkedHashMap<>();
            Map<Double, List<Double>> height = new LinkedHashMap<>();


            // 读取CSV文件的第一行（表头）
            file.readLine();  // 跳过表头行


            while ((line = file.readLine()) != null) {
                String[] tokens = line.split(",");
                List<Double> rowData = new ArrayList<>();
                double currentId = Double.parseDouble(tokens[0]);


                for (int i = 1; i < tokens.length; i++) {
                    rowData.add(Double.parseDouble(tokens[i]));
                }

                // 将数据存储到相应的map中
                length.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(1));  // SKU长度
                wide.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(2));  // SKU宽度
                height.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(3));  // SKU高度
            }

            // 将map中的数据存储到对应的vector中
            for (Map.Entry<Double, List<Double>> entry : length.entrySet()) {
                id.add(entry.getKey());  // 存储ID
                parcel_p.add(entry.getValue());  // 从map中获取值
            }

            for (Map.Entry<Double, List<Double>> entry : wide.entrySet()) {
                parcel_q.add(entry.getValue());  // 从map中获取值
            }

            for (Map.Entry<Double, List<Double>> entry : height.entrySet()) {
                parcel_r.add(entry.getValue());  // 从map中获取值
            }

        } catch (IOException e) {
            System.err.println("Failed to open the file!");
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // 初始化包裹数据
        List<List<Double>> myPackages = Arrays.asList(
                Arrays.asList(1.0, 45000.0, 60.0, 25.0, 30.0)
        );
        List<List<Double>> parcel_p = new ArrayList<>();
        List<List<Double>> parcel_q = new ArrayList<>();
        List<List<Double>> parcel_r = new ArrayList<>();
        List<Double> id = new ArrayList<>();

        String filename = "src/main/resources/dblf_converted_plus_tabu_converge.csv";  // CSV文件名
        processCSV(filename, parcel_p, parcel_q, parcel_r, id);
        System.out.println("处理完成");

        // 打开输出CSV文件
        try (BufferedWriter csvFile = Files.newBufferedWriter(Paths.get("milpLabeled/output0_4w_300.csv"))) {
            // 写入CSV文件头
            csvFile.write("ID,ReturnValue,Time");
            csvFile.newLine();

            for (int i = 0; i < 40000; i++) {

                Instant start_time = Instant.now(); // 开始计时

                int result = MILP_Loading_2orientations.MILP_single_box(id, parcel_p.get(i), parcel_q.get(i), parcel_r.get(i), myPackages,1);

                Instant end_time = Instant.now();  // 结束计时

                // 计算执行时间
                long duration = Duration.between(start_time, end_time).toMillis(); // 计算时间差

                // 将结果写入CSV文件
                csvFile.write(String.format("%d,%d,%d", id.get(i).intValue(), result, duration));
                csvFile.newLine();

                // 输出结果到控制台
                System.out.printf("ID: %f, Return Value: %d executed in %d milliseconds%n", id.get(i), result, duration);
            }
        } catch (IOException e) {
            System.err.println("Failed to open output CSV file!");
            e.printStackTrace();
        }
    }
}


