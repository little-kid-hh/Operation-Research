package org.example;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class collect_fragility_data {

//    public static void processCSV(String filename, List<List<Double>> parcel_p, List<List<Double>> parcel_q, List<List<Double>> parcel_r,List<List<Integer>> parcel_fragility, List<Double> id) {
//        try (BufferedReader file = Files.newBufferedReader(Paths.get(filename), Charset.forName("UTF-8"))) {
//            String line;
//            Map<Double, List<Double>> length = new LinkedHashMap<>();
//            Map<Double, List<Double>> wide = new LinkedHashMap<>();
//            Map<Double, List<Double>> height = new LinkedHashMap<>();
//            Map<Double, List<Integer>> fragility = new LinkedHashMap<>();
//
//            // 读取CSV文件的第一行（表头）
//            file.readLine();  // 跳过表头行
//
//
//            while ((line = file.readLine()) != null) {
//                String[] tokens = line.split(",");
//                List<Double> rowData = new ArrayList<>();
//                double currentId = Double.parseDouble(tokens[0]);
//
//
//                for (int i = 1; i < tokens.length; i++) {
//                    rowData.add(Double.parseDouble(tokens[i]));
//                }
//
//                // 将数据存储到相应的map中
//                length.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(1));  // SKU长度
//                wide.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(2));  // SKU宽度
//                height.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(3));  // SKU高度
////                System.out.println("ID: " + currentId + " Length: " + rowData.get(1) + " Wide: " + rowData.get(2) + " Height: " + rowData.get(3)+ " Fragility: " + rowData.get(4));
//                fragility.computeIfAbsent(currentId, k -> new ArrayList<>()).add(rowData.get(4).intValue());  // 易碎性
//            }
//
//            // 将map中的数据存储到对应的vector中
//            for (Map.Entry<Double, List<Double>> entry : length.entrySet()) {
//                id.add(entry.getKey());  // 存储ID
//                parcel_p.add(entry.getValue());  // 从map中获取值
//            }
//
//            for (Map.Entry<Double, List<Double>> entry : wide.entrySet()) {
//                parcel_q.add(entry.getValue());  // 从map中获取值
//            }
//
//            for (Map.Entry<Double, List<Double>> entry : height.entrySet()) {
//                parcel_r.add(entry.getValue());  // 从map中获取值
//            }
//
//            for (Map.Entry<Double, List<Integer>> entry : fragility.entrySet()) {
//                parcel_fragility.add(entry.getValue());  // 从map中获取值
//            }
//
//        } catch (IOException e) {
//            System.err.println("Failed to open the file!");
//            e.printStackTrace();
//        }
//    }

    public static void processCSV(String filename,
                                  List<List<Double>> parcel_p,
                                  List<List<Double>> parcel_q,
                                  List<List<Double>> parcel_r,
                                  List<List<Integer>> parcel_fragility,
                                  List<List<Integer>> parcel_sequence,
                                  List<Double> id) {

        try (BufferedReader file = Files.newBufferedReader(Paths.get(filename), Charset.forName("GBK"))) {
            String line;
            Map<Double, List<List<Double>>> length = new LinkedHashMap<>();
            Map<Double, List<List<Double>>> wide = new LinkedHashMap<>();
            Map<Double, List<List<Double>>> height = new LinkedHashMap<>();
            Map<Double, List<List<Integer>>> fragility = new LinkedHashMap<>();
            Map<Double, List<List<Integer>>> sequence = new LinkedHashMap<>();

            // 跳过表头
            file.readLine();

            while ((line = file.readLine()) != null) {
                String[] tokens = line.split(",");
                double currentId = Double.parseDouble(tokens[0]);

                double l = Double.parseDouble(tokens[2]);
                double w = Double.parseDouble(tokens[3]);
                double h = Double.parseDouble(tokens[4]);
                int f = (int) Double.parseDouble(tokens[5]);
                int s = (int) Double.parseDouble(tokens[6]);

                length.computeIfAbsent(currentId, k -> new ArrayList<>()).add(Collections.singletonList(l));
                wide.computeIfAbsent(currentId, k -> new ArrayList<>()).add(Collections.singletonList(w));
                height.computeIfAbsent(currentId, k -> new ArrayList<>()).add(Collections.singletonList(h));
                fragility.computeIfAbsent(currentId, k -> new ArrayList<>()).add(Collections.singletonList(f));
                sequence.computeIfAbsent(currentId, k -> new ArrayList<>()).add(Collections.singletonList(s));
            }

            Set<String> seenGroups = new HashSet<>();

            // 用于记录 fragility 分析数据
            List<String[]> outputRows = new ArrayList<>();
            outputRows.add(new String[]{"id", "fragile_ratio", "fragile_volume"}); // CSV 表头

            for (Double groupId : length.keySet()) {
                List<List<Double>> lList = length.get(groupId);
                List<List<Double>> wList = wide.get(groupId);
                List<List<Double>> hList = height.get(groupId);
                List<List<Integer>> fList = fragility.get(groupId);
                List<List<Integer>> sList = sequence.get(groupId);

                List<String> itemDescriptions = new ArrayList<>();
                for (int i = 0; i < lList.size(); i++) {
                    String desc = lList.get(i).get(0) + "," + wList.get(i).get(0) + "," + hList.get(i).get(0) + "," + fList.get(i).get(0) ;
                    itemDescriptions.add(desc);
                }

                Collections.sort(itemDescriptions);
                String groupKey = String.join(";", itemDescriptions);

                if (!seenGroups.contains(groupKey)) {
                    seenGroups.add(groupKey);

                    List<Double> p = new ArrayList<>();
                    List<Double> q = new ArrayList<>();
                    List<Double> r = new ArrayList<>();
                    List<Integer> fr = new ArrayList<>();
                    List<Integer> s = new ArrayList<>();

                    int totalCount = itemDescriptions.size();
                    int fragileCount = 0;
                    double fragileVolume = 0.0;

                    for (int i = 0; i < itemDescriptions.size(); i++) {
                        double li = lList.get(i).get(0);
                        double wi = wList.get(i).get(0);
                        double hi = hList.get(i).get(0);
                        int fi = fList.get(i).get(0);
                        int si = sList.get(i).get(0);

                        p.add(li);
                        q.add(wi);
                        r.add(hi);
                        fr.add(fi);
                        s.add(si);

                        if (fi == 1) {
                            fragileCount++;
                            fragileVolume += li * wi * hi;
                        }
                    }

                    id.add(groupId);
                    parcel_p.add(p);
                    parcel_q.add(q);
                    parcel_r.add(r);
                    parcel_fragility.add(fr);
                    parcel_sequence.add(s);

                    double fragileRatio = totalCount > 0 ? ((double) fragileCount / totalCount) : 0.0;

                    // 添加到输出行（保留 6 位小数）
                    outputRows.add(new String[]{
                            String.valueOf(groupId),
                            String.format("%.6f", fragileRatio),
                            String.format("%.2f", fragileVolume)
                    });
                }
            }

            // 输出 CSV 文件
            try (BufferedWriter writer = Files.newBufferedWriter(Paths.get("fragile_stats.csv"), Charset.forName("UTF-8"))) {
                for (String[] row : outputRows) {
                    writer.write(String.join(",", row));
                    writer.newLine();
                }
            }

            System.out.println("分组数（唯一组）： " + seenGroups.size());
            System.out.println("fragile_stats.csv 已输出。");

        } catch (IOException e) {
            System.err.println("Failed to open the file!");
            e.printStackTrace();
        }
    }
    // 辅助方法：检查 SKU 数据是否有效
    private static boolean isValidSKUData(String length, String width, String height, String fragility,String sequence) {
        try {
            Double.parseDouble(length);
            Double.parseDouble(width);
            Double.parseDouble(height);
            Integer.parseInt(fragility);
            Integer.parseInt(sequence);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public static void main(String[] args) {
        // 初始化包裹数据
        List<List<Double>> myPackages = Arrays.asList(
                Arrays.asList(1.0, 45000.0, 60.0, 25.0,30.0 )
        );
        List<List<Double>> parcel_p = new ArrayList<>();
        List<List<Double>> parcel_q = new ArrayList<>();
        List<List<Double>> parcel_r = new ArrayList<>();
        List<List<Integer>> parcel_fragility = new ArrayList<>();
        List<List<Integer>> parcel_sequence = new ArrayList<>();
        List<Double> id = new ArrayList<>();
        String csvF = "milpLabeled/output0_20w_label.csv"; // 替换为你的CSV文件路径
        ArrayList<Integer> resultList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvF))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(","); // 假设CSV使用逗号分隔
                if (values.length >= 2) { // 确保至少有两列
                    try {
                        int firstColumn = Integer.parseInt(values[0].trim());
                        int secondColumn = Integer.parseInt(values[1].trim());

//                        if (secondColumn == -1) {
//                            resultList.add(firstColumn);
//                        }
                        resultList.add(firstColumn);
                    } catch (NumberFormatException e) {
                        // 跳过无法解析为数字的行
                        continue;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

//        // 输出结果
//        System.out.println("收集到的数据: " + resultList);



        String filename = "src/main/resources/dblf_converted_plus_tabu_converge_sequence_fragility.csv";  // CSV文件名
        processCSV(filename, parcel_p, parcel_q, parcel_r, parcel_fragility,parcel_sequence,id);
        System.out.println("处理完成");
//
//        // 打开输出CSV文件
//        try (BufferedWriter csvFile = Files.newBufferedWriter(Paths.get("milpLabeled/output0_20w_3_10s_sequence_fragility.csv"))) {
//            // 写入CSV文件头
//            csvFile.write("ID,ReturnValue,Time");
//            csvFile.newLine();
//
//            for (int i = 1; i < 200; i++) {
////                if(id.get(i).intValue()<=889717){
////                    continue;
////                }
//
//                Instant start_time = Instant.now(); // 开始计时
//                int result = MILP_Loading_2orientations_sequence_fragility.MILP_single_box(id, parcel_p.get(i), parcel_q.get(i), parcel_r.get(i),parcel_fragility.get(i), parcel_sequence.get(i),myPackages,1);
////                int result =1;
//                Instant end_time = Instant.now();  // 结束计时
//
//                // 计算执行时间
//                long duration = Duration.between(start_time, end_time).toMillis(); // 计算时间差
//
//                // 将结果写入CSV文件
//                csvFile.write(String.format("%d,%d,%d,%d", id.get(i).intValue(), 0, duration,result));
//                csvFile.newLine();
//                csvFile.flush();
//
//                // 输出结果到控制台
//                System.out.printf("ID: %f, Return Value: %d executed in %d milliseconds%n", id.get(i), result, duration);
//
//            }
//        } catch (IOException e) {
//            System.err.println("Failed to open output CSV file!");
//            e.printStackTrace();
//        }
    }
}


