package org.example;

import java.io.*;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class GeneratePerminLabels {
    public static void main(String[] args) {
        // Adjust these paths if running from a different working directory
        String inputCsv = "../../permin_dataset_processing/milp_labels/items_to_label.csv";
        String outputCsv = "../../permin_dataset_processing/milp_labels/ground_truth_labels.csv";

        // Standard Vehicle/Bin limits: Length, Capacity(Vol), L, W, H
        List<List<Double>> myPackages = Arrays.asList(
                Arrays.asList(1.0, 45000.0, 60.0, 25.0, 30.0) 
        );

        try (BufferedReader br = Files.newBufferedReader(Paths.get(inputCsv));
             BufferedWriter bw = Files.newBufferedWriter(Paths.get(outputCsv))) {
            
            String line = br.readLine(); // skip header
            bw.write("instance_name,order_id,label_2ori,time_2ori_ms,label_6ori,time_6ori_ms");
            bw.newLine();

            String currentInstance = "";
            String currentOrder = "";
            List<Double> pList = new ArrayList<>();
            List<Double> qList = new ArrayList<>();
            List<Double> rList = new ArrayList<>();
            List<Double> idList = new ArrayList<>();
            double itemId = 0;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 5) continue;

                String inst = parts[0];
                String ord = parts[1];
                double p = Double.parseDouble(parts[2]);
                double q = Double.parseDouble(parts[3]);
                double r = Double.parseDouble(parts[4]);

                if (!inst.equals(currentInstance) || !ord.equals(currentOrder)) {
                    if (!pList.isEmpty()) {
                        processOrder(currentInstance, currentOrder, pList, qList, rList, idList, myPackages, bw);
                    }
                    currentInstance = inst;
                    currentOrder = ord;
                    pList.clear(); qList.clear(); rList.clear(); idList.clear();
                    itemId = 0;
                }
                pList.add(p);
                qList.add(q);
                rList.add(r);
                idList.add(itemId++);
            }
            // Process the very last order
            if (!pList.isEmpty()) {
                processOrder(currentInstance, currentOrder, pList, qList, rList, idList, myPackages, bw);
            }

            System.out.println("Finished generating labels!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processOrder(String inst, String ord, List<Double> pList, List<Double> qList, List<Double> rList, List<Double> idList, List<List<Double>> myPackages, BufferedWriter bw) throws IOException {
        System.out.println("Processing " + inst + " Order " + ord + " | Items: " + pList.size());
        
        // 1. Solve with 2-orientations
        Instant start2 = Instant.now();
        int res2 = 0;
        try {
            res2 = MILP_Loading_2orientations.MILP_single_box(idList, pList, qList, rList, myPackages, 1);
        } catch (Exception | Error e) {
            System.err.println("Gurobi Error 2ori: " + e.getMessage());
            res2 = -1; // -1 indicates error/infeasible/timeout
        }
        long time2 = Duration.between(start2, Instant.now()).toMillis();

        // 2. Solve with 6-orientations
        Instant start6 = Instant.now();
        int res6 = 0;
        try {
            res6 = MILP_Loading_6orientations.MILP_single_box(idList, pList, qList, rList, myPackages, 1);
        } catch (Exception | Error e) {
            System.err.println("Gurobi Error 6ori: " + e.getMessage());
            res6 = -1; 
        }
        long time6 = Duration.between(start6, Instant.now()).toMillis();

        // Write results
        bw.write(inst + "," + ord + "," + res2 + "," + time2 + "," + res6 + "," + time6);
        bw.newLine();
        bw.flush();
    }
}
