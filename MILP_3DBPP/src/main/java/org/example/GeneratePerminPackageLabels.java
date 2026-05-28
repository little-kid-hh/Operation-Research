package org.example;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.BufferedWriter;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class GeneratePerminPackageLabels {
    private static final Pattern BASE_PERFORMANCE_XML = Pattern.compile("^BSP_\\d+_O6_\\d+\\.xml$");

    private static class PackageType {
        final int id;
        final double length;
        final double width;
        final double height;

        PackageType(int id, double length, double width, double height) {
            this.id = id;
            this.length = length;
            this.width = width;
            this.height = height;
        }

        List<Double> toMilpPackage() {
            List<Double> row = new ArrayList<>();
            row.add((double) id);
            row.add(length * width * height);
            row.add(length);
            row.add(width);
            row.add(height);
            return row;
        }
    }

    private static class OrderData {
        final String orderId;
        final List<Double> ids = new ArrayList<>();
        final List<Double> p = new ArrayList<>();
        final List<Double> q = new ArrayList<>();
        final List<Double> r = new ArrayList<>();

        OrderData(String orderId) {
            this.orderId = orderId;
        }
    }

    public static void main(String[] args) throws Exception {
        Path xmlDir = args.length > 0 ? Path.of(args[0]) : Path.of("S3DBSP-main/performanceTest");
        Path packagesPath = args.length > 1 ? Path.of(args[1]) : Path.of("S3DBSP-main/performanceTest/packages.txt");
        Path outputPath = args.length > 2 ? Path.of(args[2]) : Path.of("permin_dataset_processing/milp_labels/ground_truth_package_labels.csv");
        int maxXml = args.length > 3 ? Integer.parseInt(args[3]) : -1;
        int maxOrdersPerXml = args.length > 4 ? Integer.parseInt(args[4]) : -1;
        int maxPackages = args.length > 5 ? Integer.parseInt(args[5]) : -1;
        boolean includeVariants = args.length > 6 && Boolean.parseBoolean(args[6]);
        long startGlobalTask = args.length > 7 ? Long.parseLong(args[7]) : 0L;
        long maxTasks = args.length > 8 ? Long.parseLong(args[8]) : -1L;

        List<Path> xmlFiles = selectXmlFiles(xmlDir, includeVariants);
        List<PackageType> packages = readPackages(packagesPath);
        if (maxXml > 0 && maxXml < xmlFiles.size()) {
            xmlFiles = xmlFiles.subList(0, maxXml);
        }
        if (maxPackages > 0 && maxPackages < packages.size()) {
            packages = packages.subList(0, maxPackages);
        }

        Files.createDirectories(outputPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writer.write("instance_name,order_id,package_id,package_l,package_w,package_h,label_2ori,time_2ori_ms,label_6ori,time_6ori_ms");
            writer.newLine();

            long globalTaskIndex = 0L;
            long writtenTasks = 0L;
            outer:
            for (Path xmlPath : xmlFiles) {
                List<OrderData> orders = readOrders(xmlPath);
                if (maxOrdersPerXml > 0 && maxOrdersPerXml < orders.size()) {
                    orders = orders.subList(0, maxOrdersPerXml);
                }
                for (OrderData order : orders) {
                    for (PackageType packageType : packages) {
                        if (globalTaskIndex++ < startGlobalTask) {
                            continue;
                        }
                        if (maxTasks >= 0 && writtenTasks >= maxTasks) {
                            break outer;
                        }
                        List<List<Double>> onePackage = new ArrayList<>();
                        onePackage.add(packageType.toMilpPackage());

                        Instant start2 = Instant.now();
                        int label2 = MILP_Loading_2orientations.MILP_single_box(
                                order.ids, order.p, order.q, order.r, onePackage, 1);
                        long time2 = Duration.between(start2, Instant.now()).toMillis();

                        Instant start6 = Instant.now();
                        int label6 = MILP_Loading_6orientations.MILP_single_box(
                                order.ids, order.p, order.q, order.r, onePackage, 1);
                        long time6 = Duration.between(start6, Instant.now()).toMillis();

                        writer.write(String.format(
                                "%s,%s,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d",
                                xmlPath.getFileName(),
                                order.orderId,
                                packageType.id,
                                packageType.length,
                                packageType.width,
                                packageType.height,
                                label2,
                                time2,
                                label6,
                                time6));
                        writer.newLine();
                        writer.flush();
                        writtenTasks++;
                    }
                }
            }
        }
    }

    private static List<Path> selectXmlFiles(Path xmlDir, boolean includeVariants) throws Exception {
        try (var stream = Files.list(xmlDir)) {
            return stream
                    .filter(path -> path.getFileName().toString().endsWith(".xml"))
                    .filter(path -> includeVariants || BASE_PERFORMANCE_XML.matcher(path.getFileName().toString()).matches())
                    .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                    .collect(Collectors.toList());
        }
    }

    private static List<PackageType> readPackages(Path packagesPath) throws Exception {
        List<PackageType> packages = new ArrayList<>();
        for (String line : Files.readAllLines(packagesPath)) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            String[] parts = trimmed.split("\\s+");
            packages.add(new PackageType(
                    Integer.parseInt(parts[0]),
                    Double.parseDouble(parts[1]),
                    Double.parseDouble(parts[2]),
                    Double.parseDouble(parts[3])));
        }
        return packages;
    }

    private static List<OrderData> readOrders(Path xmlPath) throws Exception {
        Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(new File(xmlPath.toString()));
        doc.getDocumentElement().normalize();
        NodeList orderNodes = doc.getElementsByTagName("order");
        List<OrderData> orders = new ArrayList<>();
        for (int i = 0; i < orderNodes.getLength(); i++) {
            Element orderElement = (Element) orderNodes.item(i);
            OrderData order = new OrderData(orderElement.getAttribute("id"));
            NodeList itemNodes = orderElement.getElementsByTagName("item");
            for (int j = 0; j < itemNodes.getLength(); j++) {
                Element itemElement = (Element) itemNodes.item(j);
                order.ids.add((double) j);
                order.p.add(Double.parseDouble(itemElement.getElementsByTagName("p").item(0).getTextContent()));
                order.q.add(Double.parseDouble(itemElement.getElementsByTagName("q").item(0).getTextContent()));
                order.r.add(Double.parseDouble(itemElement.getElementsByTagName("r").item(0).getTextContent()));
            }
            orders.add(order);
        }
        return orders;
    }
}
