package org.example;

import org.example.RouteElement.Item;
import org.example.RouteElement.Node;
import org.example.RouteElement.Parameters;

import java.util.*;

public class DBLF_TABU {
    Random random = new Random(41);


//    public ArrayList<int[]> positionCandidates = new ArrayList<int[]>();
    private static final boolean lifoSwitch = false;
    private static final boolean fragileSwitch = false;
    private static final boolean supportAreaSwitch = false;
    private static final int if_record_to_csv_switch = 0;
    private static final int orientationsLimit = 2;


    //    public double calculateLoadingLength(List<Node> visitSequence){return 0.0;}

    public ArrayList<Item> randomSwapItemSequence(ArrayList<Item> unplacedItems){
        if(unplacedItems.size() <= 1){
            return unplacedItems;
        }

        HashMap<Integer, Integer> visitOrderMap = new HashMap<Integer, Integer>(); // 记录每个访问顺序下有多少客户
        if(lifoSwitch){
            for(int i = 0; i < unplacedItems.size(); i++){
                int visitOrder = unplacedItems.get(i).visit_order;
                visitOrderMap.put(visitOrder, visitOrderMap.getOrDefault(visitOrder, 0) + 1);
            }
        }
        else{
            for(int i = 0; i < unplacedItems.size(); i++){
                int visitOrder = unplacedItems.get(i).visit_order;
                visitOrderMap.put(visitOrder, visitOrderMap.getOrDefault(visitOrder, 10) + 1);
            }
        }

        ArrayList<Item> newUnplacedItemsList = new ArrayList<>(unplacedItems);
        int randomItemIndex1;
        for(int i: visitOrderMap.keySet()){
            if(visitOrderMap.get(i) > 1){
                break;
            }
            return newUnplacedItemsList;
        }
        do{
            randomItemIndex1 = random.nextInt(newUnplacedItemsList.size());

        }while (visitOrderMap.get(newUnplacedItemsList.get(randomItemIndex1).visit_order) <= 1);
        int randomItemIndex2;
        Item item1 = newUnplacedItemsList.get(randomItemIndex1);
        if(lifoSwitch){
            do{
                randomItemIndex2 = random.nextInt(newUnplacedItemsList.size());
            }while(
                    randomItemIndex2 == randomItemIndex1 || newUnplacedItemsList.get(randomItemIndex2).visit_order != unplacedItems.get(randomItemIndex1).visit_order
            );
        }
        else{
            do{
                randomItemIndex2 = random.nextInt(newUnplacedItemsList.size());
            }while(
                    randomItemIndex1 == randomItemIndex2
            );
        }
        newUnplacedItemsList.set(randomItemIndex1, newUnplacedItemsList.get(randomItemIndex2));
        newUnplacedItemsList.set(randomItemIndex2, item1);
        return newUnplacedItemsList;
    }

    public LinkedHashSet<Item> generateUnplacedItems(List<Double> parcel_p, List<Double> parcel_q, List<Double> parcel_r, int boxNumbers) {
        LinkedHashSet<Item> unplacedItems = new LinkedHashSet<>();

        // 使用 parcel_p, parcel_q, parcel_r 创建物品
        for (int i = 0; i < parcel_p.size(); i++) {
            double length = parcel_p.get(i);
            double width = parcel_q.get(i);
            double height = parcel_r.get(i);
            Item item = new Item(i, length, width, height, i, 0); // 默认脆弱物品标记为0
            unplacedItems.add(item);
        }

        return unplacedItems;
    }

//    public LinkedHashSet<Item> generateUnplacedItems(List<Node> visitSequence){
//        LinkedHashSet<Item> unplacedItems = new LinkedHashSet<Item>();
//        // set all items to be unplaced initially
//        for (Node node : visitSequence) {
//            unplacedItems.addAll(node.demands);
//        }
//        // set items lifo sequence
//        setLoadingSequencce(visitSequence);
//        return unplacedItems;
//    }

    public double calculateLoadingLengthTabu(List<Double> L_Packages, List<Double> parcel_p,
                                             List<Double> parcel_q, List<Double> parcel_r,
                                             List<List<Double>> myPackages, int boxNumbers) {
        double initialLength = calculateLoadingLength(L_Packages, parcel_p, parcel_q, parcel_r, myPackages, boxNumbers);


        if(initialLength <= Parameters.L){
            // 记录装箱信息
            if(if_record_to_csv_switch == 1){
                LinkedHashSet<Item> unplacedItems = generateUnplacedItems(parcel_p, parcel_q, parcel_r, boxNumbers);
                ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);

            }
            return initialLength;
        }
        if(initialLength > 1.5*Parameters.L){
            if(if_record_to_csv_switch == 1){
                LinkedHashSet<Item> unplacedItems = generateUnplacedItems(parcel_p, parcel_q, parcel_r, boxNumbers);
                ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);

            }
            return initialLength;}
        // tabu search
        int counter = 0;
        int max_counter = 5;
        LinkedHashSet<Item> unplacedItems = generateUnplacedItems(parcel_p, parcel_q, parcel_r, boxNumbers);
        ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);
        unplacedItemsList.sort((item1, item2) -> {
            // SR1: 如果遵循 LIFO 约束，反向排序客户访问顺序（这里假设我们已经有客户顺序的信息）
            int orderComparison = Integer.compare(item2.visit_order, item1.visit_order);
            if (orderComparison != 0  && lifoSwitch) {
                return orderComparison;
            }

            // SR2: 如果遵循脆弱性约束，非脆弱物品排在脆弱物品前面
            int fragilityComparison = Integer.compare(item1.fragile, item2.fragile);
            if (fragilityComparison != 0 && fragileSwitch) {
                return fragilityComparison; // 非脆弱物品（false）排在脆弱物品（true）前面
            }

            // SR3: 根据体积进行降序排序
            return Double.compare(item2.l * item2.w * item2.h, item1.l * item1.w * item1.h);
        });

        while(counter < max_counter && initialLength > Parameters.L){

            ArrayList<Item> newUnplacedItemsList = new ArrayList<Item>();
            for(int i = 0; i < 10; i++){
                if(unplacedItemsList.size() == 0){
                    int x =0;
                }
                newUnplacedItemsList = randomSwapItemSequence(unplacedItemsList);
                double newLength = calculateLoadingLengthInsideTabu(newUnplacedItemsList);
                // 如果有效更新，就接受
                // 如果有效且已经满足了，直接终止
                if(newLength < initialLength){
                    initialLength = newLength;
                    unplacedItemsList = newUnplacedItemsList;
                    if(initialLength <= Parameters.L){
                        break;
                    }
                } else if (i == 9) {
                    // 如果尝试了10次都没有找到更优解，则终止
                    // 记录装箱信息

                    return initialLength;
                }
            }
            counter++;
        }



        return initialLength;


    }

    public double calculateLoadingLengthInsideTabu(ArrayList<Item> unplacedItemsList) {
        LinkedHashSet<Item> placedItems = new LinkedHashSet<Item>();
        LinkedHashSet<Item> unplacedItems = new LinkedHashSet<Item>();
        ArrayList<Item> wall = new ArrayList<Item>();

        double lambda_placed_tmp = 0.0; // 临时方案的已放下物体的长度
        double lambda_final = Double.MAX_VALUE; // 所有已放下物体的长度

        int dummyvisitorder = 1000;

        // 将左，后，下，右，上5块车板当做虚拟物体，放置在原点
        Item leftDummyItem = new Item(-5, 0, 0, 0, 2 * Parameters.L, 0, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item backDummyItem = new Item(-4, 0, 0, 0, 0, Parameters.W, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item bottomDummyItem = new Item(-3, 0, 0, 0, 2 * Parameters.L, Parameters.W, 0, 0, 0, dummyvisitorder, 0);
        Item rightDummyItem = new Item(-2, 0, Parameters.W, 0, 2 * Parameters.L, 0, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item topDummyItem = new Item(-1, 0, 0, Parameters.H, 2 * Parameters.L, Parameters.W, 0, 0, 0, -1, 1);
        placedItems.add(leftDummyItem);
        placedItems.add(backDummyItem);
        placedItems.add(bottomDummyItem);
        placedItems.add(rightDummyItem);
        placedItems.add(topDummyItem);


        ArrayList<int[]> positionCandidates = new ArrayList<int[]>();

        // set position candidates initially
        positionCandidates.add(new int[]{0, 0, 0});

        // sort unplaced items by decreasing order of volume




        // iterate over unplaced items and place them on the available space
        for (Item item : unplacedItemsList) {
            boolean placed = false;
            int[] position_best = new int[3];
            int orientation_best = 0;
            double lambda_global = Double.MAX_VALUE; // 所有已放下物体的长度
            double lambda_tmp = 0.0; // 临时方案的长度



            for (int[] position : positionCandidates) {
//            for (int orientation = 0; orientation < 6; orientation++) {

                for (int orientation = 0; orientation < orientationsLimit; orientation++) {
//                for (int orientation = 0; orientation < 6; orientation++) {
//                for (int[] position : positionCandidates) {
                    item.orientation = orientation;
//                    int l = item.getDimensions()[0];
                    int w = item.getDimensions()[1];
                    int h = item.getDimensions()[2];

                    if (position[1] + w > Parameters.W || position[2] + h > Parameters.H) {
                        continue;
                    }

                    //表示是否满足这些约束，如果满足约束，则为false
                    boolean ifOverlap = false;
                    boolean ifLifoViolation = false;
                    boolean ifSupportAreaViolation = false;
                    boolean ifFragileViolation = false;
                    double supportArea = 0.0;
                    double supportAreaRatio = 0.0;

                    item.x = position[0];
                    item.y = position[1];
                    item.z = position[2];

                    for (Item otherItem : placedItems) {
                        if (item.if_overlaps(otherItem)) {
                            ifOverlap = true;
                            break;
                        }
                    }

                    //如果没有overlap，则尝试平移，平移之后再判断其他约束
                    if (!ifOverlap) {
                        double slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
                        item.x -= slide_x;
                        double slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
                        item.z -= slide_z;
                        double slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
                        item.y -= slide_y;

                        // support area violation check & lifo violation check
//
                        for (Item otherItem : placedItems) {
                            if(lifoSwitch){
                                if (!is_lifo_satisfied(item, otherItem)){
                                    ifLifoViolation = true;
                                    break;
                                }
                            }
                            if(fragileSwitch){
                                if(!is_fragile_satisified(item, otherItem)){
                                    ifFragileViolation = true;
                                    break;
                                }
                            }

                            supportArea += calculate_support_area(item, otherItem);
                        }
                        supportAreaRatio = supportArea / (item.getDimensions()[0] * item.getDimensions()[1]);
                        if(supportAreaSwitch){
                            //为了防止装载的货物长度超过2倍车长后下方的dummy车板没法提供足够的支撑面积，对于z=0的货物直接通过support_ratio的验证
                            if(item.z == 0){
                                ifSupportAreaViolation = false;
                            }
                            else{
                                if (supportAreaRatio < Parameters.supportAreaFactor) {
                                    ifSupportAreaViolation = true;
                                }
                            }
                        }

                        if (!ifLifoViolation &&!ifSupportAreaViolation && !ifFragileViolation) {
                            placed = true;

                            // 计算方案长度
                            for (Item other : placedItems) {
                                if (!(other.equals(backDummyItem) || other.equals(bottomDummyItem) ||
                                        other.equals(topDummyItem) || other.equals(rightDummyItem) ||
                                        other.equals(leftDummyItem))){
                                    lambda_placed_tmp = Math.max(other.x + other.getDimensions()[0], lambda_placed_tmp);
                                }
                            }
                            lambda_tmp = Math.max(item.x + item.getDimensions()[0], lambda_placed_tmp);
                            if(lambda_tmp <= lambda_global){
                                lambda_global = lambda_tmp;
                                position_best = position;
                                orientation_best = orientation;
                                lambda_final = lambda_global;
                            }
                        }
                    }

                }
                if(placed) {
                    break;
                }
            }


            if(!placed) {
                // DBLF算法放不下，此时尝试DBLF+算法，在DBLF基础上，在遍历positionCandidates时考虑slide,按照DBL顺序，如果有一个方向
                // 可以slide,否则尝试另外的方向，直至有一个可以slide的位置，然后在此处继续执行普通DBLF的ifOverlap检查，如果不通过，则换一个positionCandidate,
                // 如果没有overlap,则像经典DBLF一样，尝试平移，and so on.
//                System.out.println("DBLF+ algorithm is used to place item " + item.id + "."); //要开始用DBLF+了。

                for (int[] position : positionCandidates) {
//                for (int orientation = 0; orientation < 6; orientation++) {

                    for (int orientation = 0; orientation < orientationsLimit; orientation++) {
//                    for (int orientation = 0; orientation < 6; orientation++) {
//                    for (int[] position : positionCandidates) {
                        item.orientation = orientation;
//                        int l = item.getDimensions()[0];
                        int w = item.getDimensions()[1];
                        int h = item.getDimensions()[2];

                        item.x = position[0];
                        item.y = position[1];
                        item.z = position[2];
                        double pre_slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
//                        System.out.println("pre_slide_x: "+pre_slide_x);
                        if (pre_slide_x > 0) {
//                            System.out.println("try to slide item " + item.id + " in x direction.");
                            position[0] -= (int) pre_slide_x;
                        }
                        else{
                            double pre_slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
//                            System.out.println("pre_slide_z: "+pre_slide_z);
                            if (pre_slide_z > 0) {
//                                System.out.println("try to slide item " + item.id + " in z direction.");
                                position[2] -= (int) pre_slide_z;
                            }
                            else{
                                double pre_slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
//                                System.out.println("pre_slide_y: "+pre_slide_y);
                                if (pre_slide_y > 0) {
//                                    System.out.println("try to slide item " + item.id + " in y direction.");
                                    position[1] -= (int) pre_slide_y;
                                }
                                else{
                                    continue;
                                }
                            }
                        }
                        item.x = position[0];
                        item.y = position[1];
                        item.z = position[2];

                        if (position[1] + w > Parameters.W || position[2] + h > Parameters.H) {
                            continue;
                        }

                        //表示是否满足这些约束，如果满足约束，则为false
                        boolean ifOverlap = false;
                        boolean ifLifoViolation = false;
                        boolean ifSupportAreaViolation = false;
                        boolean ifFragileViolation = false;
                        double supportArea = 0.0;
                        double supportAreaRatio = 0.0;



                        for (Item otherItem : placedItems) {
                            if (item.if_overlaps(otherItem)) {
                                ifOverlap = true;
                                break;
                            }
                        }

                        //如果没有overlap，则尝试平移，平移之后再判断其他约束
                        if (!ifOverlap) {
                            double slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
                            item.x -= slide_x;
                            double slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
                            item.z -= slide_z;
                            double slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
                            item.y -= slide_y;

                            // support area violation check & lifo violation check
//                            System.out.println("support area check");
                            for (Item otherItem : placedItems) {
                                if(lifoSwitch){
                                    if (!is_lifo_satisfied(item, otherItem)){
                                        ifLifoViolation = true;
                                        break;
                                    }
                                }
                                if(fragileSwitch){
                                    if(!is_fragile_satisified(item, otherItem)){
                                        ifFragileViolation = true;
                                        break;
                                    }
                                }

                                supportArea += calculate_support_area(item, otherItem);
                            }
                            supportAreaRatio = supportArea / (item.getDimensions()[0] * item.getDimensions()[1]);
                            if(supportAreaSwitch){
                                //为了防止装载的货物长度超过2倍车长后下方的dummy车板没法提供足够的支撑面积，对于z=0的货物直接通过support_ratio的验证
                                if(item.z == 0){
                                    ifSupportAreaViolation = false;
                                }
                                else{
                                    if (supportAreaRatio < Parameters.supportAreaFactor) {
                                        ifSupportAreaViolation = true;
                                    }
                                }
                            }


                            if (!ifLifoViolation &&!ifSupportAreaViolation && !ifFragileViolation) {
                                placed = true;

                                // 计算方案长度
                                for (Item other : placedItems) {
                                    // 确保这个条件的逻辑正确
                                    if (!(other.equals(backDummyItem) || other.equals(bottomDummyItem) ||
                                            other.equals(topDummyItem) || other.equals(rightDummyItem) ||
                                            other.equals(leftDummyItem))){
                                        lambda_placed_tmp = Math.max(other.x + other.getDimensions()[0], lambda_placed_tmp);
                                    }
                                }
                                lambda_tmp = Math.max(item.x + item.getDimensions()[0], lambda_placed_tmp);
                                if(lambda_tmp <= lambda_global){
                                    lambda_global = lambda_tmp;
                                    position_best = position;
                                    orientation_best = orientation;
                                    lambda_final = lambda_global;
                                }
                            }
                        }

                    }
                    if(placed) {
                        break;
                    }
                }
                if(placed) {
                    item.orientation = orientation_best;
                    item.x = position_best[0];
                    item.y = position_best[1];
                    item.z = position_best[2];
                    placedItems.add(item);
                    unplacedItems.remove(item);
                    positionCandidates.remove(position_best);
                    positionCandidates.add(new int[]{position_best[0] + item.getDimensions()[0], position_best[1], position_best[2]});
                    positionCandidates.add(new int[]{position_best[0], position_best[1] + item.getDimensions()[1], position_best[2]});
                    positionCandidates.add(new int[]{position_best[0], position_best[1], position_best[2] + item.getDimensions()[2]});
                    positionCandidates.sort((position1, position2) -> {
                        if (position1[0] != position2[0]) {
                            return Integer.compare(position1[0], position2[0]); // 比较 x 坐标
                        } else if (position1[2] != position2[2]) {
                            return Integer.compare(position1[2], position2[2]); // 比较 z 坐标
                        } else {
                            return Integer.compare(position1[1], position2[1]); // 比较 y 坐标
                        }
                    });
                }
                // DBLF+算法也放不下，输出失败信息，并画图记录，返回Double.MAX_VALUE
                else{
                    // 放不下，则放弃该物体，并输出信息，停止装箱
//                    System.out.println("Cannot place item " + item.id + " in the current configuration."+"global loading length: "+lambda_final);
//                    System.out.println("Placed items: ");
//                    for (Item placedItem : placedItems) {
//                        System.out.println(placedItem.id);
//                    }
//                    System.out.println("vehicle informations:");
//                    System.out.println("L: "+Parameters.L);
//                    System.out.println("W: "+Parameters.W);
//                    System.out.println("H: "+Parameters.H);
//              System.out.println("supportAreaFactor: "+Parameters.supportAreaFactor);
//                LoadingVisualizer ld = new LoadingVisualizer(new ArrayList<Item>(placedItems));
//                ld.plotPacking();

                    // 画出装箱图


                    StringBuilder mark = new StringBuilder(item.id + "unable_item_id");

                    for(Item placedItem: placedItems){
                        mark.append(placedItem.id);
                        mark.append("_");

                    }

                    List<Item> placedItemsList = new ArrayList<Item>(placedItems);

                    for (Item placeditem: placedItemsList){
                        if ((placeditem.equals(backDummyItem) || placeditem.equals(bottomDummyItem) ||
                                placeditem.equals(leftDummyItem))){
                            wall.add(placeditem);
                            placedItems.remove(placeditem);
                        }
                        if ((placeditem.equals(topDummyItem) || placeditem.equals(rightDummyItem))){
                            placedItems.remove(placeditem);
                        }
                    }


                    return Double.MAX_VALUE;
                }
            }

            // 放置成功，更新数据
            else {
                item.orientation = orientation_best;
                item.x = position_best[0];
                item.y = position_best[1];
                item.z = position_best[2];
                placedItems.add(item);
                unplacedItems.remove(item);
                positionCandidates.remove(position_best);
                positionCandidates.add(new int[]{position_best[0] + item.getDimensions()[0], position_best[1], position_best[2]});
                positionCandidates.add(new int[]{position_best[0], position_best[1] + item.getDimensions()[1], position_best[2]});
                positionCandidates.add(new int[]{position_best[0], position_best[1], position_best[2] + item.getDimensions()[2]});
                positionCandidates.sort((position1, position2) -> {
                    if (position1[0] != position2[0]) {
                        return Integer.compare(position1[0], position2[0]); // 比较 x 坐标
                    } else if (position1[2] != position2[2]) {
                        return Integer.compare(position1[2], position2[2]); // 比较 z 坐标
                    } else {
                        return Integer.compare(position1[1], position2[1]); // 比较 y 坐标
                    }
                });
            }
        }



//        System.out.println("global loading length: "+lambda_final);
//        for (Item item: placedItems){
//            System.out.println("id"+item.id);
//            System.out.println("x"+item.x+"y"+item.y+"z"+item.z+"l"+item.getDimensions()[0]+"w"+item.getDimensions()[1]+"h"+item.getDimensions()[2]);
//        }

        // 画出装箱图
        StringBuilder mark = new StringBuilder("item_id333333333");

        for(Item placedItem: placedItems){
            mark.append(placedItem.id);
            mark.append("_");

        }

        List<Item> placedItemsList = new ArrayList<Item>(placedItems);
        for (Item item: placedItemsList){
            if ((item.equals(backDummyItem) || item.equals(bottomDummyItem) ||
                    item.equals(leftDummyItem))){
                wall.add(item);
                placedItems.remove(item);
            }
            if ((item.equals(topDummyItem) || item.equals(rightDummyItem))){
                placedItems.remove(item);
            }
        }


        placedItemsList = new ArrayList<Item>(placedItems);


        return lambda_final;
    }


//    public double calculateLoadingLengthTabu(List<Node> visitSequence){
//        double initialLength = calculateLoadingLength(visitSequence);
//        if(initialLength <= Parameters.L){
//            // 记录装箱信息
//            if(if_record_to_csv_switch == 1){
//                LinkedHashSet<Item> unplacedItems = generateUnplacedItems(visitSequence);
//                ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);
//
//            }
//            return initialLength;
//        }
//        if(initialLength > 1.5*Parameters.L){
//            if(if_record_to_csv_switch == 1){
//                LinkedHashSet<Item> unplacedItems = generateUnplacedItems(visitSequence);
//                ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);
//
//            }
//            return initialLength;}
//        // tabu search
//        int counter = 0;
//        int max_counter = 5;
//        LinkedHashSet<Item> unplacedItems = generateUnplacedItems(visitSequence);
//        ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);
//        unplacedItemsList.sort((item1, item2) -> {
//            // SR1: 如果遵循 LIFO 约束，反向排序客户访问顺序（这里假设我们已经有客户顺序的信息）
//            int orderComparison = Integer.compare(item2.visit_order, item1.visit_order);
//            if (orderComparison != 0  && lifoSwitch) {
//                return orderComparison;
//            }
//
//            // SR2: 如果遵循脆弱性约束，非脆弱物品排在脆弱物品前面
//            int fragilityComparison = Integer.compare(item1.fragile, item2.fragile);
//            if (fragilityComparison != 0 && fragileSwitch) {
//                return fragilityComparison; // 非脆弱物品（false）排在脆弱物品（true）前面
//            }
//
//            // SR3: 根据体积进行降序排序
//            return Double.compare(item2.l * item2.w * item2.h, item1.l * item1.w * item1.h);
//        });
//
//        while(counter < max_counter && initialLength > Parameters.L){
//
//            ArrayList<Item> newUnplacedItemsList = new ArrayList<Item>();
//            for(int i = 0; i < 10; i++){
//                if(unplacedItemsList.size() == 0){
//                    int x =0;
//                }
//                newUnplacedItemsList = randomSwapItemSequence(unplacedItemsList);
//                double newLength = calculateLoadingLengthInsideTabu(newUnplacedItemsList);
//                // 如果有效更新，就接受
//                // 如果有效且已经满足了，直接终止
//                if(newLength < initialLength){
//                    initialLength = newLength;
//                    unplacedItemsList = newUnplacedItemsList;
//                    if(initialLength <= Parameters.L){
//                        break;
//                    }
//                } else if (i == 9) {
//                    // 如果尝试了10次都没有找到更优解，则终止
//                    // 记录装箱信息
//
//                    return initialLength;
//                }
//            }
//            counter++;
//        }
//
//
//
//        return initialLength;
//
//
//    }



    public double calculateLoadingLength(List<Double> L_Packages, List<Double> parcel_p,
                                         List<Double> parcel_q, List<Double> parcel_r,
                                         List<List<Double>> myPackages, int boxNumbers) {

        if(parcel_p.size() <=2){
            return 0.0;
        }

        LinkedHashSet<Item> placedItems = new LinkedHashSet<Item>();
        LinkedHashSet<Item> unplacedItems = new LinkedHashSet<Item>();
        ArrayList<Item> wall = new ArrayList<Item>();

        double lambda_placed_tmp = 0.0; // 临时方案的已放下物体的长度
        double lambda_final = Double.MAX_VALUE; // 所有已放下物体的长度

        int dummyvisitorder = 1000;

        // 将左，后，下，右，上5块车板当做虚拟物体，放置在原点
        Item leftDummyItem = new Item(-5, 0, 0, 0, 2 * Parameters.L, 0, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item backDummyItem = new Item(-4, 0, 0, 0, 0, Parameters.W, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item bottomDummyItem = new Item(-3, 0, 0, 0, 2 * Parameters.L, Parameters.W, 0, 0, 0, dummyvisitorder, 0);
        Item rightDummyItem = new Item(-2, 0, Parameters.W, 0, 2 * Parameters.L, 0, Parameters.H, 0, 0, dummyvisitorder, 0);
        Item topDummyItem = new Item(-1, 0, 0, Parameters.H, 2 * Parameters.L, Parameters.W, 0, 0, 0, -1, 1);
        placedItems.add(leftDummyItem);
        placedItems.add(backDummyItem);
        placedItems.add(bottomDummyItem);
        placedItems.add(rightDummyItem);
        placedItems.add(topDummyItem);

        unplacedItems= generateUnplacedItems(parcel_p, parcel_q, parcel_r, boxNumbers);

        ArrayList<int[]> positionCandidates = new ArrayList<int[]>();

        // set position candidates initially
        positionCandidates.add(new int[]{0, 0, 0});

        // sort unplaced items by decreasing order of volume
        ArrayList<Item> unplacedItemsList = new ArrayList<Item>(unplacedItems);

        //  按照lifo顺序由大到小；按照fragile排，infragile的放在前，fragile的放在后；根据体积进行降序排序
        unplacedItemsList.sort((item1, item2) -> {
            // SR1: 如果遵循 LIFO 约束，反向排序客户访问顺序（这里假设我们已经有客户顺序的信息）
            int orderComparison = Integer.compare(item2.visit_order, item1.visit_order);
            if (orderComparison != 0  && lifoSwitch) {
                return orderComparison;
            }

            // SR2: 如果遵循脆弱性约束，非脆弱物品排在脆弱物品前面
            int fragilityComparison = Integer.compare(item1.fragile, item2.fragile);
            if (fragilityComparison != 0 && fragileSwitch) {
                return fragilityComparison; // 非脆弱物品（false）排在脆弱物品（true）前面
            }

            // SR3: 根据体积进行降序排序
            return Double.compare(item2.l * item2.w * item2.h, item1.l * item1.w * item1.h);
        });


        // iterate over unplaced items and place them on the available space
        for (Item item : unplacedItemsList) {
            boolean placed = false;
            int[] position_best = new int[3];
            int orientation_best = 0;
            double lambda_global = Double.MAX_VALUE; // 所有已放下物体的长度
            double lambda_tmp = 0.0; // 临时方案的长度



            for (int[] position : positionCandidates) {
//            for (int orientation = 0; orientation < 6; orientation++) {

                for (int orientation = 0; orientation < orientationsLimit; orientation++) {
//                for (int orientation = 0; orientation < 6; orientation++) {
//                for (int[] position : positionCandidates) {
                    item.orientation = orientation;
//                    int l = item.getDimensions()[0];
                    int w = item.getDimensions()[1];
                    int h = item.getDimensions()[2];

                    if (position[1] + w > Parameters.W || position[2] + h > Parameters.H) {
                        continue;
                    }

                    //表示是否满足这些约束，如果满足约束，则为false
                    boolean ifOverlap = false;
                    boolean ifLifoViolation = false;
                    boolean ifSupportAreaViolation = false;
                    boolean ifFragileViolation = false;
                    double supportArea = 0.0;
                    double supportAreaRatio = 0.0;

                    item.x = position[0];
                    item.y = position[1];
                    item.z = position[2];

                    for (Item otherItem : placedItems) {
                        if (item.if_overlaps(otherItem)) {
                            ifOverlap = true;
                            break;
                        }
                    }

                    //如果没有overlap，则尝试平移，平移之后再判断其他约束
                    if (!ifOverlap) {
                        double slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
                        item.x -= slide_x;
                        double slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
                        item.z -= slide_z;
                        double slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
                        item.y -= slide_y;

                        // support area violation check & lifo violation check
//
                        for (Item otherItem : placedItems) {
                            if(lifoSwitch){
                                if (!is_lifo_satisfied(item, otherItem)){
                                    ifLifoViolation = true;
                                    break;
                                }
                            }
                            if(fragileSwitch){
                                if(!is_fragile_satisified(item, otherItem)){
                                    ifFragileViolation = true;
                                    break;
                                }
                            }

                            supportArea += calculate_support_area(item, otherItem);
                        }
                        supportAreaRatio = supportArea / (item.getDimensions()[0] * item.getDimensions()[1]);
                        if(supportAreaSwitch){
                            //为了防止装载的货物长度超过2倍车长后下方的dummy车板没法提供足够的支撑面积，对于z=0的货物直接通过support_ratio的验证
                            if(item.z == 0){
                                ifSupportAreaViolation = false;
                            }
                            else{
                                if (supportAreaRatio < Parameters.supportAreaFactor) {
                                    ifSupportAreaViolation = true;
                                }
                            }
                        }

                        if (!ifLifoViolation &&!ifSupportAreaViolation && !ifFragileViolation) {
                            placed = true;

                            // 计算方案长度
                            for (Item other : placedItems) {
                                if (!(other.equals(backDummyItem) || other.equals(bottomDummyItem) ||
                                        other.equals(topDummyItem) || other.equals(rightDummyItem) ||
                                        other.equals(leftDummyItem))){
                                    lambda_placed_tmp = Math.max(other.x + other.getDimensions()[0], lambda_placed_tmp);
                                }
                            }
                            lambda_tmp = Math.max(item.x + item.getDimensions()[0], lambda_placed_tmp);
                            if(lambda_tmp <= lambda_global){
                                lambda_global = lambda_tmp;
                                position_best = position;
                                orientation_best = orientation;
                                lambda_final = lambda_global;
                            }
                        }
                    }

                }
                if(placed) {
                    break;
                }
            }


            if(!placed) {
                // DBLF算法放不下，此时尝试DBLF+算法，在DBLF基础上，在遍历positionCandidates时考虑slide,按照DBL顺序，如果有一个方向
                // 可以slide,否则尝试另外的方向，直至有一个可以slide的位置，然后在此处继续执行普通DBLF的ifOverlap检查，如果不通过，则换一个positionCandidate,
                // 如果没有overlap,则像经典DBLF一样，尝试平移，and so on.
//                System.out.println("DBLF+ algorithm is used to place item " + item.id + "."); //要开始用DBLF+了。

                for (int[] position : positionCandidates) {
//                for (int orientation = 0; orientation < 6; orientation++) {

                    for (int orientation = 0; orientation < orientationsLimit; orientation++) {
//                    for (int orientation = 0; orientation < 6; orientation++) {
//                    for (int[] position : positionCandidates) {
                        item.orientation = orientation;
//                        int l = item.getDimensions()[0];
                        int w = item.getDimensions()[1];
                        int h = item.getDimensions()[2];

                        item.x = position[0];
                        item.y = position[1];
                        item.z = position[2];
                        double pre_slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
//                        System.out.println("pre_slide_x: "+pre_slide_x);
                        if (pre_slide_x > 0) {
//                            System.out.println("try to slide item " + item.id + " in x direction.");
                            position[0] -= (int) pre_slide_x;
                        }
                        else{
                            double pre_slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
//                            System.out.println("pre_slide_z: "+pre_slide_z);
                            if (pre_slide_z > 0) {
//                                System.out.println("try to slide item " + item.id + " in z direction.");
                                position[2] -= (int) pre_slide_z;
                            }
                            else{
                                double pre_slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
//                                System.out.println("pre_slide_y: "+pre_slide_y);
                                if (pre_slide_y > 0) {
//                                    System.out.println("try to slide item " + item.id + " in y direction.");
                                    position[1] -= (int) pre_slide_y;
                                }
                                else{
                                    continue;
                                }
                            }
                        }
                        item.x = position[0];
                        item.y = position[1];
                        item.z = position[2];

                        if (position[1] + w > Parameters.W || position[2] + h > Parameters.H) {
                            continue;
                        }

                        //表示是否满足这些约束，如果满足约束，则为false
                        boolean ifOverlap = false;
                        boolean ifLifoViolation = false;
                        boolean ifSupportAreaViolation = false;
                        boolean ifFragileViolation = false;
                        double supportArea = 0.0;
                        double supportAreaRatio = 0.0;



                        for (Item otherItem : placedItems) {
                            if (item.if_overlaps(otherItem)) {
                                ifOverlap = true;
                                break;
                            }
                        }

                        //如果没有overlap，则尝试平移，平移之后再判断其他约束
                        if (!ifOverlap) {
                            double slide_x = find_max_X_slide_distance_using_projections(item, placedItems);
                            item.x -= slide_x;
                            double slide_z = find_max_Z_slide_distance_using_projections(item, placedItems);
                            item.z -= slide_z;
                            double slide_y = find_max_Y_slide_distance_using_projections(item, placedItems);
                            item.y -= slide_y;

                            // support area violation check & lifo violation check
//                            System.out.println("support area check");
                            for (Item otherItem : placedItems) {
                                if(lifoSwitch){
                                    if (!is_lifo_satisfied(item, otherItem)){
                                        ifLifoViolation = true;
                                        break;
                                    }
                                }
                                if(fragileSwitch){
                                    if(!is_fragile_satisified(item, otherItem)){
                                        ifFragileViolation = true;
                                        break;
                                    }
                                }

                                supportArea += calculate_support_area(item, otherItem);
                            }
                            supportAreaRatio = supportArea / (item.getDimensions()[0] * item.getDimensions()[1]);
                            if(supportAreaSwitch){
                                //为了防止装载的货物长度超过2倍车长后下方的dummy车板没法提供足够的支撑面积，对于z=0的货物直接通过support_ratio的验证
                                if(item.z == 0){
                                    ifSupportAreaViolation = false;
                                }
                                else{
                                    if (supportAreaRatio < Parameters.supportAreaFactor) {
                                        ifSupportAreaViolation = true;
                                    }
                                }
                            }


                            if (!ifLifoViolation &&!ifSupportAreaViolation && !ifFragileViolation) {
                                placed = true;

                                // 计算方案长度
                                for (Item other : placedItems) {
                                    // 确保这个条件的逻辑正确
                                    if (!(other.equals(backDummyItem) || other.equals(bottomDummyItem) ||
                                            other.equals(topDummyItem) || other.equals(rightDummyItem) ||
                                            other.equals(leftDummyItem))){
                                        lambda_placed_tmp = Math.max(other.x + other.getDimensions()[0], lambda_placed_tmp);
                                    }
                                }
                                lambda_tmp = Math.max(item.x + item.getDimensions()[0], lambda_placed_tmp);
                                if(lambda_tmp <= lambda_global){
                                    lambda_global = lambda_tmp;
                                    position_best = position;
                                    orientation_best = orientation;
                                    lambda_final = lambda_global;
                                }
                            }
                        }

                    }
                    if(placed) {
                        break;
                    }
                }
                if(placed) {
                    item.orientation = orientation_best;
                    item.x = position_best[0];
                    item.y = position_best[1];
                    item.z = position_best[2];
                    placedItems.add(item);
                    unplacedItems.remove(item);
                    positionCandidates.remove(position_best);
                    positionCandidates.add(new int[]{position_best[0] + item.getDimensions()[0], position_best[1], position_best[2]});
                    positionCandidates.add(new int[]{position_best[0], position_best[1] + item.getDimensions()[1], position_best[2]});
                    positionCandidates.add(new int[]{position_best[0], position_best[1], position_best[2] + item.getDimensions()[2]});
                    positionCandidates.sort((position1, position2) -> {
                        if (position1[0] != position2[0]) {
                            return Integer.compare(position1[0], position2[0]); // 比较 x 坐标
                        } else if (position1[2] != position2[2]) {
                            return Integer.compare(position1[2], position2[2]); // 比较 z 坐标
                        } else {
                            return Integer.compare(position1[1], position2[1]); // 比较 y 坐标
                        }
                    });
                }
                // DBLF+算法也放不下，输出失败信息，并画图记录，返回Double.MAX_VALUE
                else{
                    // 放不下，则放弃该物体，并输出信息，停止装箱
//                    System.out.println("Cannot place item " + item.id + " in the current configuration."+"global loading length: "+lambda_final);
//                    System.out.println("Placed items: ");
//                    for (Item placedItem : placedItems) {
//                        System.out.println(placedItem.id);
//                    }
//                    System.out.println("vehicle informations:");
//                    System.out.println("L: "+Parameters.L);
//                    System.out.println("W: "+Parameters.W);
//                    System.out.println("H: "+Parameters.H);
//              System.out.println("supportAreaFactor: "+Parameters.supportAreaFactor);
//                LoadingVisualizer ld = new LoadingVisualizer(new ArrayList<Item>(placedItems));
//                ld.plotPacking();


                    // 画出装箱图
                    StringBuilder mark = new StringBuilder(item.id + "unable_item_id");

                    for(Item placedItem: placedItems){
                        mark.append(placedItem.id);
                        mark.append("_");

                    }

                    List<Item> placedItemsList = new ArrayList<Item>(placedItems);

                    for (Item placeditem: placedItemsList){
                        if ((placeditem.equals(backDummyItem) || placeditem.equals(bottomDummyItem) ||
                                placeditem.equals(leftDummyItem))){
                            wall.add(placeditem);
                            placedItems.remove(placeditem);
                        }
                        if ((placeditem.equals(topDummyItem) || placeditem.equals(rightDummyItem))){
                            placedItems.remove(placeditem);
                        }
                    }
                    placedItemsList = new ArrayList<Item>(placedItems);

                    return Double.MAX_VALUE;
                }
            }

            // 放置成功，更新数据
            else {
                item.orientation = orientation_best;
                item.x = position_best[0];
                item.y = position_best[1];
                item.z = position_best[2];
                placedItems.add(item);
                unplacedItems.remove(item);
                positionCandidates.remove(position_best);
                positionCandidates.add(new int[]{position_best[0] + item.getDimensions()[0], position_best[1], position_best[2]});
                positionCandidates.add(new int[]{position_best[0], position_best[1] + item.getDimensions()[1], position_best[2]});
                positionCandidates.add(new int[]{position_best[0], position_best[1], position_best[2] + item.getDimensions()[2]});
                positionCandidates.sort((position1, position2) -> {
                    if (position1[0] != position2[0]) {
                        return Integer.compare(position1[0], position2[0]); // 比较 x 坐标
                    } else if (position1[2] != position2[2]) {
                        return Integer.compare(position1[2], position2[2]); // 比较 z 坐标
                    } else {
                        return Integer.compare(position1[1], position2[1]); // 比较 y 坐标
                    }
                });
            }
        }



//        System.out.println("global loading length: "+lambda_final);
//        for (Item item: placedItems){
//            System.out.println("id"+item.id);
//            System.out.println("x"+item.x+"y"+item.y+"z"+item.z+"l"+item.getDimensions()[0]+"w"+item.getDimensions()[1]+"h"+item.getDimensions()[2]);
//        }


        StringBuilder mark = new StringBuilder("item_id22222222");

        for(Item placedItem: placedItems){
            mark.append(placedItem.id);
            mark.append("_");

        }

        List<Item> placedItemsList = new ArrayList<Item>(placedItems);
        for (Item item: placedItemsList){
            if ((item.equals(backDummyItem) || item.equals(bottomDummyItem) ||
                    item.equals(leftDummyItem))){
                wall.add(item);
                placedItems.remove(item);
            }
            if ((item.equals(topDummyItem) || item.equals(rightDummyItem))){
                placedItems.remove(item);
            }
        }


        placedItemsList = new ArrayList<Item>(placedItems);


//
        return lambda_final;
    }


    // 通过投影关系找一个方向上最大的可以滑动的距离
    public double find_max_X_slide_distance_using_projections(Item new_item, LinkedHashSet<Item> placed_items) {
        double max_slide_x = Double.MAX_VALUE;
        for (Item other : placed_items) {
            if (other.getDimensions()[0] + other.x > new_item.x) {
                continue;
            }
            if (new_item.y + new_item.getDimensions()[1] < other.y || new_item.y > other.y + other.getDimensions()[1] ||
                    new_item.z + new_item.getDimensions()[2] < other.z || new_item.z > other.z + other.getDimensions()[2]) {
                continue;
            }
            else {
                max_slide_x = Math.min(new_item.x - other.getDimensions()[0]-other.x, max_slide_x);
            }
        }
        if (max_slide_x == Double.MAX_VALUE) {
            return 0.0;
        }
        return max_slide_x;
    }

    public double find_max_Y_slide_distance_using_projections(Item new_item, LinkedHashSet<Item> placed_items) {
        double max_slide_y = Double.MAX_VALUE;
        for (Item other : placed_items) {
            if (other.getDimensions()[1] + other.y > new_item.y) {
                continue;
            }
            if (new_item.x + new_item.getDimensions()[0] < other.x || new_item.x > other.x + other.getDimensions()[0] ||
                    new_item.z + new_item.getDimensions()[2] < other.z || new_item.z > other.z + other.getDimensions()[2]) {
                continue;
            } else {
                max_slide_y = Math.min(new_item.y - other.getDimensions()[1]- other.y, max_slide_y);
            }
        }
        if (max_slide_y == Double.MAX_VALUE) {
            return 0.0;
        }
        return max_slide_y;
    }

    public double find_max_Z_slide_distance_using_projections(Item new_item, LinkedHashSet<Item> placed_items) {
        double max_slide_z = Double.MAX_VALUE;
        for (Item other : placed_items) {
            if (other.getDimensions()[2] + other.z > new_item.z) {
                continue;
            }
            if (new_item.x + new_item.getDimensions()[0] < other.x || new_item.x > other.x + other.getDimensions()[0] ||
                    new_item.y + new_item.getDimensions()[1] < other.y || new_item.y > other.y + other.getDimensions()[1]) {
                continue;
            } else {
                max_slide_z = Math.min(new_item.z - other.getDimensions()[2]- other.z, max_slide_z);
            }
        }
        if (max_slide_z == Double.MAX_VALUE) {
            return 0.0;
        }
        return max_slide_z;
    }

    // 用作求交集的函数
    public double calIntersection(double z1, double Dz1, double z2, double Dz2){
        // 区间 [z1, z1+Dz1]
        double start1 = z1;
        double end1 = z1+Dz1;

        // 区间 [z2, z2+Dz2]
        double start2 = z2;
        double end2 = z2+Dz2;

        // 计算交集
        double intersection_start = Math.max(start1, start2);
        double intersection_end = Math.min(end1, end2);

        // 如果没有交集，交集长度为 0
        if (intersection_start >= intersection_end) {
            return 0.0;
        } else {
            return intersection_end - intersection_start;
        }
    }

    // 判断是否满足LIFO约束，item1 在item2前面,或在item2下面

    public boolean is_lifo_satisfied(Item item1, Item item2) {
        // 如果item1 在 item2 后面，且yz平面上投影有交集,那么如果item1.order > item2.order，则不满足LIFO约束
        if (item1.visit_order > item2.visit_order && item1.x>=item2.x +item2.getDimensions()[0]&&calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) *
                calIntersection(item1.z, item1.getDimensions()[2], item2.z, item2.getDimensions()[2]) > 0) {
            return false;
        }
        // 如果item1 在 item2 前面，且yz平面上投影有交集,那么如果item1.order < item2.order，则不满足LIFO约束
        if (item1.visit_order < item2.visit_order && item1.x+item1.getDimensions()[0]<=item2.x&&calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) *
                calIntersection(item1.z, item1.getDimensions()[2], item2.z, item2.getDimensions()[2]) > 0) {
            return false;
        }
        // 如果item1 在 item2 下面，且xz平面上投影有交集,那么如果item1.order < item2.order，则不满足LIFO约束
        if (item1.visit_order < item2.visit_order && item1.z+item1.getDimensions()[2]<=item2.z&&calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]) *
                calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) > 0) {
            return false;
        }
        // 如果item1 在 item2 上面，且xz平面上投影有交集,那么如果item1.order > item2.order，则不满足LIFO约束
        if (item1.visit_order > item2.visit_order && item1.z>=item2.z +item2.getDimensions()[2]&&calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]) *
                calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) > 0) {
            return false;
        }
        return true;


    }
//    public boolean is_lifo_satisfied(Item item1, Item item2) {
//        if (item1.visit_order >= item2.visit_order) {
//            return true;
//        }
//        // 前后堵塞： y-z平面上投影有交集，并且x轴方向上 item1 在item2前面
//        if (item1.x + item1.getDimensions()[0] <= item2.x&& calIntersection(item1.z, item1.getDimensions()[2], item2.z, item2.getDimensions()[2]) *
//                calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) > 0) {
//            return false;
//        }
//        // 上下堵塞： x-y平面上投影有交集，并且z轴方向上 item1 在items下面
//        if (item1.z + item1.getDimensions()[2] <= item2.z && calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]) *
//                calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]) > 0) {
//            return false;
//        }
//        return true;
//    }

    // 计算支撑面积 item1 如果放在item2的上面，则计算一个s_q,表示由item2 提供给item1 的支持
    public double calculate_support_area(Item item1, Item item2) {
        double s_q = 0.0;
        if(item1.z != item2.z + item2.getDimensions()[2]){
            return 0.0;
        }
        double a_xy = calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]);
        double b_xy = calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]);
        s_q = a_xy * b_xy;
        return s_q;
    }

    // 判断是否满足fragile约束
    public boolean is_fragile_satisified(Item item1, Item item2) {
        // 两个物体的fragile属性相同，则返回true
        if (item1.fragile == item2.fragile) {
            return true;
        }
        // 两个物体的fragile属性不同，则判断是否有碰撞
        // 如果item1 易碎，则不能放在item2 下面
        if (item1.fragile == 1 && item2.fragile == 0) {
            // 先检查两个物体是否上下相邻
            if (item1.z + item1.getDimensions()[2] != item2.z) {
                return true;
            }
            double a_xy = calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]);
            double b_xy = calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]);
            if (a_xy * b_xy > 0) {
                return false;
            }
            return true;
        }
        // 反之同理
        if (item1.fragile == 0 && item2.fragile == 1) {
            // 先检查两个物体是否上下相邻
            if (item1.z != item2.z + item2.getDimensions()[2]) {
                return true;
            }
            double a_xy = calIntersection(item1.y, item1.getDimensions()[1], item2.y, item2.getDimensions()[1]);
            double b_xy = calIntersection(item1.x, item1.getDimensions()[0], item2.x, item2.getDimensions()[0]);
            if (a_xy * b_xy > 0) {
                return false;
            }
            return true;
        }
        return true;
    }

    public void setLoadingSequencce(List<Node> visitSequence){
        for(int i=0;i<visitSequence.size();i++){
            for(Item item: visitSequence.get(i).demands){
                item.visit_order = i;
            }
        }
    }

}



