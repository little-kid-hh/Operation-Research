package org.example.RouteElement;

public class Parameters {

    public static double c_bar = 10;
    public static double D = 90; //D is the weight capacity of a vehicle
    public static int L= 60; //L is the vehicle length
    public static int W = 25; //W is the vehicle width
    public static int H = 30; //H is the vehicle height
    public static double alpha = 20*c_bar/D;
    public static double beta = 20*c_bar/L;
//    public static double beta = 0;
    public static int vehicleCnt = 1; // 可用载具数量
    public static double supportAreaFactor = 0.75; // 最小支撑系数

    public static int intra2OptWeight = 1000;
    public static int inter2OptWeight = 4500;
    public static int intraSwapWeight = 1000;
    public static int interSwapWeight = 1000;
    public static int intraRelocateWeight = 3000;
    public static int interRelocateWeight = 3000;

    public static double distanceMatrix[][] = null; // 距离矩阵

    public static int file_num;

    public static double[] weights;
    public static double[] bias;
    public static double[] min;
    public static double[] max;

    public static boolean plot_switch = false;
    public static long total_search_stage_1_time = 0;
    public static long best_search_stage_1_time = 0;
    public static long total_search_stage_2_time = 0;
    public static long best_search_stage_2_time = 0;

    public static int tabuTenure = 30; // denote the number of iterations for which a solution is tabu
    public static int neighborhoodSize = 30; // denote the size of the neighborhood to generate during each iteration, in each
    public static int shakeInterval = 4000; // denote the number of iterations between two shaking phases
    public static int shakeRecovery = 300; // denote the number of iterations to recover from a shaking phase\
    public static int stage2TotalTime = 20*60*1000; // denote the total time for stage 2 search
    public static int shakeRange = 50;
    public static int splitTabuTenure = 15;

    public static int randomSeed = 1;
    public static float maxSolverTime = 2; // in seconds  solver time limit

}
