package org.example;

import com.gurobi.gurobi.*;

import java.util.ArrayList;
import java.util.List;

public class MILP_Loading_2orientations_sequence {
    public static int MILP_single_box(List<Double> L_Packages,
                                      List<Double> parcel_p,
                                      List<Double> parcel_q,
                                      List<Double> parcel_r,
                                      List<Integer> parcel_fragility,
                                      List<Integer> sequence,
                                      List<List<Double>> myPackages,
                                      int boxNumbers){

        try {
//            System.out.println("sequence: "+sequence);
            // 定义模型
            GRBEnv env = new GRBEnv(true);
            env.set("OutputFlag", "0"); // 设置输出参数为0，禁止输出详细信息
            env.start();
            GRBModel model = new GRBModel(env);
            // 设置时间上限为 60 秒
            model.set(GRB.DoubleParam.TimeLimit, 10.0);
            model.set(GRB.IntParam.PoolSearchMode,1);
            model.set(GRB.IntParam.PoolSolutions, 1);
            // 获取包裹和箱子的数量
            int numParcels = parcel_p.size();
            int numPackages = myPackages.size();

            // 定义最大长度、宽度、高度
            double L_max = 0.0, W_max = 0.0, H_max = 0.0;
            for (List<Double> pkg : myPackages) {
                L_max = Math.max(L_max, pkg.get(2));
                W_max = Math.max(W_max, pkg.get(3));
                H_max = Math.max(H_max, pkg.get(4));
            }

            // 使用 List 来定义变量
            List<List<GRBVar>> s = new ArrayList<>();
            for (int i = 0; i < numParcels; ++i) {
                List<GRBVar> sRow = new ArrayList<>();
                for (int j = 0; j < numPackages; ++j) {
                    sRow.add(model.addVar(0, 1, 0, GRB.BINARY, "s_" + i + "_" + j));
                }
                s.add(sRow);
            }

            List<List<List<GRBVar>>> t = new ArrayList<>();
            for (int i = 0; i < numParcels; ++i) {
                List<List<GRBVar>> tRow = new ArrayList<>();
                for (int o1 = 0; o1 < 3; ++o1) {
                    List<GRBVar> tCol = new ArrayList<>();
                    for (int o2 = 0; o2 < 3; ++o2) {
                        tCol.add(model.addVar(0, 1, 0, GRB.BINARY, "t_" + i + "_" + o1 + "_" + o2));
                    }
                    tRow.add(tCol);
                }
                t.add(tRow);
            }

            List<List<GRBVar>> axp = new ArrayList<>();
            List<List<GRBVar>> ayp = new ArrayList<>();
            List<List<GRBVar>> azp = new ArrayList<>();
            // A_z_ik = 1 表示 z_i = z_k_r
            // delta_z_ik   表示 z_i - z_k_r
            // u_z_ik 表示zi - zk_r的绝对值
            // B_z_ik = 1 表示 i真的放在了k上面
            // o_i_k =0表示 i，k之间在xy投影上有重合
            // o_i_k_yz =0 表示 ik之间在yz投影上有重合
            // block z ik 表示i被k阻挡

            List<List<GRBVar>> Azp = new ArrayList<>();
            List<List<GRBVar>> deltazp = new ArrayList<>();
            List<List<GRBVar>> uzp = new ArrayList<>();
            List<List<GRBVar>> Bzp = new ArrayList<>();
            List<List<GRBVar>> oik = new ArrayList<>();
            List<List<GRBVar>> oik_yz = new ArrayList<>();
            List<List<GRBVar>> blockzp = new ArrayList<>();
            List<List<GRBVar>> blockxp= new ArrayList<>();

            for (int i = 0; i < numParcels; ++i) {
                List<GRBVar> axpRow = new ArrayList<>();
                List<GRBVar> aypRow = new ArrayList<>();
                List<GRBVar> azpRow = new ArrayList<>();
                List<GRBVar> AzpRow = new ArrayList<>();
                List<GRBVar> deltazpRow = new ArrayList<>();
                List<GRBVar> uzpRow = new ArrayList<>();
                List<GRBVar> BzpRow = new ArrayList<>();
                List<GRBVar> oikRow = new ArrayList<>();
                List<GRBVar> oik_yzRow = new ArrayList<>();
                List<GRBVar> blockzpRow = new ArrayList<>();
                List<GRBVar> blockxpRow = new ArrayList<>();


                for (int k = 0; k < numParcels; ++k) {
                    axpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "xp_" + i + "_" + k));
                    aypRow.add(model.addVar(0, 1, 0, GRB.BINARY, "yp_" + i + "_" + k));
                    azpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "zp_" + i + "_" + k));
                    AzpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "Azp_" + i + "_" + k));
                    deltazpRow.add(model.addVar(-H_max, H_max, 0, GRB.CONTINUOUS, "deltazp_" + i + "_" + k));
                    uzpRow.add(model.addVar(0, H_max, 0, GRB.CONTINUOUS, "uzp_" + i + "_" + k));
                    BzpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "Bzp_" + i + "_" + k));
                    oikRow.add(model.addVar(0, 1, 0, GRB.BINARY, "oik_" + i + "_" + k));
                    oik_yzRow.add(model.addVar(0, 1, 0, GRB.BINARY, "oik_yz_" + i + "_" + k));
                    blockzpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "blockzp_" + i + "_" + k));
                    blockxpRow.add(model.addVar(0, 1, 0, GRB.BINARY, "blockxp_" + i + "_" + k));
                }
                axp.add(axpRow);
                ayp.add(aypRow);
                azp.add(azpRow);
                Azp.add(AzpRow);
                deltazp.add(deltazpRow);
                uzp.add(uzpRow);
                Bzp.add(BzpRow);
                oik.add(oikRow);
                oik_yz.add(oik_yzRow);
                blockzp.add(blockzpRow);
                blockxp.add(blockxpRow);

            }


            List<GRBVar> x = new ArrayList<>();
            List<GRBVar> y = new ArrayList<>();
            List<GRBVar> z = new ArrayList<>();
            List<GRBVar> xr = new ArrayList<>();
            List<GRBVar> yr = new ArrayList<>();
            List<GRBVar> zr = new ArrayList<>();
            for (int i = 0; i < numParcels; ++i) {
                x.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "x_" + i));
                y.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "y_" + i));
                z.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "z_" + i));
                xr.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "xr_" + i));
                yr.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "yr_" + i));
                zr.add(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "zr_" + i));
            }

            List<GRBVar> n = new ArrayList<>();
            for (int j = 0; j < numPackages; ++j) {
                n.add(model.addVar(0, 1, 0, GRB.BINARY, "n_" + j));
            }

            // 添加约束
            for (int i = 0; i < numParcels; ++i) {
//                GRBLinExpr expr1 = new GRBLinExpr();
//                expr1.addTerm(1.0, xr.get(i));
//                expr1.addTerm(-1.0, x.get(i));
//                expr1.addTerm(-parcel_p.get(i) / L_max,t.get(i).get(0).get(0));
//                expr1.addTerm(-parcel_q.get(i) / L_max,t.get(i).get(0).get(1));
//
//                model.addConstr(expr1, GRB.EQUAL, 0, "c2_" + i);
//
//                GRBLinExpr expr2 = new GRBLinExpr();
//                expr2.addTerm(1.0, yr.get(i));
//                expr2.addTerm(-1.0, y.get(i));
//                expr2.addTerm(-parcel_p.get(i) / W_max,t.get(i).get(1).get(0));
//                expr2.addTerm(-parcel_q.get(i) / W_max,t.get(i).get(1).get(1));
//
//                model.addConstr(expr2, GRB.EQUAL, 0, "c3_" + i);
//
//                GRBLinExpr expr3 = new GRBLinExpr();
//                expr3.addTerm(1.0, zr.get(i));
//                expr3.addTerm(-1.0, z.get(i));
//
//                expr3.addConstant(-parcel_r.get(i) / H_max);
//                model.addConstr(expr3, GRB.EQUAL, 0, "c4_" + i);


                GRBLinExpr expr1 = new GRBLinExpr();
                expr1.addTerm(1.0, xr.get(i));
                expr1.addTerm(-1.0, x.get(i));
                expr1.addTerm(-parcel_p.get(i) / L_max,t.get(i).get(0).get(0));
                expr1.addTerm(-parcel_q.get(i) / L_max,t.get(i).get(0).get(1));
                expr1.addTerm(-parcel_r.get(i) / L_max,t.get(i).get(0).get(2));
                model.addConstr(expr1, GRB.EQUAL, 0, "c2_" + i);

                GRBLinExpr expr2 = new GRBLinExpr();
                expr2.addTerm(1.0, yr.get(i));
                expr2.addTerm(-1.0, y.get(i));
                expr2.addTerm(-parcel_p.get(i) / W_max,t.get(i).get(1).get(0));
                expr2.addTerm(-parcel_q.get(i) / W_max,t.get(i).get(1).get(1));
                expr2.addTerm(-parcel_r.get(i) / W_max,t.get(i).get(1).get(2));
                model.addConstr(expr2, GRB.EQUAL, 0, "c3_" + i);

                GRBLinExpr expr3 = new GRBLinExpr();
                expr3.addTerm(1.0, zr.get(i));
                expr3.addTerm(-1.0, z.get(i));
                expr3.addTerm(-parcel_p.get(i) / H_max,t.get(i).get(2).get(0));
                expr3.addTerm(-parcel_q.get(i) / H_max,t.get(i).get(2).get(1));
                expr3.addTerm(-parcel_r.get(i) / H_max,t.get(i).get(2).get(2));
                model.addConstr(expr3, GRB.EQUAL, 0, "c4_" + i);


                GRBLinExpr sumXR = new GRBLinExpr();
                GRBLinExpr sumYR = new GRBLinExpr();
                GRBLinExpr sumZR = new GRBLinExpr();
                for (int j = 0; j < numPackages; ++j) {
                    sumXR.addTerm(myPackages.get(j).get(2) / L_max, s.get(i).get(j));
                    sumYR.addTerm(myPackages.get(j).get(3) / W_max, s.get(i).get(j));
                    sumZR.addTerm(myPackages.get(j).get(4) / H_max, s.get(i).get(j));
                }
                model.addConstr(xr.get(i), GRB.LESS_EQUAL, sumXR, "c5_" + i);
                model.addConstr(yr.get(i), GRB.LESS_EQUAL, sumYR, "c6_" + i);
                model.addConstr(zr.get(i), GRB.LESS_EQUAL, sumZR, "c7_" + i);
            }

            for (int i = 0; i < numParcels; ++i) {
                for (int k = 0; k < numParcels; ++k) {
                    for (int j = 0; j < numPackages; ++j) {
                        if (i < k) {
                            // 创建 GRBLinExpr 对象
                            GRBLinExpr leftExpr = new GRBLinExpr();
                            leftExpr.addTerm(1.0, axp.get(i).get(k)); // xp.get(i).get(k)
                            leftExpr.addTerm(1.0, axp.get(k).get(i)); // xp.get(k).get(i)
                            leftExpr.addTerm(1.0, ayp.get(i).get(k)); // yp.get(i).get(k)
                            leftExpr.addTerm(1.0, ayp.get(k).get(i)); // yp.get(k).get(i)
                            leftExpr.addTerm(1.0, azp.get(i).get(k)); // zp.get(i).get(k)
                            leftExpr.addTerm(1.0, azp.get(k).get(i)); // zp.get(k).get(i)

                            GRBLinExpr rightExpr = new GRBLinExpr();
                            rightExpr.addTerm(1.0, s.get(i).get(j)); // s.get(i).get(j)
                            rightExpr.addTerm(1.0, s.get(k).get(j)); // s.get(k).get(j)
                            rightExpr.addConstant(-1.0); // -1

                            model.addConstr(leftExpr, GRB.GREATER_EQUAL, rightExpr, "c8_"+i + "_" + k + "_" + j);                        }
                    }
                }
            }

            // Pack all items
            for (int i = 0; i < numParcels; ++i) {
                GRBLinExpr sumS = new GRBLinExpr();
                for (int j = 0; j < numPackages; ++j) {
                    sumS.addTerm(1.0, s.get(i).get(j));
                }
                model.addConstr(sumS,GRB.EQUAL,1,"c9_"+i);
            }

            for (int i = 0; i < numParcels; ++i) {
                for (int j = 0; j < numPackages; ++j) {
                    model.addConstr(s.get(i).get(j), GRB.LESS_EQUAL, n.get(j), "c10_" + i + "_" + j);
                }
            }

            // Select one orientation
            for (int i = 0; i < numParcels; ++i) {
                for (int d = 0; d < 3; ++d) {
                    GRBLinExpr sumT1 = new GRBLinExpr();
                    GRBLinExpr sumT2 = new GRBLinExpr();
                    for (int c = 0; c < 3; ++c) {
                        sumT1.addTerm(1.0, t.get(i).get(c).get(d));
                        sumT2.addTerm(1.0, t.get(i).get(d).get(c));
                    }
                    model.addConstr(sumT1, GRB.EQUAL, 1, "c11_" + i + "_" + d);
                    model.addConstr(sumT2, GRB.EQUAL, 1, "c12_" + i + "_" + d);
                }
            }


            // Logic constraints
            for (int i = 0; i < numParcels; ++i) {
                for (int k = 0; k < numParcels; ++k) {
                    GRBLinExpr expr1 = new GRBLinExpr();
                    expr1.addTerm(1.0, x.get(i)); // x.get(i)
                    expr1.addConstant(1.0); // +1
                    expr1.addTerm(-1.0, axp.get(i).get(k)); // -xp.get(i).get(k)
                    model.addConstr(xr.get(k), GRB.LESS_EQUAL, expr1, "c_13_"+i+"_"+k);

                    GRBLinExpr expr2 = new GRBLinExpr();
                    expr2.addTerm(1.0, xr.get(k)); // xr.get(k)
                    expr2.addTerm(1.0, axp.get(i).get(k)); // +xp.get(i).get(k)
                    expr2.addConstant(-1.0/L_max); // -1/L_max
                    model.addConstr(x.get(i), GRB.LESS_EQUAL, expr2, "c_14_"+i+"_"+k);

                    GRBLinExpr expr3 = new GRBLinExpr();
                    expr3.addTerm(1.0, y.get(i)); // y.get(i)
                    expr3.addConstant(1.0); // +1
                    expr3.addTerm(-1.0, ayp.get(i).get(k)); // -yp.get(i).get(k)
                    model.addConstr(yr.get(k), GRB.LESS_EQUAL, expr3, "c_15_"+i+"_"+k);

                    GRBLinExpr expr4 = new GRBLinExpr();
                    expr4.addTerm(1.0, yr.get(k)); // yr.get(k)
                    expr4.addTerm(1.0, ayp.get(i).get(k)); // +yp.get(i).get(k)
                    expr4.addConstant(-1.0/W_max); // -1/W_max
                    model.addConstr(y.get(i), GRB.LESS_EQUAL, expr4, "c_16_"+i+"_"+k);

                    GRBLinExpr expr5 = new GRBLinExpr();
                    expr5.addTerm(1.0, z.get(i)); // z.get(i)
                    expr5.addConstant(1.0); // +1
                    expr5.addTerm(-1.0, azp.get(i).get(k)); // -zp.get(i).get(k)
                    model.addConstr(zr.get(k), GRB.LESS_EQUAL, expr5, "c_17_"+i+"_"+k);
                }
            }

            // Symmetry breaking constraints
            for (int i = 0; i < 1; ++i) {
                GRBLinExpr sumL = new GRBLinExpr();
                GRBLinExpr sumW = new GRBLinExpr();
                GRBLinExpr sumH = new GRBLinExpr();
                for (int j = 0; j < numPackages; ++j) {
                    sumL.addTerm(myPackages.get(j).get(2) / L_max / 2, s.get(i).get(j));
                    sumW.addTerm(myPackages.get(j).get(3) / W_max / 2, s.get(i).get(j));
                    sumH.addTerm(myPackages.get(j).get(4) / H_max / 2, s.get(i).get(j));
                }

                // 约束 1: x.get(i) + 0.5 * (t.get(i).get(0).get(0) * parcel_p.get(i) / L_max + t.get(i).get(0).get(1) * parcel_q.get(i) / L_max + t.get(i).get(0).get(2) * parcel_r.get(i) / L_max) <= sumL
                GRBLinExpr expr1 = new GRBLinExpr();
                expr1.addTerm(1.0, x.get(i)); // x.get(i)
                expr1.addTerm(0.5 * parcel_p.get(i) / L_max, t.get(i).get(0).get(0)); // 0.5 * t.get(i).get(0).get(0) * parcel_p.get(i) / L_max
                expr1.addTerm(0.5 * parcel_q.get(i) / L_max, t.get(i).get(0).get(1)); // 0.5 * t.get(i).get(0).get(1) * parcel_q.get(i) / L_max
                expr1.addTerm(0.5 * parcel_r.get(i) / L_max, t.get(i).get(0).get(2)); // 0.5 * t.get(i).get(0).get(2) * parcel_r.get(i) / L_max
                model.addConstr(expr1, GRB.LESS_EQUAL, sumL, "c_24_"+i);

// 约束 2: y.get(i) + 0.5 * (t.get(i).get(1).get(0) * parcel_p.get(i) / W_max + t.get(i).get(1).get(1) * parcel_q.get(i) / W_max + t.get(i).get(1).get(2) * parcel_r.get(i) / W_max) <= sumW
                GRBLinExpr expr2 = new GRBLinExpr();
                expr2.addTerm(1.0, y.get(i)); // y.get(i)
                expr2.addTerm(0.5 * parcel_p.get(i) / W_max, t.get(i).get(1).get(0)); // 0.5 * t.get(i).get(1).get(0) * parcel_p.get(i) / W_max
                expr2.addTerm(0.5 * parcel_q.get(i) / W_max, t.get(i).get(1).get(1)); // 0.5 * t.get(i).get(1).get(1) * parcel_q.get(i) / W_max
                expr2.addTerm(0.5 * parcel_r.get(i) / W_max, t.get(i).get(1).get(2)); // 0.5 * t.get(i).get(1).get(2) * parcel_r.get(i) / W_max
                model.addConstr(expr2, GRB.LESS_EQUAL, sumW, "c_25_"+i);

// 约束 3: z.get(i) + 0.5 * (t.get(i).get(2).get(0) * parcel_p.get(i) / H_max + t.get(i).get(2).get(1) * parcel_q.get(i) / H_max + t.get(i).get(2).get(2) * parcel_r.get(i) / H_max) <= sumH
                GRBLinExpr expr3 = new GRBLinExpr();
                expr3.addTerm(1.0, z.get(i)); // z.get(i)
                expr3.addTerm(0.5 * parcel_p.get(i) / H_max, t.get(i).get(2).get(0)); // 0.5 * t.get(i).get(2).get(0) * parcel_p.get(i) / H_max
                expr3.addTerm(0.5 * parcel_q.get(i) / H_max, t.get(i).get(2).get(1)); // 0.5 * t.get(i).get(2).get(1) * parcel_q.get(i) / H_max
                expr3.addTerm(0.5 * parcel_r.get(i) / H_max, t.get(i).get(2).get(2)); // 0.5 * t.get(i).get(2).get(2) * parcel_r.get(i) / H_max
                model.addConstr(expr3, GRB.LESS_EQUAL, sumH, "c_26_"+i);
            }

            // Limit box
            GRBLinExpr sumN = new GRBLinExpr();
            for (int j = 0; j < numPackages; ++j) {
                sumN.addTerm(1.0, n.get(j));
            }
            model.addConstr(sumN, GRB.LESS_EQUAL, boxNumbers, "c_23");

            // t_i02 = t_i12 = t_i20 = t_i21 = 0；
            // t_i22 = 1;
            for (int i = 0; i < numParcels; ++i) {
                model.addConstr(t.get(i).get(0).get(2), GRB.EQUAL, 0, "c_27_"+i);
                model.addConstr(t.get(i).get(1).get(2), GRB.EQUAL, 0, "c_28_"+i);
                model.addConstr(t.get(i).get(2).get(0), GRB.EQUAL, 0, "c_29_"+i);
                model.addConstr(t.get(i).get(2).get(1), GRB.EQUAL, 0, "c_30_"+i);
                model.addConstr(t.get(i).get(2).get(2), GRB.EQUAL, 1, "c_31_"+i);
            }


            int SequenceMax = parcel_p.size() +1;
//            System.out.println("SequenceMax: "+SequenceMax);

            // sequence constraints
            for(int i=0;i<numParcels;i++){
                for(int k=0;k<numParcels;k++){
                    if(i==k){
                        continue; //避免自我堆叠
                    }

                    GRBLinExpr expr_azp = new GRBLinExpr();
                    expr_azp.addTerm(1.0, zr.get(k)); // zr.get(k)
                    expr_azp.addTerm(1.0, azp.get(i).get(k)); // +zp.get(i).get(k)
                    expr_azp.addConstant(-1.0/H_max); // -1/H_max
                    model.addConstr(z.get(i), GRB.LESS_EQUAL, expr_azp, "c_azp_"+i+"_"+k);


                    // oik ≤ xp  ik + xp  ki + yp  ik + yp  ki ≤ 2oik, ∀i, k
                    GRBLinExpr expr_oik = new GRBLinExpr();
                    expr_oik.addTerm(1,axp.get(i).get(k));
                    expr_oik.addTerm(1,axp.get(k).get(i));
                    expr_oik.addTerm(1,ayp.get(i).get(k));
                    expr_oik.addTerm(1,ayp.get(k).get(i));
                    model.addConstr(expr_oik, GRB.GREATER_EQUAL, oik.get(i).get(k), "c_32_"+i+"_"+k);
                    GRBLinExpr expr_oik_2 = new GRBLinExpr();
                    expr_oik_2.addTerm(2,oik.get(i).get(k));
                    model.addConstr(expr_oik_2,GRB.GREATER_EQUAL,expr_oik,"c_33_"+i+"_"+k);

                    // 上下不能阻挡  block z ik 表示i被k阻挡
                    // 1-blockz ik <= (1-azp[k][i])+oik[k][i] <= 2*(1-blockz ik)
                    GRBLinExpr expr1 = new GRBLinExpr();
                    expr1.addConstant(1);
                    expr1.addTerm(-1.0, blockzp.get(i).get(k)); // x.get(i)
                    GRBLinExpr expr2 = new GRBLinExpr();
                    expr2.addConstant(1);
                    expr2.addTerm(-1,azp.get(k).get(i));
                    expr2.addTerm(1,oik.get(k).get(i));
                    GRBLinExpr expr3 = new GRBLinExpr();
                    expr3.addTerm(-2,blockzp.get(i).get(k));
                    expr3.addConstant(2);
                    model.addConstr(expr1, GRB.LESS_EQUAL, expr2, "c_34_"+i+"_"+k);
                    model.addConstr(expr2, GRB.LESS_EQUAL, expr3, "c_35_"+i+"_"+k);


                    // o_yz_ik ≤ zp  ik + zp  ki + yp  ik + yp  ki ≤ 2oi_yz_k, ∀i, k
                    GRBLinExpr expr4 = new GRBLinExpr();
                    expr4.addTerm(1,azp.get(i).get(k));
                    expr4.addTerm(1,azp.get(k).get(i));
                    expr4.addTerm(1,ayp.get(i).get(k));
                    expr4.addTerm(1,ayp.get(k).get(i));
                    model.addConstr(expr4, GRB.GREATER_EQUAL, oik_yz.get(i).get(k), "c_36_"+i+"_"+k);
                    GRBLinExpr expr5 = new GRBLinExpr();
                    expr5.addTerm(2,oik_yz.get(i).get(k));
                    model.addConstr(expr5,GRB.GREATER_EQUAL,expr4,"c_37_"+i+"_"+k);
                    // x方向不能阻挡
                    // 1-blockx ik <= (1-axp[k][i])+oxik[k][i] <= 2*(1-blockx ik)
                    GRBLinExpr expr6 = new GRBLinExpr();
                    expr6.addConstant(1);
                    expr6.addTerm(-1.0, blockxp.get(i).get(k)); // x.get(i)
                    GRBLinExpr expr7 = new GRBLinExpr();
                    expr7.addConstant(1);
                    expr7.addTerm(-1,axp.get(k).get(i));
                    expr7.addTerm(1,oik_yz.get(k).get(i));
                    GRBLinExpr expr8 = new GRBLinExpr();
                    expr8.addConstant(2);
                    expr8.addTerm(-2,blockxp.get(i).get(k));
                    model.addConstr(expr6, GRB.LESS_EQUAL, expr7, "c_38_"+i+"_"+k);
                    model.addConstr(expr7, GRB.LESS_EQUAL, expr8, "c_39_"+i+"_"+k);

                    // if block ik == 1,  then sequence_i >= sequence_k
                    // (1-blockz_ik) * M +seuqence_i >= sequence_k
                    // (1-blockx_ik) * M +seuqence_i >= sequence_k
                    GRBLinExpr sequence_z_Expr_1 = new GRBLinExpr();
                    sequence_z_Expr_1.addTerm(-SequenceMax, blockzp.get(i).get(k));
                    sequence_z_Expr_1.addConstant(SequenceMax);
                    sequence_z_Expr_1.addConstant(sequence.get(i));
                    GRBLinExpr sequence_z_Expr_2 = new GRBLinExpr();
                    sequence_z_Expr_2.addConstant(sequence.get(k));
                    model.addConstr(sequence_z_Expr_1, GRB.GREATER_EQUAL, sequence_z_Expr_2, "c_40_"+i+"_"+k);

                    GRBLinExpr sequence_x_Expr_1 = new GRBLinExpr();
                    sequence_x_Expr_1.addTerm(-SequenceMax, blockxp.get(i).get(k));
                    sequence_x_Expr_1.addConstant(SequenceMax);
                    sequence_x_Expr_1.addConstant(sequence.get(i));
                    GRBLinExpr sequence_x_Expr_2 = new GRBLinExpr();
                    sequence_x_Expr_2.addConstant(sequence.get(k));
                    model.addConstr(sequence_x_Expr_1, GRB.GREATER_EQUAL, sequence_x_Expr_2, "c_41_"+i+"_"+k);




                    // 前后不能阻挡
                }
            }

//            // fragility constraint
//            double small_epsilon = 1e-6;
//            double big_epsilon = 1e-4;
//            for (int i = 0; i < numParcels; ++i) {
//                for (int k = 0; k < numParcels; ++k) {
//                    if(i ==k ){
//                        continue; //避免自我堆叠
//                    }
//
//
//
//                    // 1. uzp[i][k]  = |z[i] - zr[k]|
//                    GRBLinExpr expr1 = new GRBLinExpr();
//                    expr1.addTerm(1.0, z.get(i)); // z.get(i)
//                    expr1.addTerm(-1.0, zr.get(k)); // z.get(i)
//                    expr1.addConstant(2*H_max);
//                    expr1.addTerm(-2*H_max, azp.get(i).get(k));
//                    model.addConstr(uzp.get(i).get(k), GRB.LESS_EQUAL, expr1, "c_32_"+i+"_"+k);
//
//                    GRBLinExpr expr2 = new GRBLinExpr();
//                    expr2.addTerm(-1.0, z.get(i)); // z.get(i)
//                    expr2.addTerm(1.0, zr.get(k)); // z.get(i)
//                    expr2.addTerm(2*H_max, azp.get(i).get(k));
//                    model.addConstr(uzp.get(i).get(k), GRB.LESS_EQUAL, expr2, "c_33_"+i+"_"+k);
//
//                    // 2. uzp[i][k] ==0, 则Az_i_k =1, 否则Az_i_k =0
//
//                    GRBLinExpr expr2_5 = new GRBLinExpr();
//                    expr2_5.addConstant(1);
//                    expr2_5.addTerm(-1,Azp.get(i).get(k));
//                    model.addConstr(expr2_5, GRB.LESS_EQUAL, uzp.get(i).get(k), "c_34_"+i+"_"+k);
//                    GRBLinExpr expr3 = new GRBLinExpr();
//                    expr3.addConstant(H_max);
//                    expr3.addTerm(-H_max,Azp.get(i).get(k));
//                    model.addConstr(expr3, GRB.GREATER_EQUAL, uzp.get(i).get(k), "c_35_"+i+"_"+k);
//
//                    // 3.oik ≤ xp  ik + xp  ki + yp  ik + yp  ki ≤ 2oik, ∀i, k
//                    GRBLinExpr expr4 = new GRBLinExpr();
//                    expr4.addTerm(1,axp.get(i).get(k));
//                    expr4.addTerm(1,axp.get(k).get(i));
//                    expr4.addTerm(1,ayp.get(i).get(k));
//                    expr4.addTerm(1,ayp.get(k).get(i));
//                    model.addConstr(expr4, GRB.GREATER_EQUAL, oik.get(i).get(k), "c_36_"+i+"_"+k);
//                    GRBLinExpr expr5 = new GRBLinExpr();
//                    expr5.addTerm(2,oik.get(i).get(k));
//                    model.addConstr(expr5,GRB.GREATER_EQUAL,expr4,"c_37_"+i+"_"+k);
//
//                    //4. (1-B_ik)<=o_ik +1- A_ik <= 2(1-B_ik)
//                    GRBLinExpr expr6 = new GRBLinExpr();
//                    expr6.addConstant(1.0);
//                    expr6.addTerm(-1.0, Bzp.get(i).get(k));
//                    GRBLinExpr expr7 = new GRBLinExpr();
//                    expr7.addConstant(1);
//                    expr7.addTerm(-1,Azp.get(i).get(k));
//                    expr7.addTerm(1,oik.get(i).get(k));
//                    GRBLinExpr expr8 = new GRBLinExpr();
//                    expr8.addConstant(2.0);
//                    expr8.addTerm(-2.0, Bzp.get(i).get(k));
//                    model.addConstr(expr6, GRB.LESS_EQUAL, expr7, "c_38_"+i+"_"+k);
//                    model.addConstr(expr7, GRB.LESS_EQUAL, expr8, "c_39_"+i+"_"+k);
//
//
//                    // if Bik == 1,  then fragility_i >= fragility_k
//                    GRBLinExpr fragilityExpr_k = new GRBLinExpr();
//                    fragilityExpr_k.addTerm(parcel_fragility.get(k), Bzp.get(i).get(k));
//                    model.addConstr(parcel_fragility.get(i), GRB.GREATER_EQUAL, fragilityExpr_k, "c_40_"+i+"_"+k);
//                }
//            }



//             目标函数
            GRBLinExpr objUsedVolume = new GRBLinExpr();
            for (int j = 0; j < numPackages; ++j) {
                objUsedVolume.addTerm(myPackages.get(j).get(1), n.get(j));
            }
            for (int i = 0; i < numParcels; ++i) {
                objUsedVolume.addConstant(-parcel_p.get(i) * parcel_q.get(i) * parcel_r.get(i));
            }
            model.setObjective(objUsedVolume, GRB.MINIMIZE);

            // 优化模型
            model.optimize();

            // 检查是否有解
            int solCount = (int) model.get(GRB.IntAttr.SolCount);

            if (solCount > 0) {
//                System.out.println("=== Checking sequence constraints ===");
//                for (int i = 0; i < numParcels; ++i) {
//                    for (int k = 0; k < numParcels; ++k) {
//                        if (i == k) continue;
//                        System.out.println(sequence.get(i)+","+sequence.get(k));
//                        double blockz_ik = blockzp.get(i).get(k).get(GRB.DoubleAttr.X);
//                        double blockx_ik = blockxp.get(i).get(k).get(GRB.DoubleAttr.X);
//                        double seq_i = sequence.get(i);
//                        double seq_k = sequence.get(k);
//                        SequenceMax = parcel_p.size() + 1;
//
//                        // c_40: Z方向的sequence约束
//                        double lhs_z = (1 - blockz_ik) * SequenceMax + seq_i;
//                        double rhs_z = seq_k;
//                        boolean satisfied_z = lhs_z >= rhs_z - 1e-6;
//
//                        System.out.printf("c_40_%d_%d: lhs = %.2f, rhs = %.2f, satisfied = %s%n", i, k, lhs_z, rhs_z, satisfied_z);
//
//                        // c_41: X方向的sequence约束
//                        double lhs_x = (1 - blockx_ik) * SequenceMax + seq_i;
//                        double rhs_x = seq_k;
//                        boolean satisfied_x = lhs_x >= rhs_x - 1e-6;
//
//                        System.out.printf("c_41_%d_%d: lhs = %.2f, rhs = %.2f, satisfied = %s%n", i, k, lhs_x, rhs_x, satisfied_x);
//                    }
//                }
                // 当前解是可行解
                int status = (int) model.get(GRB.IntAttr.Status);
                if (status == GRB.Status.OPTIMAL || status == GRB.Status.TIME_LIMIT) {
////                    // 获取所有变量
//                    GRBVar[] vars = model.getVars();
//
//                    // 遍历所有变量并输出它们的值
//                    for (GRBVar var : vars) {
//                        System.out.println("变量名: " + var.get(GRB.StringAttr.VarName) + ", 值: " + var.get(GRB.DoubleAttr.X));
//                    }
                    return 1;  // 当前解可行（包括最优解和时间限制内的可行解）
                }
            }


            else {


                model.computeIIS();
                model.write("model.ilp");

                // 当前无可行解
                int status = (int) model.get(GRB.IntAttr.Status);
                if (status == GRB.Status.INFEASIBLE) {
                    return 0;  // 模型无解
                } else if (status == GRB.Status.TIME_LIMIT) {
                    return -1; // 时间限制内无可行解
                } else {
                    return -2; // 其他未知错误
                }
            }

            return -1;

        } catch (GRBException e) {
            System.err.println("Error code = " + e.getErrorCode());
            System.err.println(e.getMessage());
            return -1;
        } catch (Exception e) {
            System.err.println("Exception during optimization");
            return -1;
        }







    }

}
