import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def compute_tpr_at_fpr(target_fpr: float = 0.01) -> float:
    """
    复现 SVM_ModelGenerate.py 中的数据处理和模型训练流程，
    计算在指定 FPR (默认 0.01) 下的 TPR。
    """
    project_root = Path(__file__).resolve().parents[1]

    # 1. 读取数据，路径与原脚本保持一致
    csv_file_path = project_root / "data" / "training_2orientations.csv"
    df = pd.read_csv(csv_file_path, encoding="utf-8")

    # 2. 按原脚本的去重逻辑处理
    df = df.drop_duplicates(
        subset=[
            "sku_counts",
            "total_skuvolume",
            "sku_average_volume",
            "sku_length_var",
            "sku_width_var",
            "sku_height_var",
            "aspect_ratio_var",
            "vehicle_capacity",
            "sku_length_avg",
            "sku_width_avg",
            "sku_height_avg",
            "max_asr",
            "vehicle_length",
            "vehicle_width",
            "vehicle_height",
            "spare_capacity",
            "sku_concentration",
        ]
    ).reset_index(drop=True)

    # 3. 特征/标签拆分，列选择与 SVM_ModelGenerate.py 完全一致
    drop_cols = [
        "if_loaded",
        "orderid",
        "发车号",
        "aspect_ratio_var",
        "sku_max_volume",
        "sku_min_volume",
        "total_skuvolume",
        "vehicle_capacity",
        "sku_std_volume",
        "min_asr",
        "std_asr",
        "h_dev_l_avg",
        "h_dev_l_min",
        "h_dev_l_max",
        "h_dev_l_std",
        "w_dev_h_avg",
        "w_dev_h_min",
        "w_dev_h_max",
        "w_dev_h_std",
        "w_dev_l_avg",
        "w_dev_l_min",
        "w_dev_l_max",
        "w_dev_l_std",
        "lh_to_vehicle_lh_avg",
        "lh_to_vehicle_lh_min",
        "lh_to_vehicle_lh_max",
        "lh_to_vehicle_lh_std",
        "wh_to_vehicle_wh_avg",
        "wh_to_vehicle_wh_min",
        "wh_to_vehicle_wh_max",
        "wh_to_vehicle_wh_std",
        "lh_to_vehicle_lh_total",
        "wh_to_vehicle_wh_total",
    ]

    X0 = df.drop(drop_cols, axis=1)
    y = df["if_loaded"]

    # 4. 归一化 + 训练集/测试集划分
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5. 训练与原脚本相同的 SVM 模型
    svm_classifier = SVC(
        kernel="linear", C=10, class_weight={1: 1}, probability=True, random_state=42
    )
    svm_classifier.fit(X_train, y_train)

    # 6. 计算 ROC 曲线
    y_scores = svm_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    # 7. 在 FPR 上做线性插值，得到目标 FPR 对应的 TPR
    # 若 target_fpr 小于最小 FPR，则取第一个实际点；大于最大 FPR，则取最后一个实际点。
    if target_fpr <= fpr[0]:
        tpr_at_target = tpr[0]
    elif target_fpr >= fpr[-1]:
        tpr_at_target = tpr[-1]
    else:
        # 使用 numpy 插值
        tpr_at_target = float(np.interp(target_fpr, fpr, tpr))

    # 7.1 为了得到 Precision / Recall / 混淆矩阵，需要选择一个具体阈值。
    # 这里选择 FPR 最接近 target_fpr 的那个点对应的阈值。
    idx_closest = int(np.argmin(np.abs(fpr - target_fpr)))
    best_threshold = thresholds[idx_closest]

    # 根据该阈值生成预测标签
    y_pred_thresh = (y_scores >= best_threshold).astype(int)

    # 计算该点下的精确率、召回率和混淆矩阵
    precision_at_target = precision_score(y_test, y_pred_thresh)
    recall_at_target = recall_score(y_test, y_pred_thresh)
    conf_mat_at_target = confusion_matrix(y_test, y_pred_thresh)

    # 8. 绘制并保存 ROC 曲线（文件名与原始脚本区别开）
    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (reproduced)")
    plt.legend(loc="lower right")

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "svm_analysis" / "roc_curve_reproduced.png"
    plt.savefig(out_path)
    plt.close()

    # 9. 将 FPR=target_fpr 时的实验结果写入独立日志
    log_path = project_root / "svm_analysis" / "fpr_001_experiment_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Target FPR: {target_fpr}\n")
        f.write(f"Interpolated TPR at FPR={target_fpr}: {tpr_at_target}\n")
        f.write(f"Closest FPR in ROC: {fpr[idx_closest]}\n")
        f.write(f"Threshold at closest FPR: {best_threshold}\n\n")
        f.write(f"Precision at closest FPR: {precision_at_target}\n")
        f.write(f"Recall at closest FPR: {recall_at_target}\n")
        f.write("Confusion matrix at closest FPR (rows: true [0,1], cols: pred [0,1]):\n")
        f.write(f"{conf_mat_at_target}\n")

    # 同时返回 tpr_at_target 和 auc，方便主程序打印对比
    return tpr_at_target, float(auc)


if __name__ == "__main__":
    target_fpr = 0.01
    tpr_value, auc_value = compute_tpr_at_fpr(target_fpr)
    print(f"FPR = {target_fpr:.4f} 时的 TPR ≈ {tpr_value:.6f}")
    print(f"对应的 AUC ≈ {auc_value:.6f}")
    print("ROC 曲线已保存为 svm_analysis/roc_curve_reproduced.png")
    print("基于 FPR 最接近 0.01 的阈值的 Precision、Recall 和混淆矩阵已保存到 svm_analysis/fpr_001_experiment_log.txt")

