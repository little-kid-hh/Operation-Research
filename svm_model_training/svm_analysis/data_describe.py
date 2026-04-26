import pandas as pd
from pathlib import Path


def main():
    # 以脚本所在目录为基准，定位到项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # 数据文件路径：项目根目录下的 data/training_2orientations.csv
    data_path = project_root / "data" / "training_2orientations.csv"

    # 读取数据
    df = pd.read_csv(data_path, encoding="utf-8")

    # 输出目录：当前脚本所在的 svm_analysis 目录
    output_dir = script_dir

    # 数值列描述性统计
    desc_numeric = df.describe()

    # 所有列（包括非数值）的统计
    desc_all = df.describe(include="all")

    # 导出结果到专门的分析目录
    desc_numeric.to_csv(output_dir / "training_2orientations_describe_numeric.csv", encoding="utf-8-sig")
    desc_all.to_csv(output_dir / "training_2orientations_describe_all.csv", encoding="utf-8-sig")

    # 根据 SVM_ModelGenerate.py 的特征选择规则导出特征名单
    drop_cols = [
        "if_loaded", "orderid", "发车号", "aspect_ratio_var",
        "sku_max_volume", "sku_min_volume", "total_skuvolume", "vehicle_capacity",
        "sku_std_volume", "min_asr", "std_asr", "h_dev_l_avg",
        "h_dev_l_min", "h_dev_l_max", "h_dev_l_std", "w_dev_h_avg",
        "w_dev_h_min", "w_dev_h_max", "w_dev_h_std", "w_dev_l_avg",
        "w_dev_l_min", "w_dev_l_max", "w_dev_l_std", "lh_to_vehicle_lh_avg",
        "lh_to_vehicle_lh_min", "lh_to_vehicle_lh_max", "lh_to_vehicle_lh_std",
        "wh_to_vehicle_wh_avg", "wh_to_vehicle_wh_min", "wh_to_vehicle_wh_max",
        "wh_to_vehicle_wh_std", "lh_to_vehicle_lh_total", "wh_to_vehicle_wh_total",
    ]

    # 仅保留真正进入 SVM 的特征列
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 保存特征名单（逐行列出）
    feature_list_path = output_dir / "svm_feature_list.txt"
    with feature_list_path.open("w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    print("描述性统计已生成：")
    print(f" - {output_dir / 'training_2orientations_describe_numeric.csv'}")
    print(f" - {output_dir / 'training_2orientations_describe_all.csv'}")
    print(f" - {feature_list_path}（SVM 训练特征名单）")


if __name__ == "__main__":
    main()

