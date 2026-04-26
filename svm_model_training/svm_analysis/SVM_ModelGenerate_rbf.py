from sklearn.svm import SVC
from sklearn.linear_model import Lasso
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import m2cgen as m2c
import pandas as pd
import numpy as np
import os
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import json

plt.rcParams['font.family'] = 'SimHei'  # 或其他支持中文的字体

# 输出目录（避免覆盖根目录原始实验产物）
output_dir = 'svm_analysis'
os.makedirs(output_dir, exist_ok=True)

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# 设置全局日志文件路径并初始化日志记录
log_file_path = os.path.join(output_dir, 'rbf_all_results_log.txt')
file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' 表示覆盖文件，使用 'a' 表示追加
logger.addHandler(file_handler)

# 数据归一化的小工具
scaler = MinMaxScaler()

# 文件路径


csv_file_path="data/training_2orientations.csv"

# csv_file_path="../stage3data/features0.5尺寸大规模milp标记.csv"

loadmaster = pd.read_csv(csv_file_path, encoding='utf-8')

#
# # 对前十三列去重
# loadmaster = loadmaster.drop_duplicates(subset=loadmaster.columns[:13])
#
# print(loadmaster.shape)  # 打印去重后的数据形状

# loadmaster = loadmaster.iloc[:1000]
# loadmaster = loadmaster.iloc[:, :-1]  # 去掉最后一列


# 去重操作
loadmaster = loadmaster.drop_duplicates(subset=[
    'sku_counts', 'total_skuvolume', 'sku_average_volume',
    'sku_length_var', 'sku_width_var', 'sku_height_var',
    'aspect_ratio_var', 'vehicle_capacity', 'sku_length_avg',
    'sku_width_avg', 'sku_height_avg', 'max_asr', 'vehicle_length',
    'vehicle_width', 'vehicle_height', 'spare_capacity', 'sku_concentration'
])
# 在去重操作之后添加新特征
# loadmaster['volume_ratio'] = loadmaster['total_skuvolume'] / loadmaster['vehicle_capacity']
# 重新设置索引
loadmaster = loadmaster.reset_index(drop=True)


print(loadmaster.info())
print(loadmaster.shape)


# 计算容积率
loadmaster['volume_ratio'] = loadmaster['total_skuvolume'] / loadmaster['vehicle_capacity']
# 创建容积率和 if_loaded 平均值的图形
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
#


# loadmaster = loadmaster.query('0.6 <= volume_ratio <= 0.9')



# 图 1: 不同容积率分布下 if_loaded 的平均值
bins = pd.cut(loadmaster['volume_ratio'], bins=10)  # 分成10个区间
average_if_loaded = loadmaster.groupby(bins)['if_loaded'].mean()  # 计算每个区间内 if_loaded 的平均值

sns.barplot(x=average_if_loaded.index.astype(str), y=average_if_loaded.values, ax=axs[0])
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
axs[0].set_xlabel('容积率区间')
axs[0].set_ylabel('if_loaded 的平均值')
axs[0].set_title('不同容积率分布下 if_loaded 的平均值')

# 图 2: 容积率的分布情况
sns.histplot(loadmaster['volume_ratio'], bins=30, kde=True, ax=axs[1])  # 添加密度曲线
axs[1].set_xlabel('容积率')
axs[1].set_ylabel('频率')
axs[1].set_title('容积率分布情况')
global_mean = loadmaster['if_loaded'].mean()
logger.info(f"数据集: 1, 类型比例: 1")
logger.info("全局 if_loaded 的平均值:")
logger.info(f"{global_mean}\n")

# 计算并打印全局容积率的平均值
global_mean = (loadmaster['total_skuvolume'].mean()) / (loadmaster['vehicle_capacity'].mean())
global_max = (loadmaster['total_skuvolume'].max()) / (loadmaster['vehicle_capacity'].mean())
global_min = (loadmaster['total_skuvolume'].min()) / (loadmaster['vehicle_capacity'].mean())
logger.info("全局容积率的平均值、最大值、最小值:")
logger.info(f"{global_mean}, {global_max}, {global_min}\n")
plt.tight_layout()  # 调整布局
# ... existing code ...
logger.info("全局容积率的平均值、最大值、最小值:")
logger.info(f"{global_mean}, {global_max}, {global_min}\n")
plt.tight_layout()  # 调整布局
plt.savefig(os.path.join(output_dir, 'volume_ratio_analysis_rbf.png'))  # 保存图形
plt.close(fig)
# plt.show()


loadmaster.drop('volume_ratio', axis=1, inplace=True)
# 提取特征和目标变量

X0, y = loadmaster.drop(["if_loaded", "orderid", '发车号','aspect_ratio_var',
                            'sku_max_volume', 'sku_min_volume','total_skuvolume','vehicle_capacity',
                            'sku_std_volume', 'min_asr', 'std_asr','h_dev_l_avg',
                            'h_dev_l_min', 'h_dev_l_max', 'h_dev_l_std', 'w_dev_h_avg', 'w_dev_h_min',
                            'w_dev_h_max', 'w_dev_h_std', 'w_dev_l_avg', 'w_dev_l_min', 'w_dev_l_max',
                            'w_dev_l_std', 'lh_to_vehicle_lh_avg', 'lh_to_vehicle_lh_min',
                            'lh_to_vehicle_lh_max', 'lh_to_vehicle_lh_std', 'wh_to_vehicle_wh_avg',
                            'wh_to_vehicle_wh_min', 'wh_to_vehicle_wh_max', 'wh_to_vehicle_wh_std',
                             'lh_to_vehicle_lh_total', 'wh_to_vehicle_wh_total',], axis=1), loadmaster['if_loaded']



print("features:")
print(X0.info())
print('')

# 对特征进行归一化

X = scaler.fit_transform(X0)


# 按照3:1划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# 创建SVM分类器对象，其中参数已经是svmParamTune调整出的最优结果
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', class_weight={1: 1}, probability=True, random_state=42)

# 训练SVM模型
svm_classifier.fit(X_train, y_train)


feature_names = X0.columns.tolist()
bias = svm_classifier.intercept_

# RBF 核没有 closed-form 的线性权重（coef_ 不存在）
print("closed-form: ")
print("RBF kernel: no linear coef_ available.")
print(bias[0])



# 在测试集上进行预测
y_pred_svm = svm_classifier.predict(X_test)




# 计算并打印准确率、精确率和召回率
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
logger.info(f"SVM Test Accuracy: {accuracy_svm}")
logger.info(f"SVM Test Precision: {precision_svm}")
logger.info(f"SVM Test Recall: {recall_svm}")

# 计算并打印混淆矩阵
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
logger.info("SVM Test Confusion Matrix:")
logger.info(f"{conf_matrix_svm}\n")

# 绘制并保存混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (RBF SVM)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_rbf.png'))
plt.close()


# 计算ROC AUC
y_scores = svm_classifier.predict_proba(X_test)[:, 1]  # 获取属于正类的概率
roc_auc = roc_auc_score(y_test, y_scores)
logger.info(f"SVM Test ROC AUC: {roc_auc}")

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 计算 FPR = 0.01 时的 TPR（线性插值）
target_fpr = 0.01
if target_fpr <= fpr[0]:
    tpr_at_001 = float(tpr[0])
elif target_fpr >= fpr[-1]:
    tpr_at_001 = float(tpr[-1])
else:
    tpr_at_001 = float(np.interp(target_fpr, fpr, tpr))
logger.info(f"TPR at FPR=0.01 (interp): {tpr_at_001}")

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'roc_curve_rbf.png'))
plt.close()


# RBF 核不支持用 coef_ 做“特征重要性”条形图，这里跳过

# 关闭文件处理器
logger.removeHandler(file_handler)
file_handler.close()






# 准备要保存的数据
model_data = {
    "bias": bias.tolist(),  # 将偏置转换为列表
    "data_min": scaler.data_min_.tolist(),
    "data_max": scaler.data_max_.tolist()
}

# 输出权重和偏置到 JSON 文件
with open(os.path.join(output_dir, 'rbf_scaler_bias.json'), 'w') as json_file:
    json.dump(model_data, json_file, indent=4)  # 使用 json.dump 写入文件，设置 indent=4 使格式更加美观

print("偏置与Scaler已保存到 svm_analysis/rbf_scaler_bias.json 文件中。")


# 确保所有 matplotlib 资源被释放
plt.close('all')
sys.exit(0)  # 确保程序退出




