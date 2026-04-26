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

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# 设置全局日志文件路径并初始化日志记录
log_file_path = 'all_results_log.txt'
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
plt.savefig('volume_ratio_analysis.png')  # 保存图形
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
svm_classifier = SVC(kernel='linear', C=10, class_weight={1: 1}, probability=True, random_state=42)

# 训练SVM模型
svm_classifier.fit(X_train, y_train)


feature_names = X0.columns.tolist()
weights = svm_classifier.coef_
bias = svm_classifier.intercept_

# 打印权重和对应特征名称
print("closed-form: ")

for feature_name, weight in zip(feature_names, weights.flatten()):
    print(f"{feature_name}*{weight} +")
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


# 计算ROC AUC
y_scores = svm_classifier.predict_proba(X_test)[:, 1]  # 获取属于正类的概率
roc_auc = roc_auc_score(y_test, y_scores)
logger.info(f"SVM Test ROC AUC: {roc_auc}")

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()


# 各个特征的支持向量权重
svm_coefs = svm_classifier.coef_
feature_names = X0.columns

# 获取支持向量的系数和特征名称，并按照系数的绝对值降序排列
svm_coefs_abs = np.abs(svm_coefs[0])
sorted_indices = np.argsort(svm_coefs_abs)[::-1]  # 反向排序以获得降序
sorted_feature_names = feature_names[sorted_indices]
sorted_coefs = svm_coefs_abs[sorted_indices]

# 选择前 10 个最重要的特征
top_n = 10
top_feature_names = sorted_feature_names[:top_n][::-1]
top_coefs = sorted_coefs[:top_n][::-1]

# 绘制特征重要性图
plt.figure(figsize=(26, 12))
plt.barh(range(top_n), top_coefs, align='center',color = 'orange')
plt.yticks(range(top_n), top_feature_names, fontsize=26)
plt.xlabel('Coefficient Magnitude', fontsize=26)
plt.title('Top 10 Feature Importance in SVM Model (Sorted)', fontsize=26)

# 调整左侧边距
plt.subplots_adjust(left=0.4)  # 增加左侧边距，0.2 可以根据需要调整

# 保存图表
plt.savefig('feature_importance_exact_method.png')
plt.close()

# 关闭文件处理器
logger.removeHandler(file_handler)
file_handler.close()







# 准备要保存的数据
model_data = {
    "weights": weights.flatten().tolist(),  # 将权重转换为列表
    "bias": bias.tolist(),  # 将偏置转换为列表
    "data_min": scaler.data_min_.tolist(),
    "data_max": scaler.data_max_.tolist()
}

# 输出权重和偏置到 JSON 文件
with open('model/model_weights_bias_20w_exact_2orientations_SVM_Scaler_25_4w.json', 'w') as json_file:
    json.dump(model_data, json_file, indent=4)  # 使用 json.dump 写入文件，设置 indent=4 使格式更加美观

print("权重和偏置已保存到 data/model_weights_bias_20w_exact_2orientations_SVM_Scaler_25_4w.json文件中。")


# 确保所有 matplotlib 资源被释放
plt.close('all')
sys.exit(0)  # 确保程序退出




