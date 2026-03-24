import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载数据 - 使用相对路径
data_dir = os.path.join(os.path.dirname(__file__), 'data')
df_day = pd.read_csv(os.path.join(data_dir, 'day.csv'))
df_hour = pd.read_csv(os.path.join(data_dir, 'hour.csv'))

# 转换日期格式
df_day['dteday'] = pd.to_datetime(df_day['dteday'])
df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])

# 1. 季节与租赁量关系图
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='cnt', data=df_day)
plt.title('每季度自行车租赁量分布', fontsize=14)
plt.xlabel('季节 (1:春 2:夏 3:秋 4:冬)', fontsize=12)
plt.ylabel('租赁总量', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 2. 工作日与租赁量关系图
plt.figure(figsize=(10, 6))
sns.boxplot(x='workingday', y='cnt', data=df_day)
plt.title('工作日vs周末租赁量对比', fontsize=14)
plt.xlabel('是否工作日 (0:周末/假日 1:工作日)', fontsize=12)
plt.ylabel('租赁总量', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 3. 天气与租赁量关系图
plt.figure(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=df_day)
plt.title('不同天气情况下的租赁量', fontsize=14)
plt.xlabel('天气情况 (1:晴 2:多云 3:小雨/雪 4:大雨/雪)', fontsize=12)
plt.ylabel('租赁总量', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 4. 按小时的平均骑行量
hourly_avg = df_hour.groupby('hr')['cnt'].mean()
plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
plt.title('按小时的平均骑行量', fontsize=14)
plt.xlabel('小时', fontsize=12)
plt.ylabel('平均骑行量', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 5. 温度与骑行量的散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_day['temp'], df_day['cnt'], alpha=0.5)
plt.title('温度与骑行量的关系', fontsize=14)
plt.xlabel('归一化温度', fontsize=12)
plt.ylabel('骑行量', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 6. 各月份的平均骑行量
monthly_avg = df_day.groupby('mnth')['cnt'].mean()
plt.figure(figsize=(12, 6))
plt.bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='black')
plt.title('各月份的平均骑行量', fontsize=14)
plt.xlabel('月份', fontsize=12)
plt.ylabel('平均骑行量', fontsize=12)
plt.show()

# 数据预处理与建模
feature_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X = df_day[feature_cols]
y = df_day['cnt']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化模型
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.01)
}

results = {}
predictions = {}

print('训练模型中...')
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, model.predict(X_train_scaled))
    test_mse = mean_squared_error(y_test, y_pred)
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    predictions[name] = y_pred
    
    print(f'\n{name}:')
    print(f'  训练集MSE: {train_mse:.4f}')
    print(f'  测试集MSE: {test_mse:.4f}')
    print(f'  训练集R²: {train_r2:.4f}')
    print(f'  测试集R²: {test_r2:.4f}')

# 绘制综合评估图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. R²分数对比
ax = axes[0, 0]
model_names = list(results.keys())
train_r2 = [results[m]['train_r2'] for m in model_names]
test_r2 = [results[m]['test_r2'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, train_r2, width, label='训练集', color='skyblue')
ax.bar(x + width/2, test_r2, width, label='测试集', color='orange')
ax.set_ylabel('R²', fontsize=12)
ax.set_title('模型R²分数对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. MSE对比
ax = axes[0, 1]
train_mse = [results[m]['train_mse'] for m in model_names]
test_mse = [results[m]['test_mse'] for m in model_names]
ax.bar(x - width/2, train_mse, width, label='训练集', color='skyblue')
ax.bar(x + width/2, test_mse, width, label='测试集', color='orange')
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('模型MSE对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 预测值vs实际值 (第一个模型)
ax = axes[1, 0]
first_model = model_names[0]
ax.plot(y_test.values[:50], label='实际值', marker='o', linewidth=2)
ax.plot(predictions[first_model][:50], label='预测值', marker='s', linewidth=2)
ax.set_ylabel('骑行量', fontsize=12)
ax.set_title(f'{first_model} - 前50个预测值', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 预测值分布
ax = axes[1, 1]
for model in model_names:
    ax.hist(predictions[model], alpha=0.5, label=model, bins=20)
ax.set_xlabel('预测骑行量', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('各模型预测值分布', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'model_evaluation.png'), dpi=100)
plt.close()

print('\n所有可视化已保存！')
how()

