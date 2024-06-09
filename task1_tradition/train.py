import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# 读取特征张量
all_features = torch.load('all_features.pt')
labels = torch.load('labels.pt')

X = all_features
y = labels

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

hidden_layer_sizes = [(50,), (100,), (150,), (200,), (250,), (300,)]
activation_functions = ['relu', 'tanh', 'logistic']
MSEs = []

for activation in tqdm(activation_functions):
    for hidden_layer_size in hidden_layer_sizes:
        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=activation, solver='adam', max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        y_pred = mlp.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        MSEs.append([hidden_layer_size, activation, mse])

# 绘图
for activation in activation_functions:
    tmp = []
    for hidden_layer_size, act, mse in MSEs:
        if act == activation:
            tmp.append(mse)
    plt.plot(hidden_layer_sizes, tmp, label=activation)
plt.xlabel('Hidden Layer Size')
plt.ylabel('MSE')
plt.title('MLP Accuracy vs. Hidden Layer Size')
plt.grid(True)
plt.legend()
plt.show()