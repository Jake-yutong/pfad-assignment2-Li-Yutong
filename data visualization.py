import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification

# 生成简单的神经网络数据（以 make_moons 为例）
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# 绘制分型图
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
plt.title("Simple Neural Network Data (Moons)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()