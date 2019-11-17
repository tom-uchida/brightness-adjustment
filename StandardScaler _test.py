import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import japanize_matplotlib

# 特徴量を作成
np.random.seed(seed=1)
data = np.random.multivariate_normal( [5, 5], [[5, 0],[0, 2]], 150 )

# スケール変換器を作成
sc = StandardScaler()

#特徴量をスケール変換
data_std = sc.fit_transform(data)

# プロット
min_x = min(data[:,0])
max_x = max(data[:,0])

min_y = min(data[:,1])
max_y = max(data[:,1])

plt.figure(figsize=(6, 6))
plt.title('StandardScalerによる標準化', fontsize=15)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.scatter(data[:, 0], data[:, 1], c='red', marker='o', s=30, label='元々のデータ')
plt.scatter(data_std[:, 0], data_std[:, 1], c='blue', marker='o', s=30, label='スケール変換後')
plt.legend(loc='best', fontsize=15)
plt.hlines(0,xmin=-10, xmax=10, linestyles='dotted')
plt.vlines(0,ymin=-10, ymax=10, linestyles='dotted')
plt.show()