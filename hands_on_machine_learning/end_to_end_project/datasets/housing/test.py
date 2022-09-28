import numpy as np
import pandas as pd
test_ratio = 0.1
data = []
for i in range(100):
    data.append(i)
print(data)
data = pd.DataFrame(data)
shuffled_indices = np.random.permutation(len(data))
test_set_size = int(len(data) * test_ratio)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
print("测试集是\n",data.iloc[test_indices])
print("训练集是\n", data.iloc[train_indices])
