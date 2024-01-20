import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
file_path = 'C:/Users/student/yunjiee-python/MBTI_project/full/loss.csv'  # 請替換成您的文件路徑
data = pd.read_csv(file_path)

# 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(data.iloc[:, 0], data.iloc[:, 1], marker='o', color='skyblue')
plt.title('fine_tune訓練的損失率')
plt.xlabel('訓練週期')
plt.ylabel('損失率')
plt.grid(True)
plt.show()
