import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件，忽略警告
csv_file_path = ('D:/A-deng/hunter/1.17/sensor1/CH12/Waveform-right-180.csv')
encoding = 'utf-8'
data = pd.read_csv(csv_file_path, encoding=encoding, header=None, skiprows=2)

#显示文件行数
num_rows = len(data)
print(f'文件行数：{num_rows}')

# 选择序号和CH1数据列
sequence_column = data.iloc[2:, 0].astype(float)
ch1_data = data.iloc[2:, 1].astype(float)

# 计算对应的时间
time_interval = 0.0000004
time_column = np.arange(0, len(sequence_column) * time_interval, time_interval)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(time_column, ch1_data, label='CH1', marker='o', linestyle='-')
plt.title('CH1                                                                         Data Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()


