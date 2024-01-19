import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_data(root):
    file_name = root + r"\left_0\left_0_sensor1_ch12.csv"


    df = pd.read_csv(file_name)
    y = df['Math'].drop(df.index[0]).astype(float).to_numpy()
    size = len(y)
    x0 = 0
    increment = df["Increment"].iloc[0].astype(float)
    xn = increment*size
    x = np.arange(x0, xn, increment)

    return x, y

# 步骤 2: 定义模型函数
def sine_model(x, a, b):
    return a * np.sin(b * x)

# 计算移动平均
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 使用 matplotlib 绘制数据点和拟合曲线
def plot_figure(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='true')
    plt.plot(x, sine_model(x, *popt), label='curve', color='red')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sine curve')
    plt.show()

if __name__ == "__main__":
    root = r"D:\abc"
    x_data, y_data = read_data(root)
    # 步骤 4: 进行曲线拟合
    # 使用 curve_fit 找到最佳拟合参数
    popt, pcov = curve_fit(sine_model, x_data, y_data, p0=[0.4, 20])

    a = popt[0]
    b = popt[1]
    print(a, b)

    # 应用移动平均
    window_size = 5
    y_smoothed = moving_average(y_data, window_size)

    plot_figure(x_data, y_smoothed)

    a = b



# # 步骤 2: 定义模型函数
# def sine_model(x, a, b):
#     return a * np.sin(b * x)

# # 步骤 3: 生成模拟数据
# # 创建一些带有随机噪声的模拟数据
# np.random.seed(0) # 确保每次运行结果相同
# x_data = np.linspace(0, 2 * np.pi, 50)
# y_data = 3 * np.sin(1.5 * x_data) + np.random.normal(size=x_data.size)

# # 步骤 4: 进行曲线拟合
# # 使用 curve_fit 找到最佳拟合参数
# popt, pcov = curve_fit(sine_model, x_data, y_data, p0=[2, 2])

# # 步骤 5: 展示结果
# # 打印出拟合参数
# print(f"拟合参数 a 和 b: {popt}")

# # 使用 matplotlib 绘制数据点和拟合曲线
# plt.figure(figsize=(10, 6))
# plt.scatter(x_data, y_data, label='数据点')
# plt.plot(x_data, sine_model(x_data, *popt), label='拟合曲线', color='red')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('正弦函数拟合示例')
# plt.show()
    



# import numpy as np
# import matplotlib.pyplot as plt

# # 示例数据
# np.random.seed(0)
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x) + np.random.normal(0, 0.2, 100)

# # 计算移动平均
# def moving_average(data, window_size):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# # 应用移动平均
# window_size = 5
# y_smoothed = moving_average(y, window_size)

# # 绘制原始数据和平滑后的数据
# plt.figure(figsize=(12, 6))
# plt.plot(x, y, label='原始数据', alpha=0.5)
# plt.plot(x[:len(y_smoothed)], y_smoothed, label='移动平均平滑', color='red')
# plt.title("移动平均数据平滑示例")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

