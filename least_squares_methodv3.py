import numpy as np
from scipy.optimize import minimize

def generate_data(num_positions=4):
    positions = np.arange(num_positions)
    F_data = np.random.rand(2 * num_positions, 2) * 10
    # 生成随机数据，每个 S_data 是 2*1 大小
    S_data = np.random.rand(2 * num_positions, 2) * 10
    return F_data, S_data

def model(params, Su, Sd, l0):
    # 提取标定参数
    k00, k01, k10, k11 = params
    calibration_matrix = np.array([[k00, k01],[k10, k11]])

    #print("11:",calibration_matrix)
    # 标定矩阵
    calibrated_matrix = calibration_matrix / l0
    #print("111",calibrated_matrix)

    # 添加公式中的修正项
    correction_matrix = np.array([[0, 0],[-1, -1]])
    #print("1111", correction_matrix)
    #print("11111:", (calibrated_matrix + correction_matrix))
    #print("1:",np.vstack([Su, Sd]))
    # 计算F
    F = np.dot((calibrated_matrix + correction_matrix), np.vstack([Su, Sd]).T).T
    return F

def loss_function(params, F_data, Su_data, Sd_data, l0):
    Su_delta = Su_data - Su_data[0]
    Sd_delta = Sd_data - Sd_data[0]

    F_model = model(params, Su_delta, Sd_delta, l0)

    loss = np.sum((F_data - F_model) ** 2)
    return loss

def main():
    # 初始参数猜测
    initial_guess = np.ones(4)

    F_data, S_data = generate_data()
    l0 = 50  # 已知

    # 使用最小二乘法估计参数
    result = minimize(lambda params: loss_function(params, F_data, S_data[:4], S_data[4:], l0), initial_guess)

    # 获取估计的参数
    estimated_params = result.x
    calibrated_matrix = np.array([[estimated_params[0], estimated_params[1]], [estimated_params[2], estimated_params[3]]])

    #获取标定后的k00、k01:
    k00_calibrated = calibrated_matrix[0, 0]
    k01_calibrated = calibrated_matrix[0, 1]
    l2=k01_calibrated-k00_calibrated
    print("标定后的l1:",k00_calibrated)
    print("标定后的l2:",l2)

    # 添加公式中的修正项
    correction_matrix = np.array([[0, 0], [-l0, -l0]])

    # 计算标定后的矩阵
    final_calibrated_matrix = (calibrated_matrix + correction_matrix) / l0

    print("Final Calibrated Matrix:")
    print(calibrated_matrix)
    print("标定后的A：")
    print(final_calibrated_matrix)
    return final_calibrated_matrix,k00_calibrated,l2

if __name__=="__main__":
    main()