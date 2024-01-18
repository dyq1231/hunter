import numpy as np
import least_squares_methodv3
import unbalanced_quantity

def calculate_F(final_calibrated_matrix, E1_matrix):
    F = np.dot(final_calibrated_matrix, E1_matrix)
    return F

def calculate_M(F, omega, R):
    M = F / (omega**2 * R)
    return M

def main():
    final_calibrated_matrix,k00_calibrated,l2 = least_squares_methodv3.main()
    E1_matrix,Phase_matrix = unbalanced_quantity.main()

    result_F = calculate_F(final_calibrated_matrix, E1_matrix)

    print("Matrix F:")
    print(result_F)

    # 假设已知的参数
    omega = 10
    R = 5

    # 调用函数计算矩阵 M
    result_M = calculate_M(result_F, omega, R)

    print("Matrix M:")
    print(result_M)
    return result_M

if __name__=="__main__":
    main()
