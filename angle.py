import numpy as np
import unbalanced_quantity
import least_squares_methodv3
import final

def calculate_phi(theta_u, theta_d, theta_1, theta_2, theta_3, theta_4, l0, l1, l2, Mu, Mv, omega, R):
    # 计算公式中的各项
    term1 = (l1 / l0) * np.cos(theta_u + theta_1) + ((l1 + l2) / l0) * np.cos(theta_d + theta_2)
    term2 = (-1 * (l0 + l1) / l0) * np.cos(theta_u + theta_3) - ((l0 + l1 + l2) / l0) * np.cos(theta_d + theta_4)

    # 计算角度
    phi_u = np.arccos(term1 / (Mu * omega ** 2 * R))
    phi_d = np.arccos(term2 / (Mv * omega ** 2 * R))

    return np.degrees(phi_u), np.degrees(phi_d)

def main():
    # 随机生成角度
    theta_1 = np.radians(np.random.rand())
    theta_2 = np.radians(np.random.rand())
    theta_3 = np.radians(np.random.rand())
    theta_4 = np.radians(np.random.rand())

    # 已知参数
    l0 = 50  # 已知
    omega = 10
    R = 5

    #theta u、d
    E1_matrix, Phase_matrix=unbalanced_quantity.main()
    theta_u=Phase_matrix[0]
    theta_d=Phase_matrix[1]
    #print("theta_u",theta_u)

    #l1,l2
    final_calibrated_matrix, k00_calibrated, l2=least_squares_methodv3.main()
    l1=k00_calibrated

    #mu\mv
    result_M=final.main()
    Mu=result_M[0]
    Mv=result_M[1]


     #计算角度
    phi_u, phi_d = calculate_phi(theta_u, theta_d, theta_1, theta_2, theta_3, theta_4, l0, l1, l2, Mu, Mv, omega, R)

    print("Phi_u:", phi_u)
    print("Phi_d:", phi_d)

if __name__=="__main__":
    main()


