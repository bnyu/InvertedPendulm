import numpy as np
from sympy import *
import Daolibai as Dao

# 倒立摆系统模型
cart_mass = 1.0
pendulum_mass = 0.1
pendulum_length = 2.0
gravity = 9.81
a21 = (cart_mass + pendulum_mass) * gravity / (cart_mass * pendulum_length / 2)
a41 = -pendulum_mass * gravity / cart_mass
b21 = -1 / (cart_mass * pendulum_length / 2)
b41 = 1 / cart_mass
A = np.array([[0, 1, 0, 0], [a21, 0, 0, 0], [0, 0, 0, 1], [a41, 0, 0, 0]])
B = np.array([[0], [b21], [0], [b41]])


def calculate():
    # 系统能控性
    temp = B
    h = temp.T
    for n in range(1, 4):
        temp = np.dot(A, temp)
        h = np.vstack((h, temp.T))
    d = np.linalg.det(h)
    if d != 0:
        print("系统是完全能控的")
    else:
        print("系统是不完全能控的")
        exit(1)

    # 系统开环极点
    r = Symbol('r', complex=True)
    pole = solve(Matrix.det(Matrix(r * eye(4) - Matrix(A))), r)
    print("系统开环极点:", pole)

    # 闭环极点配置
    k1, k2, k3, k4 = symbols('k1 k2 k3 k4', real=True)
    kr = np.array([[k1, k2, k3, k4]])
    r1 = Symbol('r1', complex=True)
    ri = diag(r1, r1, r1, r1)
    s2 = (r1 + 1) * (r1 + 2) * (r1 + 1 - 1j) * (r1 + 1 + 1j)
    print("系统闭环极点:", -1, -2, -1 + 1j, -1 - 1j)
    s1 = Matrix.det(ri - (Matrix(A) - Matrix(B) * Matrix(kr)))
    kr = solve(s1 - s2, [k1, k2, k3, k4])
    k_1 = kr[k1]
    k_2 = kr[k2]
    k_3 = kr[k3]
    k_4 = kr[k4]
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("状态反馈矩阵:", kf)
    return kf

K = calculate()


# 状态反馈控制
def control(k):
    x1 = min(25.0, max(-25.0, float(input('请输入摆杆角度:')))) / 180 * np.pi
    x2 = 0.0
    x3 = min(25.0, max(-25.0, float(input('请输入小车位置:')))) / 10
    x4 = 0.0
    x = np.array([[x1], [x2], [x3], [x4]])
    y = x.T
    z = np.zeros((4, 1))
    f = x
    dt = 0.01
    for i in range(1, 1000):
        f = f + (np.dot((A - (np.dot(B, k))), (f - z))) * dt
        y = np.vstack((y, f.T))
    return y

Y = control(K)


if input('''是否作轨迹图y/n''') == 'y':
    Dao.track(Y)

if input('''是否作动画y/n''') == 'y':
    Dao.animated(Y)


