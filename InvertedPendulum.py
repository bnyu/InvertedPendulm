import numpy as np
from sympy import *
import Daolibai as Dao


def model():
    print('小车质量：1.0kg\n倒立摆质量：0.1kg\n倒立摆长度：2.0m')
    if input('是否默认倒立摆模型：Enter/n') != 'n':
        cart_mass = 1.0
        pendulum_mass = 0.1
        pendulum_length = 2.0
    else:
        cart_mass = min(2.0, max(0.5, float(input('请输入小车质量(0.5kg--2kg)：'))))
        pendulum_mass = min(0.2, max(0.05, float(input('请输入摆杆质量(0.05kg--0.2kg)：'))))
        pendulum_length = min(4.0, max(1.0, float(input('请输入摆杆长度(1m--4m)：'))))
    gravity = 9.81
    a21 = (cart_mass + pendulum_mass) * gravity / (cart_mass * pendulum_length / 2)
    a41 = -pendulum_mass * gravity / cart_mass
    b21 = -1 / (cart_mass * pendulum_length / 2)
    b41 = 1 / cart_mass
    a = np.array([[0, 1, 0, 0], [a21, 0, 0, 0], [0, 0, 0, 1], [a41, 0, 0, 0]])
    b = np.array([[0], [b21], [0], [b41]])
    return a, b


def calculate(a, b):
    print('\n正在计算倒立摆系统能控性...')
    temp = b
    h = temp.T
    for n in range(1, 4):
        temp = np.dot(a, temp)
        h = np.vstack((h, temp.T))
    d = np.linalg.det(h)
    if d != 0:
        print("系统是完全能控的。")
    else:
        print("系统是不完全能控的。")
        exit(1)

    print('\n正在计算系统开环极点...')
    r = Symbol('r', complex=True)
    pole = solve(Matrix.det(Matrix(r * eye(4) - Matrix(a))), r)
    print("系统开环极点：", pole)

    print('\n正在进行闭环系统极点配置...')
    k1, k2, k3, k4 = symbols('k1 k2 k3 k4', real=True)
    kr = np.array([[k1, k2, k3, k4]])
    r1 = Symbol('r1', complex=True)
    ri = diag(r1, r1, r1, r1)
    s2 = (r1 + 1) * (r1 + 2) * (r1 + 1 - 1j) * (r1 + 1 + 1j)
    print("配置系统闭环极点： [", -1, -2, -1 + 1j, -1 - 1j, "]")
    print('\n正在计算状态反馈矩阵...')
    s1 = Matrix.det(ri - (Matrix(a) - Matrix(b) * Matrix(kr)))
    kr = solve(s1 - s2, [k1, k2, k3, k4])
    if type(kr[k1]) == Symbol:
        print('未能解出反馈矩阵')
        print(kr)
        if input('是否手动求解：Enter/n') == 'n':
            exit(1)
        k_1 = float(input('请输入反馈系数k1：'))
        k_2 = float(input('请输入反馈系数k2：'))
        k_3 = float(input('请输入反馈系数k3：'))
        k_4 = float(input('请输入反馈系数k4：'))
    else:
        k_1 = kr[k1].evalf()
        k_2 = kr[k2].evalf()
        k_3 = kr[k3].evalf()
        k_4 = kr[k4].evalf()
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("状态反馈矩阵：", kf)
    return kf


def control(k):
    print('\n初始摆杆角度：-15度\n初始摆杆角速度：-2度/秒\n初始小车位置：2.5米\n初始小车速度：-1.5米/秒')
    if input('是否默认初始倒立摆状态：Enter/n') != 'n':
        x1 = -15.0/180 * np.pi
        x2 = -2.0
        x3 = 2.5
        x4 = -1.5
    else:
        x1 = min(20.0, max(-20.0, float(input('请输入摆杆角度(小于20度)：')))) / 180 * np.pi
        x2 = 0.0
        x3 = min(4.0, max(-4.0, float(input('请输入小车位置：'))))
        x4 = 0.0
    x = np.array([[x1], [x2], [x3], [x4]])
    y = x.T
    z = np.zeros((4, 1))
    f = x
    dt = 0.01
    print('\n正在进行状态反馈控制...')
    for i in range(1, 1000):
        f = f + (np.dot((A - (np.dot(B, k))), (f - z))) * dt
        y = np.vstack((y, f.T))
    return y


print('Start')
while(True):
    A, B = model()
    K = calculate(A, B)
    Y = control(K)
    if input('\n是否作动画：Enter/n') != 'n':
        Dao.animated(Y)
    if input('\n是否作轨迹图：Enter/n') != 'n':
        Dao.track(Y)
    if input('\n是否继续：Enter/n') == 'n':
        print('End')
        break
    
    


