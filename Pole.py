import numpy as np
from sympy import *
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# 因为[]中有数字会被识别为引用,故用zero,one 代替0,1
zero = 0
one = 1


# 对比轨迹图
def track(y1, y2, y3, t):
    time = np.arange(0, t)
    angle1 = y1[:, 0]
    angle_velocity1 = y1[:, 1]
    position1 = y1[:, 2]
    velocity1 = y1[:, 3]
    angle2 = y2[:, 0]
    angle_velocity2 = y2[:, 1]
    position2 = y2[:, 2]
    velocity2 = y2[:, 3]
    angle3 = y3[:, 0]
    angle_velocity3 = y3[:, 1]
    position3 = y3[:, 2]
    velocity3 = y3[:, 3]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplot(2, 2, 1)
    plt.plot(time, angle1)
    plt.plot(time, angle2)
    plt.plot(time, angle3)    
    plt.title('Pendulum Angle')
    plt.subplot(2, 2, 3)
    plt.plot(time, angle_velocity1)
    plt.plot(time, angle_velocity2)
    plt.plot(time, angle_velocity3)  
    plt.title('Angular Velocity')
    plt.subplot(2, 2, 2)
    plt.plot(time, position1)
    plt.plot(time, position2)
    plt.plot(time, position3)    
    plt.title('Cart Position')
    plt.subplot(2, 2, 4)
    plt.plot(time, velocity1)
    plt.plot(time, velocity2)
    plt.plot(time, velocity3)   
    plt.title('Cart Velocity')
    plt.show(fig)


def model():
    print('小车质量：1.0kg\n倒立摆质量：0.1kg\n倒立摆长度：2.0m')
    if input('是否默认倒立摆模型：Enter/n') != 'n':
        cart_mass = 1.0
        pendulum_mass = 0.1
        pendulum_length = 2.0
    else:
        cart_mass = (0.5*min(3, max(1, int(
            input('请选择小车质量：1:(0.5kg)，2:(1.0kg)，3:(1.5kg)，4:(2.0kg)')))))
        pendulum_mass = (0.05*min(3, max(1, int(
            input('请选择摆杆质量：1:(0.05kg)，2:(0.1kg)，3:(0.15kg)，4:(0.2kg)')))))
        pendulum_length = (1.0*min(3, max(1, int(
            input('请选择摆杆长度：1:(1.0m)，2:(2.0m)，3:(3m)，4:(4.0m)')))))
    gravity = 9.81
    a21 = (cart_mass + pendulum_mass) * gravity / (cart_mass * pendulum_length / 2)
    a41 = - pendulum_mass * gravity / cart_mass
    b21 = -1 / (cart_mass * pendulum_length / 2)
    b41 = 1 / cart_mass
    a = np.array([[zero, 1, 0, 0], [a21, 0, 0, 0], [zero, 0, 0, 1], [a41, 0, 0, 0]])
    b = np.array([[zero], [b21], [zero], [b41]])
    return a, b


def calculate(a, b, sn):
    k1, k2, k3, k4 = symbols('k1 k2 k3 k4', real=True)
    kr = np.array([[k1, k2, k3, k4]])
    r1 = Symbol('r1', complex=True)
    ri = diag(r1, r1, r1, r1)
    s1 = Matrix.det(ri - (Matrix(a) - Matrix(b) * Matrix(kr)))
    # 设置极点
    if sn == 2:
        s2 = (r1 + 1) * (r1 + 2) * (r1 + 1 - 2j) * (r1 + 1 + 2j)
    elif sn == 3:
        s2 = (r1 + 3) * (r1 + 2) * (r1 + 3 - 1j) * (r1 + 3 + 1j)
    else:
        s2 = (r1 + 1) * (r1 + 2) * (r1 + 1 - 1j) * (r1 + 1 + 1j)
    kr = solve(s1 - s2, [k1, k2, k3, k4])
    k_1 = kr[k1].evalf()
    k_2 = kr[k2].evalf()
    k_3 = kr[k3].evalf()
    k_4 = kr[k4].evalf()
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("状态反馈矩阵：", kf)
    return kf


def initialize():
    print('\n初始摆杆角度：0度\n初始摆杆角速度：0度/秒\n'
          '初始小车位置：-9米\n初始小车速度：0米/秒')
    if input('是否默认初始倒立摆状态：Enter/n') != 'n':
        x1 = 0.0 / 180 * np.pi
        x2 = 0.0 / 180 * np.pi
        x3 = -9.0
        x4 = 0.0
    else:
        x1 = min(20.0, max(-20.0, float(
            input('请输入初始摆杆角度(绝对值小于20度)：')))) / 180 * np.pi
        x2 = min(60.0, max(-60.0, float(
            input('请输入初始摆杆角速度(绝对值小于90度/秒)：')))) / 180 * np.pi
        x3 = min(20.0, max(-20.0, float(
            input('请输入初始小车位置(绝对值小于20米)：'))))
        x4 = min(30.0, max(-30.0, float(
            input('请输入初始小车速度(绝对值小于30米/秒)：'))))
    x = np.array([[x1], [x2], [x3], [x4]])
    return x


def control(a, b, k, x):
    y = x.T
    z = np.zeros((4, 1))
    dt = 0.01
    times = 2000
    rd = np.array([zero])
    for jr in range(1, times):
        r = 0.0
        if jr % 4 == 0:
            r = random.uniform(-3.0, 3.0)
        rd = np.vstack((rd, r))
    print('\n正在进行状态反馈动态控制...')
    for i in range(1, times):
        x = x + ((np.dot((a - (np.dot(b, k))), (x - z))) + np.dot(b, [rd[i]+0.0])) * dt
        y = np.vstack((y, x.T))
    return y


A, B = model()
X = initialize()
T = 2000
K1 = calculate(A, B, 1)
Y1 = control(A, B, K1, X)
K2 = calculate(A, B, 2)
Y2 = control(A, B, K2, X)
K3 = calculate(A, B, 3)
Y3 = control(A, B, K3, X)
track(Y1, Y2, Y3, T)


