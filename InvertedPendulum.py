import numpy as np
from sympy import *
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('TkAgg')


# 开环响应 闭环响应 轨迹图
def track(y, t, rd, step1, step0, size=2.0):
    time = np.arange(0, t)
    s = np.zeros(t)
    if step1 != 0:
        for j in range(step1, step0):
            s[j] = size
    angle = y[:, 0]
    angle_velocity = y[:, 1]
    position = y[:, 2]
    velocity = y[:, 3]
    fig, ax = plt.subplots(nrows=2, ncols=3)
    plt.subplot(2, 3, 1)
    plt.plot(time, angle)
    plt.title('Pendulum Angle')
    plt.subplot(2, 3, 4)
    plt.plot(time, angle_velocity)
    plt.title('Angular Velocity')
    plt.subplot(2, 3, 2)
    plt.plot(time, position)
    plt.title('Cart Position')
    plt.subplot(2, 3, 5)
    plt.plot(time, velocity)
    plt.title('Cart Velocity')
    plt.subplot(2, 3, 3)
    plt.plot(time, rd)
    plt.title('Random Disturbing')
    plt.subplot(2, 3, 6)
    plt.plot(time, s)
    plt.title('Step signal')
    plt.show(fig)


# 动画仿真
def animated(y, t):
    cart_high = 3
    cart_width = 5
    length = 5.5
    wheel_radius = 0.5
    hinge_radius = 0.08
    circle = np.arange(0, 2*np.pi, np.pi/100)
    wheel = [wheel_radius*np.sin(circle), wheel_radius*np.cos(circle)]
    hinge = [hinge_radius*np.sin(circle), hinge_radius*np.cos(circle)]
    fig = plt.figure()
    ax = plt.axes(xlim=(-15, 15), ylim=(0, 20))
    line_wheel_left, = ax.plot([], [], lw=2, color='g')
    line_wheel_right, = ax.plot([], [], lw=2, color='g')
    line_cart_left, = ax.plot([], [], lw=2, color='k')
    line_cart_right, = ax.plot([], [], lw=2, color='k')
    line_cart_top, = ax.plot([], [], lw=2, color='k')
    line_cart_bottom, = ax.plot([], [], lw=2, color='k')
    line_hinge, = ax.plot([], [], lw=4, color='y')
    line_pendulum, = ax.plot([], [], lw=5, color='y')

    def animate(ii):
        line_wheel_left.set_data(y[ii, 2] + wheel[0] - cart_width/3,
                                 wheel[1] + wheel_radius)
        line_wheel_right.set_data(y[ii, 2] + wheel[0] + cart_width/3,
                                  wheel[1] + wheel_radius)
        line_cart_bottom.set_data([(y[ii, 2] - cart_width/2, y[ii, 2] + cart_width/2),
                                   (wheel_radius, wheel_radius)])
        line_cart_top.set_data([(y[ii, 2] - cart_width/2, y[ii, 2] + cart_width/2),
                                (wheel_radius + cart_high, wheel_radius + cart_high)])
        line_cart_left.set_data([(y[ii, 2] - cart_width/2, y[ii, 2] - cart_width/2),
                                 (wheel_radius, wheel_radius + cart_high)])
        line_cart_right.set_data([(y[ii, 2] + cart_width/2, y[ii, 2] + cart_width/2),
                                  (wheel_radius, wheel_radius + cart_high)])
        line_hinge.set_data(y[ii, 2] + hinge[0], wheel_radius + cart_high + hinge[1])
        pendulum_x = length*math.sin(y[ii, 0])
        pendulum_y = length*math.cos(y[ii, 0])
        line_pendulum.set_data([(y[ii, 2], y[ii, 2] + pendulum_x),
                                (wheel_radius + cart_high,
                                 wheel_radius + cart_high + pendulum_y)])
        return (line_wheel_left, line_wheel_right, line_cart_bottom, line_cart_top,
                line_cart_left, line_cart_right, line_hinge, line_pendulum)

    ani = animation.FuncAnimation(fig, animate, frames=t, interval=5, blit=True, init_func=None)
    ani.save('line.mp4', fps=100)
    plt.show(ani)


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
    a = np.array([[0, 1, 0, 0], [a21, 0, 0, 0], [0, 0, 0, 1], [a41, 0, 0, 0]])
    b = np.array([[0], [b21], [0], [b41]])
    return a, b


def calculate(a, b):
    print('A=', a)
    print('B=', b)
    c = np.eye(4)
    print('C=', c)
    print('\n正在计算倒立摆系统能控性...')
    temp = b
    h = temp.T
    for n in range(1, 4):
        temp = np.dot(a, temp)
        h = np.vstack((h, temp.T))
    d = np.linalg.det(h)
    if d != 0:
        print('We=', h, '为行满秩')
        print("系统是完全能控的。")
    else:
        print('We=', h, '不为行满秩')
        print("系统是不完全能控的。")
        exit(1)

    print('\n正在计算倒立摆系统能观性...')
    temp2 = c
    h2 = temp2
    for n2 in range(1, 4):
        temp2 = np.dot(temp2, a)
        h2 = np.vstack((h2, temp2))
    d = np.linalg.matrix_rank(h2)
    if d == 4:
        print('Wo=', h2, '为列满秩')
        print("系统是完全能观的。")
    else:
        print('Wo=', h2, '不为列满秩')
        print("系统是不完全能观的。")
        exit(1)

    print('正在计算系统开环响应...')
    x = np.array([[0.0], [0.0], [0.0], [0.0]])
    y = x.T
    if input('\n是否默认阶跃大小为4N：Enter/n') != 'n':
        f = 4.0
    else:
        f = min(5.0, max(-5.0, float(input('请输入阶跃大小：'))))
    dt = 0.01
    # 简易模型没有考虑摆杆向下情况
    times = 150
    rd = np.zeros(times)
    for i in range(1, times):
        x = x + (np.dot(a, x) + np.dot(b, [[f]])) * dt
        y = np.vstack((y, x.T))
    if input('\n是否作图开环阶跃响应：Enter/n') != 'n':
        track(y, times, rd, 1, times, f)
    if input('\n是否动画演示：Enter/n') != 'n':
        animated(y, times)

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
    k_1 = kr[k1].evalf()
    k_2 = kr[k2].evalf()
    k_3 = kr[k3].evalf()
    k_4 = kr[k4].evalf()
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("状态反馈矩阵：", kf)

    print('阶跃大小不变，正在计算系统闭环响应...')
    x = np.array([[0.0], [0.0], [0.0], [0.0]])
    z = np.zeros((4, 1))
    y = x.T
    times = 1000
    rd = np.zeros(times)
    for i2 in range(1, times):
        x = x + ((np.dot((a - (np.dot(b, kf))), x-z)) + np.dot(b, [[f]])) * dt
        y = np.vstack((y, x.T))
    if input('\n是否作图闭环阶跃响应：Enter/n') != 'n':
        track(y, times, rd, 1, times, f)
    if input('\n是否动画演示：Enter/n') != 'n':
        animated(y, times)
    return kf


def control(k):
    print('\n初始摆杆角度：-15度\n初始摆杆角速度：-5度/秒\n'
          '初始小车位置：7.5米\n初始小车速度：-3.5米/秒')
    if input('是否默认初始倒立摆状态：Enter/n') != 'n':
        x1 = -15.0 / 180 * np.pi
        x2 = -5.0 / 180 * np.pi
        x3 = 7.5
        x4 = -3.5
    else:
        x1 = min(20.0, max(-20.0, float(
            input('请输入初始摆杆角度(绝对值小于20度)：')))) / 180 * np.pi
        x2 = min(10.0, max(-10.0, float(
            input('请输入初始摆杆角速度(绝对值小于10度/秒)：')))) / 180 * np.pi
        x3 = min(10.0, max(-10.0, float(
            input('请输入初始小车位置(绝对值小于10米)：'))))
        x4 = min(30.0, max(-30.0, float(
            input('请输入初始小车速度(绝对值小于30米/秒)：'))))
    x = np.array([[x1], [x2], [x3], [x4]])
    y = x.T
    z = np.zeros((4, 1))
    dt = 0.01
    times = 2000
    rd = np.array([0])
    if input('\n是否产生随机干扰信号：Enter/n') != 'n':
        for jr in range(1, times):
            r = 0.0
            if jr % 4 == 0:
                r = random.uniform(-0.6, 0.6)
            rd = np.vstack((rd, r))
    else:
        rd = np.zeros(times)
    s = np.zeros(times)
    step1 = 0
    step0 = times
    size = 0.0
    if input('\n是否在随机一段时间产生方波推力：Enter/n') != 'n':
        if input('\n是否默认阶跃大小为3N：Enter/n') != 'n':
            size = 3.0
        else:
            size = min(5.0, max(-5.0, float(input('请输入阶跃大小：'))))
        step1 = random.randint(700, 800)
        step0 = random.randint(1000, 1200)
        for j in range(step1, step0):
            s[j] = size
    print('\n正在进行状态反馈动态控制...')
    for i in range(1, times):
        x = x + ((np.dot((A - (np.dot(B, k))), (x - z))) + np.dot(B, [rd[i]+s[i]])) * dt
        y = np.vstack((y, x.T))
    return y, rd, step1, step0, times, size



print('Start')
while (True):
    A, B = model()
    K = calculate(A, B)
    Y, Rd, Step1, Step0, T, Size = control(K)
    if input('\n是否作动画：Enter/n') != 'n':
        animated(Y, T)
    if input('\n是否作轨迹图：Enter/n') != 'n':
        track(Y, T, Rd, Step1, Step0, Size)
    if input('\n是否继续：Enter/n') == 'n':
        print('End')
        break

