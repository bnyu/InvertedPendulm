import numpy as np
from sympy import *
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('TkAgg')


# 作轨迹图
def track(y, t, rd, step):
    if t==2000:
        t-=1
    time = np.arange(0, t+1)
    s = np.zeros(t+1)
    if step!=0:
        for j in range(step, t+1):
            s[j]=4
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


# 作动画
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
    ax = plt.axes(xlim=(-5, 5), ylim=(0, 10))
    line_wheel_left, = ax.plot([], [], lw=2, color='g')
    line_wheel_right, = ax.plot([], [], lw=2, color='g')
    line_cart_left, = ax.plot([], [], lw=2, color='k')
    line_cart_right, = ax.plot([], [], lw=2, color='k')
    line_cart_top, = ax.plot([], [], lw=2, color='k')
    line_cart_bottom, = ax.plot([], [], lw=2, color='k')
    line_hinge, = ax.plot([], [], lw=4, color='y')
    line_pendulum, = ax.plot([], [], lw=5, color='y')

    def animate(ii):
        line_wheel_left.set_data(y[ii, 2] + wheel[0] - cart_width/3, wheel[1] + wheel_radius)
        line_wheel_right.set_data(y[ii, 2] + wheel[0] + cart_width/3, wheel[1] + wheel_radius)
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
                                (wheel_radius + cart_high, wheel_radius + cart_high + pendulum_y)])
        return (line_wheel_left, line_wheel_right, line_cart_bottom, line_cart_top,
                line_cart_left, line_cart_right, line_hinge, line_pendulum)

    ani = animation.FuncAnimation(fig, animate, frames=t, interval=5, blit=True, init_func=None)
    plt.show(ani)


def model():
    print('小车质量：1.0kg\n倒立摆质量：0.1kg\n倒立摆长度：2.0m')
    if input('是否默认倒立摆模型：Enter/n') != 'n':
        cart_mass = 1.0
        pendulum_mass = 0.1
        pendulum_length = 2.0
    else:
        cart_mass = 0.5*min(3, max(1, int(input('请选择小车质量：1:(0.5kg)，2:(1.0kg)，3:(2.0kg)'))))
        pendulum_mass = 0.05*min(3, max(1, int(input('请选择摆杆质量：1:(0.05kg)，2:(0.1kg)，3:(0.2kg)'))))
        pendulum_length = 1.0*min(3, max(1, int(input('请选择摆杆长度：1:(1.0m)，2:(2.0m)，3:(4.0m)'))))
    gravity = 9.81
    a21 = (cart_mass + pendulum_mass) * gravity / (cart_mass * pendulum_length / 2)
    a41 = - pendulum_mass * gravity / cart_mass
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
    k_1 = kr[k1].evalf()
    k_2 = kr[k2].evalf()
    k_3 = kr[k3].evalf()
    k_4 = kr[k4].evalf()
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("状态反馈矩阵：", kf)
    return kf


def control(k):
    print('\n初始摆杆角度：-15度\n初始摆杆角速度：-1.5弧度/秒\n初始小车位置：2.5米\n初始小车速度：-1.5米/秒')
    if input('是否默认初始倒立摆状态：Enter/n') != 'n':
        x1 = -15.0/180 * np.pi
        x2 = -1.5
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
    dt = 0.01
    time = 2000
    sign = True
    rd = np.array([0])
    if input('\n是否产生随机干扰信号：Enter/n') != 'n':
        d = 1
        sign = False
    else:
        d = 0
    s = np.zeros(2000)
    step = 0
    if input('\n是否在随机时间产生阶跃信号：Enter/n') != 'n':
        step = random.randint(800, 1000)
        for j in range(step, 1000):
            s[j] = 4
        sign = False
    print('\n正在进行状态反馈动态控制...')
    for i in range(1, 2000):
        if (not sign) and i%3==0:
            r = random.uniform(-0.5, 0.5) * d
        else:
            r = 0
        rd = np.vstack((rd, r))
        x = x + ((np.dot((A - (np.dot(B, k))), (x - z))) + np.dot(B, r+s[i])) * dt
        y = np.vstack((y, x.T))
        if (sign) and (abs(x[0])<1e-2 and abs(x[1])<1e-2 and abs(x[2])<1e-2 and abs(x[3])<1e-2):
            time = i
            break
    return y, time, rd, step


print('Start')
while(True):
    A, B = model()
    K = calculate(A, B)
    Y, T, Rd, Step = control(K)
    if input('\n是否作动画：Enter/n') != 'n':
        animated(Y, T)
    if input('\n是否作轨迹图：Enter/n') != 'n':
        track(Y, T, Rd, Step)
    if input('\n是否继续：Enter/n') == 'n':
        print('End')
        break
    
    


