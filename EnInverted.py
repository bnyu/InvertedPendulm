import numpy as np
import sympy as sy
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('TkAgg')


def model():
    print('Cart Mass:1.0kg\nPendulum Mass:0.1kg\nPendulum Length:2.0m\n')
    if input('Default set:Enter/n') != 'n':
        cart_mass = 1.0
        pendulum_mass = 0.1
        pendulum_length = 2.0
    else:
        cart_mass = min(2.0, max(0.5, float(input('Please input cart mass:(0.5kg--2kg):'))))
        pendulum_mass = min(0.2, max(0.05, float(input('Please input pendulum mass:(0.05kg--0.2kg):'))))
        pendulum_length = min(4.0, max(1.0, float(input('Please input pendulum length:(1m--4m):'))))
    gravity = 9.81
    a21 = (cart_mass + pendulum_mass) * gravity / (cart_mass * pendulum_length / 2)
    a41 = -pendulum_mass * gravity / cart_mass
    b21 = -1 / (cart_mass * pendulum_length / 2)
    b41 = 1 / cart_mass
    a = np.array([[0, 1, 0, 0], [a21, 0, 0, 0], [0, 0, 0, 1], [a41, 0, 0, 0]])
    b = np.array([[0], [b21], [0], [b41]])
    return a, b


def calculate(a, b):
    print('\nCalculating controllable of inverted pendulum system...')
    temp = b
    h = temp.T
    for n in range(1, 4):
        temp = np.dot(a, temp)
        h = np.vstack((h, temp.T))
    d = np.linalg.det(h)
    if d != 0:
        print("The system is totally controllable.")
    else:
        print("The system is not totally controllable.")
        exit(1)

    print('\nCalculating system open loop poles...')
    r = sy.Symbol('r', complex=True)
    pole = sy.solve(sy.Matrix.det(sy.Matrix(r * sy.eye(4) - sy.Matrix(a))), r)
    print("System open loop poles:", pole)

    print('\nConfiguring pole placement of closed-loop systems...')
    k1, k2, k3, k4 = sy.symbols('k1 k2 k3 k4', real=True)
    kr = np.array([[k1, k2, k3, k4]])
    r1 = sy.Symbol('r1', complex=True)
    ri = sy.diag(r1, r1, r1, r1)
    s2 = (r1 + 1) * (r1 + 2) * (r1 + 1 - 1j) * (r1 + 1 + 1j)
    print("System closed loop poles: [", -1, -2, -1 + 1j, -1 - 1j, "]")
    print('\nCalculating state feedback matrix...')
    s1 = sy.Matrix.det(ri - (sy.Matrix(a) - sy.Matrix(b) * sy.Matrix(kr)))
    kr = sy.solve(s1 - s2, [k1, k2, k3, k4])
    if type(kr[k1]) == sy.Symbol:
        print('Error', kr)
        exit(1)
    k_1 = kr[k1].evalf()
    k_2 = kr[k2].evalf()
    k_3 = kr[k3].evalf()
    k_4 = kr[k4].evalf()
    kf = np.array([[k_1, k_2, k_3, k_4]])
    print("State feedback matrix:", kf)
    return kf


def control(k):
    print('\nInitial pendulum degrees:-15degrees\nInitial pendulum angular speed:-2degrees/s\n'
          'Initial cart position:2.5m\nInitial cart speed:1.5m/s\n')
    if input('Default set:Enter/n') != 'n':
        x1 = -15.0/180 * np.pi
        x2 = -2.0
        x3 = 2.5
        x4 = 1.5
    else:
        x1 = min(20.0, max(-20.0, float(input('Please input initial pendulum degrees:')))) / 180 * np.pi
        x2 = 0.0
        x3 = min(2.0, max(-2.0, float(input('Please input initial cart position:'))))
        x4 = 0.0
    x = np.array([[x1], [x2], [x3], [x4]])
    y = x.T
    z = np.zeros((4, 1))
    f = x
    dt = 0.01
    print('\nState feedback controlling...')
    for i in range(1, 1000):
        f = f + (np.dot((A - (np.dot(B, k))), (f - z))) * dt
        y = np.vstack((y, f.T))
    return y


def track(y):
    time = np.arange(0, 1000)
    angle = y[:, 0]
    angle_velocity = y[:, 1]
    position = y[:, 2]
    velocity = y[:, 3]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplot(2, 2, 1)
    plt.plot(time, angle)
    plt.title('Pendulum Angle')
    plt.subplot(2, 2, 2)
    plt.plot(time, angle_velocity)
    plt.title('Angular Velocity')
    plt.subplot(2, 2, 3)
    plt.plot(time, position)
    plt.title('Cart Position')
    plt.subplot(2, 2, 4)
    plt.plot(time, velocity)
    plt.title('Cart Velocity')
    plt.show(fig)


def animated(y):
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

    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=2, blit=True, init_func=None)
    plt.show(ani)


print('Start')
while(True):
    A, B = model()
    K = calculate(A, B)
    Y = control(K)
    if input('\nShow animation:Enter/n') != 'n':
        animated(Y)
    if input('\nShow tracking:Enter/n') != 'n':
        track(Y)
    if input('\nContinue:Enter/n') == 'n':
        break
print('End')
