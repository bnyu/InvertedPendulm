import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 作轨迹图
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


# 作动画
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

