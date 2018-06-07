import numpy as np
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def relu(x, matrix):
    y = matrix.dot(x)
    y[y < 0] = 0
    return y


def test_relu():
    n = 3
    x = np.arange(n)
    matrix = np.random.random((n, n))-0.5
    return relu(x, matrix)


def get_xy(n=301):
    cen = n / 2.0
    a = np.arange(n)
    x, y = np.meshgrid(a, a)
    x = x - cen
    y = y - cen
    return x, y


def relu_xy(x, y, matrix):
    xy = [(i, j) for i, j in zip(x.flatten(), y.flatten())]
    v = [relu(i, matrix) for i in xy]
    v1 = np.reshape(np.array([i[0] for i in v]), x.shape)
    v2 = np.reshape(np.array([i[1] for i in v]), x.shape)
    return v1, v2


def get_relu_trans():
    x, y = get_xy()
    n = x.shape[0]
    matrix = np.random.randn(2, 2)
    v1, v2 = relu_xy(x, y, matrix)
    return x, y, v1, v2


def get_relu_region():
    x, y, v1, v2 = get_relu_trans()

    w1, w2 = 10 * np.random.randn(2)
    v1 -= w1
    v2 -= w2

    v = v1 ** 2 + v2 ** 2
    reg2 = x ** 2 + y ** 2
    reg2 *= 1e-4
    reg1 = abs(x) + abs(y)
    reg1 *= 0.3
    reg = reg1
    vreg = v + reg

    return x, y, v1, v2, v, vreg


def get_mins_max(x, y, v):
    v_min = v.min()
    v_max = v.max()
    w_min = v == v_min
    x_min = x[w_min]
    y_min = y[w_min]

    # print('min=%s\nx_min=%s\ny_min=%s' % (v_min, x_min, y_min))
    return x_min, y_min, v_min, v_max


def scale(v):
    return np.log10(1e-5 + abs(v))


def show_four(x, y, v1, v2, v, vreg):
    plt.clf()

    plt.subplot(2, 2, 1)
    show_relu(x, y, v1**2)
    plt.title('V1^2')
    plt.subplot(2, 2, 2)
    show_relu(x, y, v2**2)
    plt.title('V2^2')
    plt.subplot(2, 2, 3)
    show_relu(x, y, v)
    plt.title('V1^2 + V2^2')
    plt.subplot(2, 2, 4)
    show_relu(x, y, vreg, with_min=True)
    plt.show()
    plt.title('V1^2 + V2^2 + L1-norm')


def show_relu(x, y, v, with_min=False):
    v_scaled = scale(v)
    plt.imshow(v_scaled, interpolation='bicubic')

    x_min, y_min, v_min, v_max = get_mins_max(x, y, v)
    nx = x.shape[0]
    cen = nx/2.0
    vs_min = v_scaled.min()
    vs_max = v_scaled.max()
    levels = np.linspace(vs_min, vs_max, 300)
    plt.contour(v_scaled, levels)

    plt.vlines(cen, 0, nx, color='gray')
    plt.hlines(cen, 0, nx, color='gray')

    if with_min:
        plt.vlines(cen + x_min, 0, nx, color='red', linestyle='--')
        plt.hlines(cen + y_min, 0, nx, color='red', linestyle='--')
    plt.draw()


def show_sum_squares():
    x, y, v1, v2, v, vreg = get_relu_region()
    show_four(x, y, v1, v2, v, vreg)


def show_mult(num, with_min=True):
    v_sum = 0.0
    vv1 = 0.0
    vv2 = 0.0
    for i in range(num):
        x, y, v1, v2, v, vreg = get_relu_region()
        v_sum = v_sum + v
        vv1 = vv1 + v1
        vv2 = vv2 + v2

    plt.clf()
    show_four(x, y, vv1, vv2, v_sum, v_sum + vreg)
    #show_relu(x, y, v_sum, with_min=with_min)


def loop_sum_squares():
    key = None
    while True:
        show_sum_squares()
        key = input('\nok? enter q to exit\n')
        if key == 'q':
            return


def prox(f, lam):
    x = np.linspace(-20,20, 10000)
    def prox_func(v):
        func = f(x) + lam*(x-v)**2
        i_min = np.argmin(func)
        return x[i_min]
    return prox_func


def half_quad(x):
    y = 2.0
    q = x**2 - 2*x*y
    q[q<0] = 0
    return q


def plot_prox_half_quad():
    v = np.linspace(0,10,100)
    prox_func = prox(half_quad, 1.0)
    prox_val = np.array([prox_func(i) for i in v])
    plt.plot(v,prox_val)


def theta(x, r):
    return 1.0*(x > r)


def show_relax_maj():
    x = np.linspace(-2, 3, 1000)
    y = 1.0
    f = y**2 + theta(x, 0)*(x**2 - 2*x*y)
    plt.clf()
    plt.plot(x, f, linewidth=6, label="Non-convex", color="black")
    f = theta(x, y) * (x-y)**2
    plt.plot(x, f, linewidth=2, label="Convex relaxation", color="blue", linestyle='--')
    f = (x - y) ** 2
    plt.plot(x, f, linewidth=2, label="Majorization, w_0 > 0", color="magenta", linestyle='--')
    th = theta(x, 2*y)
    f = (1-th) * y**2 + th * (x - y) ** 2
    plt.plot(x, f, linewidth=2, label="Majorization, w_0 < 0", color="red", linestyle='--')
    plt.ylim(-1, 5)
    plt.legend()


def show_dcc():

    x = np.linspace(-3, 3, 10000)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(x, abs(x), linewidth=3)
    plt.title('f(x) = |x|')

    plt.subplot(2, 2, 2)
    plt.plot(x, 0.5 * x**2, linewidth=3)
    plt.title('g(x) = 1/2 x^2')

    plt.subplot(2, 2, 3)
    plt.plot(x, abs(x) - 0.5*(x ** 2), linewidth=3)
    plt.title('h(x) = |x| - 1/2 x^2')

    plt.subplot(2, 2, 4)
    plt.plot(x, abs(x) - (2 + 2* (x-2)), linewidth=3)
    plt.plot(x, abs(x) - 0.5 * (x ** 2), color='red', linewidth=3, linestyle='--')
    plt.title('H(x) = |x| - (2 + 2 (x-2))')
    plt.show()



if __name__ == "__main__":
    loop_sum_squares()
