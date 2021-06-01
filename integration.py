import numpy as np
from math import comb
import math
"""
Вычисляем моменты весовой функции с 0-го по max_s-ый на интервале [xl, xr]
Весовая функция: p(x) = 1 / (x-a)^alpha / (b-x)^beta, причём гарантируется, что:
        1) 0 <= alpha < 1
        2) 0 <= beta < 1
        3) alpha * beta = 0

:param max_s:   номер последнего момента
:return:        список значений моментов
"""

def mA(x, a, alpha, s):
    ans = 0
    for i in range(0, s+1):
        ans += comb(s, i) * a**(s-i) * (x-a)**(i+1-alpha) / (i+1-alpha)
    return ans

def momentA(xl, xr, a, alpha, s):
        return mA(xr, a, alpha, s) - mA(xl,  a, alpha, s)

def mB(x, b, beta, s):
    ans = 0
    for i in range(0, s+1):
        ans += (-1)**(i+1) * comb(s, i) * b**(s-i) * (b - x)**(i+1-beta) / (i+1-beta)
    return ans

def momentB(xl, xr, b, beta, s):
        return mB(xr, b, beta, s) - mB(xl, b, beta, s)

def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):

    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        return [momentA(xl, xr, a, alpha, s) for s in range(0, max_s + 1)]
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        return [momentB(xl, xr, b, beta, s) for s in range(0, max_s + 1)]

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError



def runge(s0, s1, m, L):
    """
    Оценка погрешности последовательных приближений s0 и s1 по правилу Рунге

    :param m:   порядок погрешности
    :param L:   кратность шага
    :return:    оценки погрешностей s0 и s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оценка порядка главного члена погрешности по последовательным приближениям s0, s1 и s2 по правилу Эйткена
    Считаем, что погрешность равна R(h) = C*h^m + o(h^m)

    :param L:   кратность шага
    :return:    оценка порядка главного члена погрешности (m)
    """
    return - (np.log(abs((s2 - s1) / (s1 - s0))) / np.log(L))


def quad(func, x0, x1, xs, **kwargs):
    """
    Интерполяционная квадратурная формула

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param xs:      узлы
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    n = len(xs)
    nodes = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(nodes), (n, n))
    A = np.linalg.solve(w, moments(n - 1, x0, x1, **kwargs))
    return sum(A * np.array([func(x) for x in xs]))



def quad_gauss(func, x0, x1, n, **kwargs):
    """
    Интерполяционная квадратурная формула типа Гаусса

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param n:       количество узлов
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """

    nyu = moments(2 * n - 1, x0, x1, **kwargs)
    w = np.array([[(nyu[j + s]) for j in range(n)] for s in range(n)])
    b = - np.array(nyu[n:])
    coef_a = np.append(np.linalg.solve(w, b), 1)
    xs = np.roots(coef_a[::-1])
    nodes = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(nodes), (n, n))
    A = np.linalg.solve(w, nyu[:n])
    return sum(A * np.array([func(x) for x in xs]))


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    Составная квадратурная формула

    :param func:        интегрируемая функция
    :param x0, x1:      интервал
    :param n_intervals: количество интервалов
    :param n_nodes:     количество узлов на каждом интервале
    :param kwargs:      параметры весовой функции (должны передаваться в moments)
    :return:            значение СКФ
    """
    bounds = np.linspace(x0, x1, n_intervals+1)
    ans = 0
    for i in range(n_intervals):
        ans += quad(func, bounds[i], bounds[i+1], np.linspace(bounds[i], bounds[i+1], n_nodes), **kwargs)
    return ans


def integrate(func, x0, x1, tol):
    """
    Интегрирование с заданной точностью (error <= tol)
    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге и выбираем оптимальный размер шага
    Делаем так, пока оценка погрешности не уложится в tol
    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param tol:     допуск
    :return:        значение интеграла, оценка погрешности
    """
    n_nodes = 3
    L = 2
    h_opt = x1 - x0
    error = tol + 1
    while error >= tol:
        n_intervals = [math.ceil(int((x1 - x0) / h_opt)) * L ** s for s in range(3)]
        compos_h = [composite_quad(func, x0, x1, interval, n_nodes) for interval in n_intervals]
        m = aitken(*compos_h, L)
        error = runge(*compos_h[:-1], m, L)[0]
        h_opt = compos_h[0] * (tol / error) ** (1 / m)
        print(n_intervals)
    return compos_h[0], error
