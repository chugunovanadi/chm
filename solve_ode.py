import enum
import numpy as np

from utils.ode_collection import ODE
from S3T2_solve_ode.py.one_step_methods import OneStepMethod
from S3T2_solve_ode.py.one_step_methods import ExplicitEulerMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, ode: ODE, y_start, ts):
    """
    Интегрирование одношаговым методом с фиксированным шагом

    :param method:  одношаговый метод
    :param ode:     СОДУ
    :param y_start: начальное значение
    :param ts:      набор значений t
    :return:        список значений t (совпадает с ts), список значений y
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(ode, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, ode: ODE, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    Интегрирование одношаговым методом с адаптивным выбором шага.
    Допуски контролируют локальную погрешность:
        err <= atol
        err <= |y| * rtol

    :param method:      одношаговый метод
    :param ode:         СОДУ
    :param y_start:     начальное значение
    :param t_span:      интервал интегрирования (t0, t1)
    :param adapt_type:  правило Рунге (AdaptType.RUNGE) или вложенная схема (AdaptType.EMBEDDED)
    :param atol:        допуск на абсолютную погрешность
    :param rtol:        допуск на относительную погрешность
    :return:            список значений t (совпадает с ts), список значений y
    """
    y = y_start
    t, t_end = t_span

    ys = [y]
    ts = [t]

    p = method.p
    tol = rtol * np.linalg.norm(y) + atol
    beg = ode(t, y)
    delta = (1 / max(abs(t), abs(t_end))) ** (p + 1) + (np.linalg.norm(beg)) ** (p + 1)
    h1 = (tol / delta) ** (1 / (p + 1))
    u1 = ExplicitEulerMethod().step(ode, t, y, h1)
    beg = ode(t + h1, u1)
    delta = (1 / max(abs(t), t_end)) ** (p + 1) + (np.linalg.norm(beg)) ** (p + 1)
    h1_new = pow(tol / delta, 1 / (p + 1))
    h = min(h1, h1_new)
    while t < t_end:
        if t + h > t_end:
            h = t_end - t
        if adapt_type == AdaptType.RUNGE:
            y1 = method.step(ode, t, y, h)
            y_half = method.step(ode, t, y, h / 2)
            y2 = method.step(ode, t + h / 2, y_half, h / 2)
            err = (y2 - y1) / (1 - 2 ** (-p))
        else:
            y2, err = method.embedded_step(ode, t, y, h)
        tol = rtol * np.linalg.norm(y) + atol
        err = np.linalg.norm(err)
        if err > tol * (2 ** p):
            h /= 2
            if adapt_type == AdaptType.RUNGE:
                y = y_half
        elif (err > tol) and (err <= tol * (2 ** p)):
            t += h
            h /= 2
            y = y2
            ys.append(y)
            ts.append(t)
        else:
            t += h
            if adapt_type == AdaptType.RUNGE:
                y = y1
            else:
                y = y2
            ys.append(y)
            ts.append(t)
        if err < tol / 2 ** (p + 1):
            h *= 2
    return ts, ys