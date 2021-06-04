import numpy as np
from scipy.optimize import fsolve

import S3T2_solve_ode.py.coeffs_collection as collection
from utils.ode_collection import ODE


class OneStepMethod:
    def __init__(self, **kwargs):
        self.name = 'default_method'
        self.p = None  # порядок
        self.__dict__.update(**kwargs)

    def step(self, ode: ODE, t, y, dt):
        """
        делаем шаг: t => t+dt, используя ode(t, y)
        """
        return ode(t+dt, y)



class ExplicitEulerMethod(OneStepMethod):
    """
    Явный метод Эйлера (ничего менять не нужно)
    """
    def __init__(self):
        super().__init__(name='Euler (explicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        return y + dt * ode(t, y)


class ImplicitEulerMethod(OneStepMethod):
    """
    Неявный метод Эйлера
    Подробности: https://en.wikipedia.org/wiki/Backward_Euler_method
    """
    def __init__(self):
        super().__init__(name='Euler (implicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        def left(y1):
            return y1 - y - dt * ode(t, y1)
        return fsolve(left, y)

class RungeKuttaMethod(OneStepMethod):
    """
    Явный метод Рунге-Кутты с коэффициентами (A, b)
    Замените метод step() так, чтобы он не использовал встроенный класс RK45
    """
    def __init__(self, coeffs: collection.RKScheme):
        super().__init__(**coeffs.__dict__)

    def step(self, ode: ODE, t, y, dt):
        A, b = self.A, self.b
        n = len(A)
        st = 0
        K = []
        for i in range(n):
            temp = 0
            c = np.sum(A[i, :])
            for j in range(0, i):
                temp += A[i, j] * K[j]
            K.append(ode(t + dt * c, y + dt * temp))
            st += b[i] * K[i]
        return y + dt * st


class EmbeddedRungeKuttaMethod(RungeKuttaMethod):
    """
        Вложенная схема Рунге-Кутты с параметрами (A, b, e):
        Шаг с использованием вложенных методов
        y1 = RK(func, A, b)
        y2 = RK(func, A, d), где d = b+e
        embedded_step() должен возвращать:
            - приближение (y1)
            - разность приближений (dy = y2-y1)
        """
    def __init__(self, coeffs: collection.EmbeddedRKScheme):
        super().__init__(coeffs=coeffs)

    def embedded_step(self, ode: ODE, t, y, dt):
        A, b, e = self.A, self.b, self.e
        c = np.sum(A, axis=1)
        n = len(A)
        st = 0
        K = []
        for i in range(n):
            temp = 0
            for j in range(i):
                temp += A[i, j] * K[j]
            K.append(ode(t + dt * c, y + dt * temp))
            st += b[i] * K[i]
        return y + dt * st, np.dot(e, K)



class EmbeddedRosenbrockMethod(OneStepMethod):
    """
        Вложенный метод Розенброка с параметрами (A, G, gamma, b, e):
        Шаг с использованием вложенных методов
        y1 = Rosenbrock(func, A, G, gamma, b)
        y2 = Rosenbrock(func, A, G, gamma, d), где d = b+e
        embedded_step() должен возвращать:
            - приближение (y1)
            - разность приближений (dy = y2-y1)
        """
    def __init__(self, coeffs: collection.EmbeddedRosenbrockScheme):
        super().__init__(**coeffs.__dict__)

    def embedded_step(self, ode: ODE, t, y, dt):
        A, G, g, b, e, q = self.A, self.G, self.gamma, self.b, self.e, self.q
        I = np.eye(len(y))
        J = ode.jacobian(t, y)
        coeff_k = I - g * dt * J
        k = [np.linalg.solve(coeff_k, dt * ode(t, y))]
        for i in range(1, q):
            temp_a = sum(A[i, j] * k[j] for j in range(i))
            temp_g = sum(G[i, j] * k[j] for j in range(i))
            right_k = dt * ode(t, y + temp_a) + dt * np.dot(J, temp_g)
            k += [np.linalg.solve(coeff_k, right_k)]
        return y + np.dot(b, k), np.dot(e, k)
