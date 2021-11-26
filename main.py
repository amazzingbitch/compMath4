import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.abc import p
import math


# исходная функция
def function(x):
    return pow(2, x) - 3*x - 2


# функция для нахождения значения ошибок аппроксимации
def mistake(a, b):
    return math.fabs(a - b)


# полином Тейлора
def taylor_polynomial(x, dec_p, n, func):
    s = func.evalf(subs={p: dec_p})
    func = func.diff(p)

    for i in range(n + 1):
        s += (func.evalf(subs={p: dec_p}) * ((x - dec_p) ** (i + 1))) / math.factorial(i + 1)
        func = func.diff(p)

    return s


# нахождение ошибок аппроксимации
def approximation_mistakes(a, b, st, dec_p, func):
    m1, m2, m3, m4 = [], [], [], []
    j = a

    while j < b - st:
        m1.append(taylor_polynomial(j, dec_p, 1, func))
        m2.append(taylor_polynomial(j, dec_p, 2, func))
        m3.append(taylor_polynomial(j, dec_p, 3, func))
        m4.append(taylor_polynomial(j, dec_p, 4, func))
        j += st

    return m1, m2, m3, m4


# графики ошибок аппроксимации
def draw_mistakes(x, m1, m2, m3, m4, dec_p):
    plt.title("Plots of approximation mistakes for the decomposition point x =" + str(dec_p))
    plt.plot(x, m1, label='Approx. mistake for a polynomial of pow 1')
    plt.plot(x, m2, label='Approx. mistake for a polynomial of pow 2')
    plt.plot(x, m3, label='Approx. mistake for a polynomial of pow 3')
    plt.plot(x, m4, label='Approx. mistake for a polynomial of pow 4')
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.grid(True)
    plt.legend()
    plt.show()


# интерполяционный полином Лагранжа
def lagrange_polynomial(x, lag_p, func, n):
    s = 0
    for i in range(n):
        t = 1
        for j in range(n):
            if j != i:
                t *= (x - lag_p[j]) / (lag_p[i] - lag_p[j])
        s += (func.evalf(subs={p: lag_p[i]}) * t)
    return s


# графики интерполяционного полинома Лагранжа
def draw_lagrange_poly(x, func, lag_p, n):
    plt.title("The graphs of the original function and the Lagrange polynomial for n =" + str(n))
    plt.plot(x, lagrange_polynomial(x, lag_p, func, n), 'g--')
    plt.plot(x, function(x), color="red")
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.grid(True)
    plt.show()

A = -4.5
B = 3.7
ST = 0.01
X = np.arange(A, B, ST)
xn = 0.9
dec_points = [-4.3, 3.5, -0.4]
lag_points1 = [-3.2, -1.8, 0.8, 2.5, 3.2]
lag_points2 = [-3.9, -2.3, -1.4, -0.3, 0.7, 1.1, 2.6, 2.9, 3.4, 3.6]
f = pow(2, p) - 3*p - 2
y = f.evalf(subs={p: xn})

# Исследование ошибок аппроксимации многочленом Тейлора
for i in range(len(dec_points)):
    app_m1, app_m2, app_m3, app_m4 = approximation_mistakes(A, B, ST, dec_points[i], f)
    draw_mistakes(X, app_m1, app_m2, app_m3, app_m4, dec_points[i])

for n in range(1, 5):
    print("\nFor n =", n, "mistakes:")
    print("c = a: %.7f" % mistake(y, taylor_polynomial(xn, dec_points[0], n, f)))
    print("c = b: %.7f" % mistake(y, taylor_polynomial(xn, dec_points[1], n, f)))
    print("c = (a+b)/2: %.7f" % mistake(y, taylor_polynomial(xn, dec_points[2], n, f)))

# Исследование ошибок интерполяции многочленом Лагранжа
print("\nLagrange, n = 5")
draw_lagrange_poly(X, f, lag_points1, 5)
y1 = lagrange_polynomial(xn, lag_points1, f, 5)
for n in range(1, 5):
    print("For power n =", n)
    print("mistake = %.7f\n" % mistake(y1, taylor_polynomial(xn, dec_points[2], n, f)))

print("\nLagrange, n = 10")
draw_lagrange_poly(X, f, lag_points2, 10)
y2 = lagrange_polynomial(xn, lag_points2, f, 10)
for n in range(1, 5):
    print("For pow n =", n)
    print("mistake = %.7f\n" % mistake(taylor_polynomial(xn, dec_points[2], n, f), y2))