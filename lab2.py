import numpy as np
import pandas as pd

def solve_system(matr, vec, N):
    res = np.zeros(N)
    mu = np.zeros(N)
    nu = np.zeros(N)

    # Прямой ход прогонки
    mu[0] = vec[0] / matr[0][1]
    nu[0] = -matr[0][2] / matr[0][1]
    
    for i in range(1, N - 1):
        mu[i] = (vec[i] - matr[i][0] * mu[i - 1]) / (matr[i][1] + matr[i][0] * nu[i - 1])
        nu[i] = -matr[i][2] / (matr[i][1] + matr[i][0] * nu[i - 1])

    mu[N - 1] = vec[N - 1] / matr[N - 1][1]
    nu[N - 1] = -matr[N - 1][0] / matr[N - 1][1]

    # Обратный ход
    res[N - 1] = mu[N - 1] + nu[N - 1] * mu[N - 2] / (1 - nu[N - 1] * nu[N - 2])
    for i in range(N - 2, -1, -1):
        res[i] = mu[i] + nu[i] * res[i + 1]

    return res

def cubic_spline(X, Y, A, B):
    N = len(X) - 1
    h = np.diff(X)

    # Проверка на ошибки
    if N < 2:
        return None, 1  # Недостаточно точек
    if not np.all(np.diff(X) > 0):
        return None, 2  # Нарушен порядок возрастания

    # Инициализация массивов
    matr = np.zeros((N + 1, 3))  # Трёхдиагональная матрица
    vec = np.zeros(N + 1)  # Вектор правых частей

    # Устанавливаем граничные условия для начальной точки
    matr[0][1] = 1  # Главная диагональ для первой точки
    vec[0] = A  # c_0 = A

    # Заполнение основной части матрицы и правых частей
    for i in range(1, N):
        matr[i][0] = h[i - 1]  # Поддиагональный элемент
        matr[i][1] = 2 * (h[i - 1] + h[i])  # Главная диагональ
        matr[i][2] = h[i]  # Наддиагональный элемент
        vec[i] = 6 * ((Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1])  # Правая часть

    # Устанавливаем граничные условия для последней точки
    matr[N][0] = h[N - 1] / 2  # Поддиагональный элемент
    matr[N][1] = h[N - 1]  # Главная диагональ для последней точки
    vec[N] = 3 / h[N - 1] * (B - (Y[N] - Y[N - 1]) / h[N - 1])  # Учет условия на первую производную

    # Решение системы
    C = solve_system(matr, vec, N + 1)

    # Заполнение коэффициентов сплайна
    b = np.zeros(N)
    d = np.zeros(N)
    second_derivatives = np.zeros(N + 1)

    for i in range(N):
        second_derivatives[i] = C[i]  # Вторая производная

    for i in range(N):
        d[i] = (C[i + 1] - C[i]) / (3 * h[i])  # Вычисление d

    for i in range(N):
        b[i] = (Y[i + 1] - Y[i]) / h[i] - h[i] * (2 * C[i] + C[i + 1]) / 3  # Вычисление b

    return second_derivatives, b, d, 0  # Возвращаем вторые производные, b и d

# Пример входных данных
X = np.array([0, 1, 2, 3, 4])
#Y = np.array([0, 1, 0, 1, 0])
Y = X**3
print(Y)
A = 0  # Вторая производная в начале
B = 1  # Первая производная в конце

# Построение сплайна
second_derivatives, b_coefficients, d_coefficients, IER = cubic_spline(X, Y, A, B)

# Вывод таблицы
if IER == 0:
    table = pd.DataFrame({
        'x': X[:-1],  # Исключаем последний элемент X, чтобы длины совпадали
        'f(x)': Y[:-1],  # Соответственно, убираем последний элемент Y
        "f''(x)": second_derivatives[:-1],  # исключаем последний элемент
        'b': b_coefficients,
        'd': d_coefficients
    })
    print(table.to_string(index=False))
else:
    print("Ошибка:", IER)

