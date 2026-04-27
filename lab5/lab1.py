import numpy as np

# Чтение матриц из файлов
A = np.loadtxt('mat1.txt')
B = np.loadtxt('mat2.txt')

# Проверка размеров
if A.shape != B.shape or A.shape[0] != A.shape[1]:
    print('Ошибка: матрицы должны быть квадратными и одинакового размера')
else:
    # Умножение матриц
    C = A @ B   # или np.dot(A, B)
    
    # Сохранение результата
    np.savetxt('PYresult.txt', C, fmt='%.0f')
    print('Результат сохранён в PYresult.txt')