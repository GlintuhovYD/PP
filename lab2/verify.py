import numpy as np
import subprocess
import os
import sys

SIZE = 2000     # Размер квадратной матрицы
MATRIX_A = "mat1.txt"
MATRIX_B = "mat2.txt"
CPP_RESULT = "CPPresult.txt"
PY_RESULT = "PYresult.txt"
CPP_EXE = "lab1.exe"
PY_SCRIPT = "lab1.py"

def generate_matrices(n):
    """Генерирует две случайные матрицы и сохраняет в файлы"""
    A = np.random.randint(0, 100, size=(SIZE, SIZE))
    B = np.random.randint(0, 100, size=(SIZE, SIZE))
    np.savetxt(MATRIX_A, A, fmt='%.0f')
    np.savetxt(MATRIX_B, B, fmt='%.0f')
    print(f"Сгенерированы матрицы {n}x{n} в {MATRIX_A} и {MATRIX_B}")
    return A, B

def run_cpp():
    """Запускает скомпилированную C++ программу"""
    if not os.path.exists(CPP_EXE):
        print(f"Ошибка: не найден {CPP_EXE}")
        return False
    try:
        subprocess.run([CPP_EXE], check=True, capture_output=True, text=True)
        print("C++ программа завершена.")
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка при запуске C++ программы:")
        print(e.stderr)
        return False

def run_python():
    """Запускает Python-скрипт умножения"""
    if not os.path.exists(PY_SCRIPT):
        print(f"Ошибка: не найден {PY_SCRIPT}")
        return False
    try:
        subprocess.run([sys.executable, PY_SCRIPT], check=True, capture_output=True, text=True)
        print("Python программа завершена.")
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка при запуске Python программы:")
        print(e.stderr)
        return False

def extract_matrix_from_cpp(filename):
    """Извлекает матрицу из CPPresult.txt (до появления текстовой строки)"""
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                row = [float(x) for x in parts]
                matrix.append(row)
            except ValueError:
                break
    return np.array(matrix)

def compare():
    """Сравнивает результаты C++ и Python"""
    py_mat = np.loadtxt(PY_RESULT)
    cpp_mat = extract_matrix_from_cpp(CPP_RESULT)

    if py_mat.shape != cpp_mat.shape:
        print("Размеры матриц не совпадают!")
        print(f"Python: {py_mat.shape}, C++: {cpp_mat.shape}")
        return False

    max_diff = np.max(np.abs(py_mat - cpp_mat))
    mean_diff = np.mean(np.abs(py_mat - cpp_mat))

    if max_diff < 1e-10:
        print("Результаты совпадают в пределах погрешности.")
        return True
    else:
        print("Результаты различаются!")
        return False

def main():
    generate_matrices(SIZE)

    if not run_cpp():
        return

    if not run_python():
        return

    compare()

if __name__ == "__main__":
    main()