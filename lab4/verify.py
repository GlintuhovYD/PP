import numpy as np
import subprocess
import os
import sys

SIZE = 500
THREADS = 32
MATRIX_A = "mat1.txt"
MATRIX_B = "mat2.txt"
CPP_RESULT = "CPPresult.txt"
PY_RESULT = "PYresult.txt"
CPP_EXE = "lab1.exe"
PY_SCRIPT = "lab1.py"

def generate_matrices(n):
    A = np.random.randint(0, 100, size=(n, n))
    B = np.random.randint(0, 100, size=(n, n))
    np.savetxt(MATRIX_A, A, fmt='%.0f')
    np.savetxt(MATRIX_B, B, fmt='%.0f')
    print(f"Сгенерированы матрицы {n}x{n} в {MATRIX_A} и {MATRIX_B}")

def run_cpp(threads):
    if not os.path.exists(CPP_EXE):
        print(f"Ошибка: не найден {CPP_EXE}")
        return False
    try:
        subprocess.run([CPP_EXE, str(threads)], check=True, capture_output=True, text=True)
        print("C++ программа завершена.")
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка при запуске C++ программы:")
        if e.stderr:
            print(e.stderr)
        return False

def run_python():
    if not os.path.exists(PY_SCRIPT):
        print(f"Ошибка: не найден {PY_SCRIPT}")
        return False
    try:
        subprocess.run([sys.executable, PY_SCRIPT], check=True, capture_output=True, text=True)
        print("Python программа завершена.")
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка при запуске Python программы:")
        if e.stderr:
            print(e.stderr)
        return False

def compare():
    py_mat = np.loadtxt(PY_RESULT)
    cpp_mat = np.loadtxt(CPP_RESULT)
    if py_mat.shape != cpp_mat.shape:
        print("Размеры матриц не совпадают!")
        return False
    max_diff = np.max(np.abs(py_mat - cpp_mat))
    if max_diff < 1e-10:
        print("Результаты совпадают в пределах погрешности.")
        return True
    else:
        print("Результаты различаются!")
        return False

def main():
    generate_matrices(SIZE)
    if not run_cpp(THREADS):
        return
    if not run_python():
        return
    compare()

if __name__ == "__main__":
    main()