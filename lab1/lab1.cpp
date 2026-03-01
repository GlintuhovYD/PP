#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <omp.h>

std::vector<double> readMatrix(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::ofstream log("log.txt");
        log << "Не удалось открыть файл: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    double val;
    int count = 0;
    while (iss >> val) count++;
    n = count;

    file.clear();
    file.seekg(0);

    std::vector<double> mat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            file >> mat[i * n + j];

    file.close();
    return mat;
}

void writeResultAndInfo(const std::string& filename, const std::vector<double>& mat, int n, double elapsed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::ofstream log("log.txt");
        log << "Не удалось создать файл: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            file << mat[i * n + j] << " ";
        file << std::endl;
    }
    file << "Размер матрицы: " << n << "x" << n << std::endl;
    file << "Объем задачи (элементов результата): " << n * n << std::endl;
    file << "Время выполнения: " << elapsed << " секунд" << std::endl;
    file.close();
}

int main() {
    std::string fileA = "mat1.txt";
    std::string fileB = "mat2.txt";
    std::string outFile = "CPPresult.txt";

    int nA, nB;
    std::vector<double> A = readMatrix(fileA, nA);
    std::vector<double> B = readMatrix(fileB, nB);

    if (nA != nB) {
        std::ofstream log("log.txt");
        log << "Матрицы должны быть квадратными и одного размера!" << std::endl;
        return 1;
    }
    int n = nA;

    std::vector<double> C(n * n, 0.0);

    double start = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    double end = omp_get_wtime();
    double elapsed = end - start;

    writeResultAndInfo(outFile, C, n, elapsed);

    return 0;
}