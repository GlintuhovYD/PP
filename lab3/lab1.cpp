#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

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

void writeMatrixToFile(const std::string& filename, const std::vector<double>& mat, int n) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::ofstream log("log.txt");
        log << "Не удалось создать файл: " << filename << std::endl;
        exit(1);
    }

    file << std::fixed;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            file << mat[i * n + j] << " ";
        file << std::endl;
    }
    file.close();
}

void writeInfoToFile(const std::string& filename, int n, int num_procs, double elapsed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::ofstream log("log.txt");
        log << "Не удалось создать файл: " << filename << std::endl;
        exit(1);
    }
    file << "Количество процессов MPI: " << num_procs << std::endl;
    file << "Размер матрицы: " << n << "x" << n << std::endl;
    file << "Объем задачи (элементов результата): " << n * n << std::endl;
    file << "Время выполнения умножения: " << elapsed << " секунд" << std::endl;
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string fileA = "mat1.txt";
    std::string fileB = "mat2.txt";
    std::string outFile = "CPPresult.txt";
    std::string infoFile = "info.txt";

    int nA, nB;
    std::vector<double> A = readMatrix(fileA, nA);
    std::vector<double> B = readMatrix(fileB, nB);

    if (nA != nB) {
        std::ofstream log("log.txt");
        log << "Матрицы должны быть квадратными и одного размера!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int n = nA;

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    std::vector<double> local_C(local_rows * n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        int global_i = start_row + i;
        for (int k = 0; k < n; ++k) {
            double aik = A[global_i * n + k];
            for (int j = 0; j < n; ++j) {
                local_C[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double elapsed = end - start;

    std::vector<double> C;
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        C.resize(n * n);
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int rows_p = rows_per_proc + (p < remainder ? 1 : 0);
            recv_counts[p] = rows_p * n;
            displs[p] = offset;
            offset += recv_counts[p];
        }
    }

    MPI_Gatherv(local_C.data(), local_rows * n, MPI_DOUBLE,
                C.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        writeMatrixToFile(outFile, C, n);
        writeInfoToFile(infoFile, n, size, elapsed);
    }

    MPI_Finalize();
    return 0;
}