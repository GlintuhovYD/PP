#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

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

void writeInfoToFile(const std::string& filename, int n, int threadsPerBlock, dim3 grid, double elapsed) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::ofstream log("log.txt");
        log << "Не удалось создать файл: " << filename << std::endl;
        exit(1);
    }
    file << "Размер матрицы: " << n << "x" << n << std::endl;
    file << "Объем задачи (элементов результата): " << n * n << std::endl;
    file << "Потоков на блок: " << threadsPerBlock << "x" << threadsPerBlock << std::endl;
    file << "Размер сетки: " << grid.x << "x" << grid.y << std::endl;
    file << "Время выполнения на GPU: " << elapsed << " секунд" << std::endl;
    file.close();
}

int main(int argc, char* argv[]) {
    int threadsPerBlock = 16;
    if (argc > 1) {
        threadsPerBlock = std::atoi(argv[1]);
        if (threadsPerBlock < 1) threadsPerBlock = 1;
        if (threadsPerBlock > 32) {
            std::cerr << "Предупреждение: threadsPerBlock > 32 (будет ограничено 32)" << std::endl;
            threadsPerBlock = 32;
        }
    }

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
        return 1;
    }
    int n = nA;

    size_t size = n * n * sizeof(double);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice));

    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double elapsed = elapsed_ms / 1000.0;

    std::vector<double> C(n * n);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    writeMatrixToFile(outFile, C, n);
    writeInfoToFile(infoFile, n, threadsPerBlock, grid, elapsed);

    std::cout << "Умножение завершено. Результат в " << outFile << std::endl;
    return 0;
}