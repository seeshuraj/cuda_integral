#include <cuda_runtime.h>
#include <math.h>
#include "exponentialIntegralGPU.h"

__device__ float computeExpIntFloat(int n, float x) {
    return expf(-x) / (n + 1.0f);
}

__global__ void exponentialIntegralFloatKernel(float* out, int n, int m, float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (idx < total) {
        int i = idx / m;
        int j = idx % m;
        float x = a + j * (b - a) / m;
        out[idx] = computeExpIntFloat(i + 1, x);
    }
}

void runExponentialIntegralFloatGPU(float* output, int n, int m, float a, float b) {
    int total = n * m;
    float* d_output;
    cudaMalloc(&d_output, total * sizeof(float));
    
    dim3 blockSize(256);
    dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
    exponentialIntegralFloatKernel<<<gridSize, blockSize>>>(d_output, n, m, a, b);

    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}

__device__ double computeExpIntDouble(int n, double x) {
    return exp(-x) / (n + 1.0);
}

__global__ void exponentialIntegralDoubleKernel(double* out, int n, int m, double a, double b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (idx < total) {
        int i = idx / m;
        int j = idx % m;
        double x = a + j * (b - a) / m;
        out[idx] = computeExpIntDouble(i + 1, x);
    }
}

void runExponentialIntegralDoubleGPU(double* output, int n, int m, double a, double b) {
    int total = n * m;
    double* d_output;
    cudaMalloc(&d_output, total * sizeof(double));
    
    dim3 blockSize(256);
    dim3 gridSize((total + blockSize.x - 1) / blockSize.x);
    exponentialIntegralDoubleKernel<<<gridSize, blockSize>>>(d_output, n, m, a, b);

    cudaMemcpy(output, d_output, total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}
