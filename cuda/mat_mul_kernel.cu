#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void mat_mul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        float sum = 0;
        for (int l = 0; l < k; l++) {
            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
    }
}

extern "C" void launch_mat_mul(float *A, float *B, float *C, int m, int k, int n) {
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);

    const int threads = 32;
    dim3 dim_block(threads, threads);
    dim3 dim_grid((n + threads - 1) / threads, (m + threads - 1) / threads);

    mat_mul_gpu<<<dim_grid, dim_block>>>(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}