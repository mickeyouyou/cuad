#include <cuda.h>
#include <chrono>
#include <iostream>
#include "adolc/adoublecuda.h"

using namespace std;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s:%d, ", __FILE__, __LINE__);
    printf("Code: %d, message: %s\n", result, cudaGetErrorString(result));
  }
  return result;
}

__global__ void kernel(double* inx, double* outy, double* outderiv) {
  // const int index = threadIdx.x;
  const int index1 = threadIdx.y;
  // const int index2 = blockIdx.x;
  // const int dim = blockDim.x * blockDim.y;
  // const int index3 = blockDim.x;

  int col = threadIdx.x + blockDim.x * blockIdx.x;  // width
  int row = threadIdx.y + blockIdx.y * blockDim.y;  // height
  int M = 8;
  if (col < 3 && row < M) {
    int global_index = row * 3 + col;
    // printf("Col: %d, Row:%d, global_index: %d\n", col, row, global_index);

    // Declare dependent and independent variables as adoubles
    adtlc::adouble y[3];
    adtlc::adouble x[3];

    // Read out point for function evaluation
    // for (int i = 0; i < 3; i++) x[i] = inx[index2 * dim + index * 3 + i];
    x[global_index] = inx[global_index];

    // Set direction for calculation of derivatives
    x[index1].setADValue(1);

    // Function evaluation
    y[0] = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    y[1] = atan(sqrt(x[0] * x[0] + x[1] * x[1]) / x[2]);
    y[2] = atan(x[1] / x[0]);

    // for (int i = 0; i < 3; i++)
    // outy[(index2 * index3 + index) * 3 + i] = y[i].getValue();

    outy[global_index] = y[global_index].getValue();
    // for (int i = 0; i < 3; i++)
    // outderiv[(index2 * dim + index * 3 + index1) * 3 + i] =
    // y[i].getADValue();
    outderiv[global_index] = y[global_index].getADValue();
  }
}

cudaError_t kernellaunch(double* inx, double* outy, double* outderiv, int n) {
  // Two dimensional (M/Blocks) x 3 blocks
  dim3 threadsPerBlock(8, 3);
  // Create 16 blocks
  dim3 Blocks(n / threadsPerBlock.x + 1, n / threadsPerBlock.y + 1);

  // Call kernel function with 16 blocks with (M/Blocks) x 3 threads per block
  kernel<<<Blocks, threadsPerBlock>>>(inx, outy, outderiv);
  cudaError_t cudaErr = cudaGetLastError();

  return cudaErr;
}

int main() {
  int M = 8;
  double* deriv = new double[9 * M];
  double* y = new double[3 * M];
  double* x = new double[3 * M];

  // Initialize x_i
  for (int k = 0; k < M; k++) {
    for (int i = 0; i < 3; ++i) x[k * 3 + i] = i + 1 / (k + 1);
  }

  // for (int k = 0; k < M; k++) {
  //   for (int i = 0; i < 3; ++i) {
  //     printf("x[%d]: %f.3\n", k * 3 + i, x[k * 3 + i]);
  //   }
  // }

  auto t1 = std::chrono::steady_clock::now();

  // Allocate array for independent and dependent variables and Jacobian
  // matrices on GPU
  // checkCuda(cudaFree(0));
  double* devx;
  checkCuda(cudaMalloc((void**)&devx, 3 * M * sizeof(double)));

  double* devy;
  checkCuda(cudaMalloc((void**)&devy, 3 * M * sizeof(double)));

  double* devderiv;
  checkCuda(cudaMalloc((void**)&devderiv, 3 * 3 * M * sizeof(double)));

  // Copy values of independent variables from host to GPU
  cudaMemcpy(devx, x, sizeof(double) * 3 * M, cudaMemcpyHostToDevice);
  cudaError_t error;

  auto t2 = std::chrono::steady_clock::now();
  std::cout << "prepare time on CUDA(milli secs):"
            << double(std::chrono::duration<double>(t2 - t1).count() * 1000)
            << std::endl;

  // Call function to specify amount of blocks and threads to be used
  error = kernellaunch(devx, devy, devderiv, M);
  // cudaDeviceSynchronize();
  auto t3 = std::chrono::steady_clock::now();
  std::cout << "kernel launch time on CUDA(milli secs):"
            << double(std::chrono::duration<double>(t3 - t2).count() * 1000)
            << std::endl;

  // error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Code: %d, Message: %s\n", error, cudaGetErrorString(error));
  }

  // Copy values of dependent variables and Jacobian matrices from GPU to host
  cudaMemcpy(y, devy, sizeof(double) * 3 * M, cudaMemcpyDeviceToHost);

  cudaMemcpy(deriv, devderiv, sizeof(double) * M * 3 * 3,
             cudaMemcpyDeviceToHost);

  cudaFree(devx);
  cudaFree(devy);
  cudaFree(devderiv);

  auto t4 = std::chrono::steady_clock::now();
  std::cout << "Total time on CUDA(milli secs):"
            << double(std::chrono::duration<double>(t4 - t1).count() * 1000)
            << std::endl;
}
