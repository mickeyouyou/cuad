#include <cuda.h>
#include <stdio.h>
// main routine
int main() {
  float time;
  cudaEvent_t start, stop;

  for (size_t size = 32; size < 1024 * 1024 * 1024; size *= 2) {
    float* d_Data;
    float* h_Data = new float[size];

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_Data, size * sizeof(float));
    cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (size > 1024 * 1024) {
      printf("Allocation to device: %fms with size %dMB\n", time,
             static_cast<int>((size * sizeof(float)) / (1024 * 1024)));
    } else if (size > 1024) {
      printf("Allocation to device: %fms with size %dKB\n", time,
             static_cast<int>((size * sizeof(float)) / 1024));
    } else {
      printf("Allocation to device: %fms with size %dB\n", time,
             static_cast<int>(size * sizeof(float)));
    }
    delete[] h_Data;
    cudaFree(d_Data);
  }

  return 0;
}