#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int total_size = 1024 * 1024;  // 1MB

void test(int size) {
  double iStart, iElaps;
  int num = total_size / size;
  float *d[num];

  iStart = seconds();
  for (int i = 0; i < num; i++) {
    cudaMalloc((float **)&d[i], size);
  }
  iElaps = seconds() - iStart;
  printf("cudaMalloc(%d) x %d Time elapsed %f sec\n", size, num, iElaps);

  iStart = seconds();
  for (int i = 0; i < num; i++) {
    cudaFree(d[i]);
  }
  iElaps = seconds() - iStart;
  printf("cudaFree(%d) x %d Time elapsed %f sec\n", size, num, iElaps);
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  int size = atoi(argv[1]);
  test(size);

  return (0);
}