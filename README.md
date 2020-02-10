# cuad
 Automatic Differentiation try with cuda 



Test 

```bash
apollo@in_dev_docker:/apollo/cuad/coordinate_transfer$ nvprof ./mycodes 
==15894== NVPROF is profiling process 15894, command: ./mycodes
prepare time on CUDA(milli secs):298.865
kernel launch time on CUDA(milli secs):0.071102
Total time on CUDA(milli secs):299.031
==15894== Profiling application: ./mycodes
==15894== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.09%  10.976us         1  10.976us  10.976us  10.976us  kernel(double*, double*, double*)
                   13.00%  1.7600us         2     880ns     864ns     896ns  [CUDA memcpy DtoH]
                    5.91%     800ns         1     800ns     800ns     800ns  [CUDA memcpy HtoD]
      API calls:   99.47%  132.64ms         3  44.214ms  4.7030us  132.63ms  cudaMalloc
                    0.24%  325.94us         1  325.94us  325.94us  325.94us  cuDeviceTotalMem
                    0.16%  215.23us        96  2.2410us     239ns  89.546us  cuDeviceGetAttribute
                    0.05%  67.804us         3  22.601us  4.0870us  56.577us  cudaFree
                    0.03%  39.172us         1  39.172us  39.172us  39.172us  cuDeviceGetName
                    0.02%  31.508us         3  10.502us  8.2170us  13.479us  cudaMemcpy
                    0.02%  20.336us         1  20.336us  20.336us  20.336us  cudaLaunchKernel
                    0.00%  5.0260us         1  5.0260us  5.0260us  5.0260us  cuDeviceGetPCIBusId
                    0.00%  2.5500us         3     850ns     220ns  1.7210us  cuDeviceGetCount
                    0.00%  1.2180us         2     609ns     317ns     901ns  cuDeviceGet
                    0.00%     571ns         1     571ns     571ns     571ns  cudaGetLastError
                    0.00%     436ns         1     436ns     436ns     436ns  cuDeviceGetUuid

```