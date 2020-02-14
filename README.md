# cuad
 Automatic Differentiation try with cuda 


1. Run in apollo docker image.
 
 ## Reference Performance Comparison Test Data

Jacobian Matrix Computation Based On ADOLC

1. 坐标转换，使用数据量为1024的数据点，输入变量数3，输出变量数3；
2. 耗时仅包含求导的计算过程部分，不包含结果的获取与打印输出，保持统一标准。

Coordinate Transform Example Codes:
- CPU version: coordinate_transform/tapeless_scalar.cpp
- CUDA version : coordinate_transform/my_codes.cu

CPU (first row) and CUDA(second row) test result table:

| | | | | | | | | | |
| :-----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| 1.82727 | 1.9159 | 2.38486 | 2.51743 | 2.193 | 6.74031 | 2.35056 | 2.28936 | 1.72139 | 6.13136 |
|0.355435 | 0.31505 | 0.303447 | 0.30466 | 0.397284 | 0.399476 | 0.317374 | 0.341977 | 0.304162 | 0.306204|

**The average execution time of CPU is 2.787844ms, some value of  CUDA is 0.3345069ms, Speed up times 8.33.**
