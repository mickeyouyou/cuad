nvcc -arch=sm_61 -std=c++11 -I/usr/local/adolc/include -o traceless_cuda traceless_cuda.cu


g++ -std=c++11 -I/usr/local/adolc/include tapeless_scalar.cpp -L/usr/local/adolc/lib64 -ladolc -o tapeless_scalar