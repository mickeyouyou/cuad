nvcc -arch=sm_61 -std=c++11 -I/usr/local/adolc/include -o mycodes my_codes.cu

g++ -std=c++11 -I/usr/local/adolc/include tapeless_scalar.cpp -L/usr/local/adolc/lib64 -ladolc -o tapeless_scalar