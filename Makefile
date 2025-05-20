NVCC = /usr/local/cuda-12.8/bin/nvcc
CXX = /usr/bin/g++  # use system g++ for fallback if needed

CXXFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3

EXEC = exponentialIntegral.out

CU_SRC = exponentialIntegralGPU.cu
CPP_SRC = main.cpp
OBJ = main.o exponentialIntegralGPU.o

all: $(EXEC)

$(EXEC): $(OBJ)
	$(NVCC) $(OBJ) -o $@

# Use nvcc to compile main.cpp (because it includes CUDA headers)
main.o: main.cpp exponentialIntegralGPU.h
	$(NVCC) $(NVCCFLAGS) -c main.cpp -o main.o

exponentialIntegralGPU.o: exponentialIntegralGPU.cu exponentialIntegralGPU.h
	$(NVCC) $(NVCCFLAGS) -c exponentialIntegralGPU.cu -o exponentialIntegralGPU.o

clean:
	rm -f *.o $(EXEC))
