# Compiler setup
NVCC = nvcc
CXX = g++
CXXFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3

# Output binary
EXEC = exponentialIntegral.out

# Source and object files
CU_SRC = exponentialIntegralGPU.cu
CPP_SRC = main.cpp
OBJ = main.o exponentialIntegralGPU.o

# Build rules
all: $(EXEC)

$(EXEC): $(OBJ)
	$(NVCC) $(OBJ) -o $@

main.o: main.cpp exponentialIntegralGPU.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

exponentialIntegralGPU.o: exponentialIntegralGPU.cu exponentialIntegralGPU.h
	$(NVCC) $(NVCCFLAGS) -c exponentialIntegralGPU.cu -o exponentialIntegralGPU.o

clean:
	rm -f *.o $(EXEC)