# CUDA Exponential Integral (MAP55616-03)

This repository contains an implementation of the exponential integral function \( E_n(x) \), using both CPU and GPU (CUDA) acceleration. This assignment was completed for the **MAP55616 High Performance Computing** module at **Trinity College Dublin (TCD)**.

## Features
- Supports both `float` and `double` precision modes
- CPU and GPU modes controlled via command-line flags
- Measures and compares execution time of CPU vs GPU
- Full benchmarking across problem sizes up to 20000 × 20000
- Speedup graph generator included (`plot_speedup.py`)

## How to Compile
```bash
make clean
make
```

## How to Run
```bash
# Run both CPU and GPU (default)
./exponentialIntegral.out -n 5000 -m 5000 -t

# GPU only
./exponentialIntegral.out -c -n 8192 -m 8192 -t

# CPU only
./exponentialIntegral.out -g -n 8192 -m 8192 -t
```

### Command-line Flags:
- `-n <int>`: number of orders of E_n
- `-m <int>`: number of x samples in interval [a, b]
- `-a <float>`: start of interval (default 0.0)
- `-b <float>`: end of interval (default 10.0)
- `-c`: skip CPU execution (GPU only)
- `-g`: skip GPU execution (CPU only)
- `-t`: show execution time
- `-v`: verbose mode

## Benchmark Results
| Size (n × m)     | CPU Time (s) | GPU Time (s) | Speedup |
|------------------|--------------|--------------|---------|
| 5000 × 5000      | 0.5289       | 0.5259       | 1.01×   |
| 8192 × 8192      | 1.3160       | 0.3723       | 3.53×   |
| 16384 × 16384    | 5.2538       | 1.4670       | 3.58×   |
| 20000 × 20000    | 8.4552       | 2.3334       | 3.62×   |

## Speedup Plot
To generate the GPU speedup plot:
```bash
python3 plot_speedup.py
```

This will output `speedup_plot.png` showing speedup vs problem size.

## Files
- `main.cpp`: Main driver with CPU and GPU logic
- `exponentialIntegralGPU.cu`: CUDA kernel implementation
- `exponentialIntegralGPU.h`: Function declarations
- `Makefile`: Compile CPU and CUDA code with nvcc
- `plot_speedup.py`: Script to generate speedup plot

## Author
**Seeshuraj Bhoopalan**  
MSc HPC, Trinity College Dublin  
MAP55616-03 Assignment (CUDA Exponential Integral)
