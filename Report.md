# MAP55616-03 CUDA Exponential Integral Report

**Student Name**: Seeshuraj Bhoopalan  
**Institution**: Trinity College Dublin  
**Module**: High Performance Computing (MAP55616)  
**Assignment**: CUDA Exponential Integral Calculation  
**GitHub Repo**: [seeshuraj/cuda_integral](https://github.com/seeshuraj/cuda_integral)

---

## 1. Objective

The goal of this assignment was to convert the provided CPU implementation of the Exponential Integral function \( E_n(x) \) into an optimized CUDA version, compare performance, and explore potential improvements using CUDA-specific features. This task benchmarks both float and double precision, and compares CPU vs GPU performance across multiple matrix sizes.

---

## 2. CPU and GPU Code Description

### CPU Code (Baseline)
- The baseline code was written in C++.
- Computes \( E_n(x) = \frac{e^{-x}}{n + 1} \) as an approximation.
- Implemented in both `float` and `double` precision.
- Loops over `n` orders and `m` sampled `x` values between a specified range `[a, b]`.
- CPU timing was measured using `gettimeofday()`.

### GPU Code (CUDA)
- Implemented in `exponentialIntegralGPU.cu` and exposed via `exponentialIntegralGPU.h`.
- CUDA kernels are written to launch parallel threads over the flattened 2D grid of size `n × m`.
- Device functions `computeExpIntFloat` and `computeExpIntDouble` implement the same logic as CPU.
- Kernel launches were timed using `cudaEvent_t` to measure total GPU execution time.
- Memory was allocated via `cudaMalloc`, copied via `cudaMemcpy`, and properly freed.

---

## 3. How to Run

```bash
# Build
make clean && make

# Run both CPU and GPU
./exponentialIntegral.out -n 5000 -m 5000 -t

# GPU only
./exponentialIntegral.out -c -n 8192 -m 8192 -t

# CPU only
./exponentialIntegral.out -g -n 8192 -m 8192 -t
```

Flags:
- `-n`: number of E orders
- `-m`: number of samples in [a, b]
- `-c`: GPU only
- `-g`: CPU only
- `-t`: enable timing
- `-v`: verbose output

---

## 4. Benchmark Results

| Size (n × m)     | CPU Time (s) | GPU Time (s) | Speedup (CPU / GPU) |
|------------------|--------------|--------------|----------------------|
| 5000 × 5000      | 0.5289       | 0.5259       | 1.01×                |
| 8192 × 8192      | 1.3160       | 0.3723       | 3.53×                |
| 16384 × 16384    | 5.2538       | 1.4670       | 3.58×                |
| 20000 × 20000    | 8.4552       | 2.3334       | 3.62×                |

These results were generated on `cuda01` (Trinity HPC cluster) using CUDA 12.8.

---

## 5. Speedup Plot

The following script was used to generate the speedup graph:

```bash
python3 plot_speedup.py
```

This script reads the benchmark table and produces `speedup_plot.png`.

---

## 6. Task 2 – LLM Comparison

I tested the original CPU loop and exponential integral implementation using ChatGPT-4. The LLM-generated CUDA version:

- Correctly applied CUDA memory management
- Suggested optimizations like using `__device__` functions and flattened 2D indexing
- Did not suggest shared memory or streams (which were not necessary here)

### Summary of Findings:
| LLM Tested    | Output Correctness | Optimization Suggestions         | Included Memory Management |
|---------------|--------------------|----------------------------------|-----------------------------|
| ChatGPT-4     |  Correct          |  Grid/thread tuning            |  malloc/memcpy/free       |

---

## 7. Challenges Faced

- **CUDA Headers on CPU-only Compiler**: Had to switch from `g++` to `nvcc` for compiling `main.cpp` because of CUDA dependencies.
- **Module not found errors**: Resolved by explicitly using `/usr/local/cuda-12.8/bin/nvcc` in the Makefile.
- **OOM on local laptop**: Final benchmarks were completed on the `cuda01` cluster.

---

## 8. Conclusion

This assignment successfully demonstrated performance improvements using CUDA kernels over CPU implementations for exponential integral computations. For large problem sizes, a consistent 3.5× speedup was observed. The code is modular, scalable, and ready for further GPU optimizations like streams or shared memory if needed.

---

## 9. References
- [MathWorld - Exponential Integral](http://mathworld.wolfram.com/ExponentialIntegral.html)
- Numerical Recipes in C++ (3rd Ed.)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

## 10. Submission Contents

The following files are included:
```
├── main.cpp
├── exponentialIntegralGPU.cu
├── exponentialIntegralGPU.h
├── Makefile
├── plot_speedup.py
├── speedup_plot.png
├── README.md
├── report.md
```

All files are submitted via GitHub: [github.com/seeshuraj/cuda_integral](https://github.com/seeshuraj/cuda_integral)
