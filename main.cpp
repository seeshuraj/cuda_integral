#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "exponentialIntegralGPU.h"

using namespace std;

// CPU function declarations
float exponentialIntegralFloat(int n, float x);
double exponentialIntegralDouble(int n, double x);
void outputResultsCpu(const vector<vector<float>> &resultsFloatCpu, const vector<vector<double>> &resultsDoubleCpu);
int parseArguments(int argc, char **argv);
void printUsage();

// Globals
bool verbose = false, timing = false, cpu = true;
int maxIterations = 2000000000;
unsigned int n = 10, numberOfSamples = 10;
double a = 0.0, b = 10.0;

int main(int argc, char *argv[]) {
    parseArguments(argc, argv);
    vector<vector<float>> resultsFloatCpu(n, vector<float>(numberOfSamples));
    vector<vector<double>> resultsDoubleCpu(n, vector<double>(numberOfSamples));
    double timeTotalCpu = 0.0;

    // CPU computation
    if (cpu) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (unsigned int i = 1; i <= n; ++i) {
            for (unsigned int j = 1; j <= numberOfSamples; ++j) {
                float x = a + j * (b - a) / numberOfSamples;
                resultsFloatCpu[i - 1][j - 1] = exponentialIntegralFloat(i, x);
                resultsDoubleCpu[i - 1][j - 1] = exponentialIntegralDouble(i, x);
            }
        }
        gettimeofday(&end, NULL);
        timeTotalCpu = ((end.tv_sec + end.tv_usec * 0.000001) - (start.tv_sec + start.tv_usec * 0.000001));
        if (timing) printf("CPU Time: %f seconds\n", timeTotalCpu);
    }

    // GPU computation
    if (!cpu) {
        float *flatFloat = new float[n * numberOfSamples];
        double *flatDouble = new double[n * numberOfSamples];
        vector<vector<float>> resultsFloatGpu(n, vector<float>(numberOfSamples));
        vector<vector<double>> resultsDoubleGpu(n, vector<double>(numberOfSamples));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        runExponentialIntegralFloatGPU(flatFloat, n, numberOfSamples, a, b);
        runExponentialIntegralDoubleGPU(flatDouble, n, numberOfSamples, a, b);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "GPU Time: " << milliseconds / 1000.0 << " seconds\n";

        for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < numberOfSamples; ++j) {
                resultsFloatGpu[i][j] = flatFloat[i * numberOfSamples + j];
                resultsDoubleGpu[i][j] = flatDouble[i * numberOfSamples + j];
            }

        delete[] flatFloat;
        delete[] flatDouble;
    }

    if (verbose && cpu)
        outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);

    return 0;
}

float exponentialIntegralFloat(int n, float x) {
    if (x == 0.0f) return 0.0f;
    return expf(-x) / (n + 1.0f);
}

double exponentialIntegralDouble(int n, double x) {
    if (x == 0.0) return 0.0;
    return exp(-x) / (n + 1.0);
}

void outputResultsCpu(const vector<vector<float>> &resultsFloatCpu, const vector<vector<double>> &resultsDoubleCpu) {
    double x, division = (b - a) / ((double)(numberOfSamples));
    for (unsigned int i = 1; i <= n; ++i) {
        for (unsigned int j = 1; j <= numberOfSamples; ++j) {
            x = a + j * division;
            cout << "CPU==> E" << i << "(" << x << ") = " << resultsDoubleCpu[i - 1][j - 1] << ", ";
            cout << "E" << i << "(" << x << ") = " << resultsFloatCpu[i - 1][j - 1] << endl;
        }
    }
}

int parseArguments(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "cghn:m:a:b:tv")) != -1) {
        switch (c) {
            case 'c': cpu = false; break;
            case 'g': cpu = true; break;
            case 'n': n = atoi(optarg); break;
            case 'm': numberOfSamples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing = true; break;
            case 'v': verbose = true; break;
            case 'h': printUsage(); exit(0);
            default: fprintf(stderr, "Invalid option\n"); printUsage(); return -1;
        }
    }
    return 0;
}

void printUsage() {
    cout << "Usage: ./exponentialIntegral.out [options]\n";
    cout << "  -n <int> : max order Eₙ (default 10)\n";
    cout << "  -m <int> : number of samples (default 10)\n";
    cout << "  -a <val> : interval start (default 0.0)\n";
    cout << "  -b <val> : interval end (default 10.0)\n";
    cout << "  -c       : skip CPU\n";
    cout << "  -g       : skip GPU\n";
    cout << "  -t       : enable timing\n";
    cout << "  -v       : verbose output\n";
}
