#ifndef EXPONENTIALINTEGRALGPU_H
#define EXPONENTIALINTEGRALGPU_H

void runExponentialIntegralFloatGPU(float* output, int n, int m, float a, float b);
void runExponentialIntegralDoubleGPU(double* output, int n, int m, double a, double b);

#endif
