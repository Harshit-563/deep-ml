#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a,
                           const float* b,
                           float* c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

std::vector<float> vector_add(const std::vector<float>& a,
                              const std::vector<float>& b) {
    int n = a.size();

    std::vector<float> v(n);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a.data(),
               n * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_b, b.data(),
               n * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(v.data(),
               d_c,
               n * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return v;
}