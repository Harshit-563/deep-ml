#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        out[i]=i;
    }

}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n ints
    // 2. launch the kernel with enough threads to cover n
    std::vector<int> v(n);
    int* d_out;
    cudaMalloc(&d_out, n * sizeof(int));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    index_kernel<<<blocks, threads>>>(d_out,n);
    cudaMemcpy(v.data(), d_out, n * sizeof(int),
           cudaMemcpyDeviceToHost);
    // 3. copy the result back to the host and return it
    ;
    cudaFree(d_out);
    return v;
}
