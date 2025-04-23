#include <iostream>
#include <cuda_runtime.h>

#define N 10 

__global__ void matrixVectorDotProduct(const int *M, const int *V, int *R) {
    // Shared memory for reduction. one row, N elements
    __shared__ int partialSum[N]; 

    // Thread index in a block and block index that relates to the matrix row
    int tid = threadIdx.x;       
    int row = blockIdx.x;         

    // Each thread multiplies one element
    if (tid < N) {
        int idx = row * N + tid;
        partialSum[tid] = M[idx] * V[tid]; 
    }

    __syncthreads();

    // Parallel reduction
    for (int stride = N / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 in each block holds results
    if (tid == 0) {
        R[row] = partialSum[0];
    }
}

int main() {
    int hostM[N * N], hostV[N], hostR[N];

    // Fill vector 1-10
    for (int i = 0; i < N; ++i) {
        hostV[i] = i + 1;
        for (int j = 0; j < N; ++j) {
            hostM[i * N + j] = j + 1;
        }
    }

    // Allocate device memory
    int *devM, *devV, *devR;
    cudaMalloc((void**)&devM, N * N * sizeof(int));
    cudaMalloc((void**)&devV, N * sizeof(int));
    cudaMalloc((void**)&devR, N * sizeof(int));

    // Copy data
    cudaMemcpy(devM, hostM, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devV, hostV, N * sizeof(int), cudaMemcpyHostToDevice);

    // N blocks, each with N threads. one per matrix row
    matrixVectorDotProduct<<<N, N>>>(devM, devV, devR);

    // Copy result
    cudaMemcpy(hostR, devR, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print vector
    std::cout << "Vector V:\n";
    for (int i = 0; i < N; ++i) std::cout << hostV[i] << " ";
    std::cout << "\n\n";

    // Print matrix
    std::cout << "Matrix M:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) std::cout << hostM[i * N + j] << "\t";
        std::cout << "\n";
    }

    // Print result
    std::cout << "\nResult Vector R (M x V):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << hostR[i] << " ";
    }
    std::cout << "\n";

    // Clean up
    cudaFree(devM);
    cudaFree(devV);
    cudaFree(devR);

    return 0;
}
