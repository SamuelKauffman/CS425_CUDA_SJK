#include <iostream>
#include <cuda_runtime.h>

#define N 10 // Size of matrix and vector

// Matrix multiplication function
__global__ void matrixVectorMultiply(int *M, int *V) {
    // Getting row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Multiply Mij by Vj
    if (row < N && col < N) {
        int idx = row * N + col;
        M[idx] *= V[col];        
    }
}

int main() {
    int hostM[N * N], hostV[N];

    // Initializing the vector 1-10
    for (int j = 0; j < N; ++j) {
        hostV[j] = j + 1;
    }

    // Initialize matrix so every row equals V
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            hostM[i * N + j] = j + 1;
        }
    }

    // Allocate memory
    int *devM, *devV;
    cudaMalloc((void**)&devM, N * N * sizeof(int));
    cudaMalloc((void**)&devV, N * sizeof(int));

    // Copy data
    cudaMemcpy(devM, hostM, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devV, hostV, N * sizeof(int), cudaMemcpyHostToDevice);

    // Assign grid and block dimensions
    dim3 blockDim(10, 10);
    dim3 gridDim(1, 1);

    // Launch the kernel
    matrixVectorMultiply<<<gridDim, blockDim>>>(devM, devV);

    // Copy result back to host
    cudaMemcpy(hostM, devM, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print original vector
    std::cout << "Vector V:\n";
    for (int j = 0; j < N; ++j) {
        std::cout << hostV[j] << " ";
    }
    std::cout << "\n\n";

    // Print multiplied matrix
    std::cout << "Updated Matrix M:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostM[i * N + j] << "\t";
        }
        std::cout << "\n";
    }

    // Free device memory
    cudaFree(devM);
    cudaFree(devV);

    return 0;
}

