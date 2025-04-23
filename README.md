# CS425_CUDA_SJK

# Matrix Vector Multiply

## Description

This program multiplies each element of a 10×10 matrix `M` with the corresponding element of a 10-element vector `V`, column-wise. That is: M[i][j] = M[i][j] * V[j]

## Grid and Block Dimensions

**Each thread handles one matrix element.**
- `Block size: 10×10`

**100 threads are needed in total.**
- `Grid size: 1×1`

This makes it so that all 100 elements in a 10x10 matrix are covered by the thread block.

## Thread Mapping

Each thread computes its own `(row, col)` index using:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;


