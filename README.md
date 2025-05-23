# CS425_CUDA_SJK

# Assignment 1: Matrix Vector Multiply

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
```

# Assignment 2: Matrix Vector Dot Product Multiplication

## Description

This program calculates the matrix-vector product `R = M × V` using CUDA.

- `M` is a 10×10 matrix with all rows `[1, 2, ..., 10]`.
- `V` is a 10-element vector `[1, 2, ..., 10]`.
- Each element of result vector `R` is the dot product of a row in `M` and vector `V`.

## Threads

- **Grid**: 10 blocks. 1 per matrix row
- **Block**: 10 threads 1 per matrix column

Each block computes one dot product. Threads multiply one element from the matrix row with a vector element.

