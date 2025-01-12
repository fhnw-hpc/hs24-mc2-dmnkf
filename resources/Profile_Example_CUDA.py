# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="NoV0YWpWeX0Y" outputId="709250cb-0d42-4bbc-9708-b8aefef07c54"
# !nvidia-smi

# + colab={"base_uri": "https://localhost:8080/"} id="HVVwNqwjeYL0" outputId="da02e775-bfe9-427c-d9f7-82674e8feae5"
# %%writefile reduce.cu
/* This program will performe a reduce vectorA (size N)
* with the + operation.
+---------+ 
|111111111| 
+---------+
     |
     N

vectorA   = all Ones
N = Sum of vectorA
*/
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


// CUDA macro wrapper for checking errors
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


// CPU reduce
void reduce(int* vectorA, int* sum, int size)
{
    sum[0] = 0;
    for (int i = 0; i < size; i++)
        sum[0] += vectorA[i];
}


// EXERCISE
// Read: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
// Implement the reduce kernel based on the information
// of the Nvidia blog post.
// Implement both options, using no shared mem at all but global atomics
// and using shared mem for the seconds recution phase.
__global__ void cudaEvenFasterReduceAddition() {
    //ToDo
}


// Already optimized reduce kernel using shared memory.
__global__ void cudaReduceAddition(int* vectorA, int* sum)
{
    int globalIdx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int shmArray[];

    shmArray[threadIdx.x] = vectorA[globalIdx];
    shmArray[threadIdx.x + blockDim.x] = vectorA[globalIdx + blockDim.x];

    for (int stride = blockDim.x; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmArray[threadIdx.x] += shmArray[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[blockIdx.x] = shmArray[0];
    }
}


// Compare result vectors
int compareResultVec(int* vectorCPU, int* vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "No errors. All good!" << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


int main(void)
{
    // Define the size of the vector: 1048576 elements
    const int N = 1 << 20;
    const int NBR_BLOCK = 512;

    // Allocate and prepare input
    int* hostVectorA = new int[N];
    int hostSumCPU[1];
    int hostSumGPU[1];
    for (int i = 0; i < N; i++) {
        hostVectorA[i] = 1;
    }

    // Alloc N times size of int at address of deviceVector[A-C]
    int* deviceVectorA;
    int* deviceSum;
    gpuErrCheck(cudaMalloc(&deviceVectorA, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceSum, NBR_BLOCK* sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVectorA, hostVectorA, N * sizeof(int), cudaMemcpyHostToDevice));

    // Run the vector kernel on the CPU
    reduce(hostVectorA, hostSumCPU, N);

    // Run kernel on all elements on the GPU
    cudaReduceAddition <<<NBR_BLOCK, 1024, 2 * 1024 * sizeof(int)>>> (deviceVectorA, deviceSum);
    gpuErrCheck(cudaPeekAtLastError());
    cudaReduceAddition <<<1, NBR_BLOCK / 2, NBR_BLOCK * sizeof(int) >> > (deviceSum, deviceSum);
    gpuErrCheck(cudaPeekAtLastError());

    // Copy the result stored in device_y back to host_y
    gpuErrCheck(cudaMemcpy(hostSumGPU, deviceSum, sizeof(int), cudaMemcpyDeviceToHost));

    // Check for errors
    auto isValid = compareResultVec(hostSumCPU, hostSumGPU, 1);

    // Free memory on device
    gpuErrCheck(cudaFree(deviceVectorA));
    gpuErrCheck(cudaFree(deviceSum));

    // Free memory on host
    delete[] hostVectorA;

    return isValid;
}


# + [markdown] id="sXRAf_-9lub9"
# **Attention:** If you get a K80, you should compile it like this:
# `!nvcc -arch=sm_37 -o reduce reduce.cu`

# + id="4lLHNUzHef9q"
# !nvcc -o reduce reduce.cu

# + colab={"base_uri": "https://localhost:8080/"} id="57IzTD97eqDX" outputId="6aceadaf-3698-4c1a-e9d9-0d69d0c4535c"
# !./reduce

# + [markdown] id="7p-YpNLPnMEV"
# **Profiling on old GPU (K80)**
#
# *   `!nvprof --print-gpu-trace ./reduce`
# *   `!nvprof --analysis-metrics -o reduce_out.nvprof ./reduce`
# *   --> Use the Visual Profiler
#
# **Profiling on newer GPUs**
#
# *   `!nsys profile -f true -o reduce_out -t cuda ./reduce`
# *   `!nsys stats --report gputrace reduce.qdrep`
# *   `!nsys stats reduce.qdrep`
# *   `!ncu -f -o reduce --set full ./reduce`
# *   --> Nvidia Nsight Tools

# + [markdown] id="oPbbe5AaUU4Y"
# Example:

# + colab={"base_uri": "https://localhost:8080/"} id="8FszJbI5nVjz" outputId="7e836e7a-2ad7-41ab-99a2-906a6fec252f"
# !nvprof --print-gpu-trace ./reduce

# + colab={"base_uri": "https://localhost:8080/"} id="dZrPgfUgn1DX" outputId="5459d5d2-fdad-4f19-b5c5-49f516d13d1a"
# !nvprof --analysis-metrics -o reduce_out.nvprof ./reduce

# + [markdown] id="PTzsnXS4UYtc"
# Use the file *reduce_out.nvprof* with the visual profiler of Nvidia, which you have locally installed.

# + id="lrwzKmITUmJp"
# !nsys profile -f true --stats=true -o reduce_out -t cuda ./reduce
# -

# !nsys stats reduce_out.qdrep


