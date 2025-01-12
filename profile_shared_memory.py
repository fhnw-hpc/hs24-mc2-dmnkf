import os
import math
import numpy as np
from numba import cuda, float32
import time

def create_test_matrix(N):
    """Create a random NxN matrix and its SVD components"""
    A = np.random.randn(N, N).astype(np.float32)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    k = min(u.shape[1], vt.shape[0]) // 3
    return u, s, vt, k

@cuda.jit
def shared_memory_kernel(u, s, vt, C, k):
    """
    SVD reconstruction kernel using shared memory (from 5.2.2)
    C[i,j] = sum(U[i,r] * s[r] * V^T[r,j]) for r in range(k)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if i >= C.shape[0] or j >= C.shape[1]:
        return
        
    # Shared memory for tiles
    tile_size = 16  # Must match threads_per_block
    u_tile = cuda.shared.array(shape=(16, 16), dtype=float32)
    vt_tile = cuda.shared.array(shape=(16, 16), dtype=float32)
    s_tile = cuda.shared.array(shape=16, dtype=float32)
    
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    
    acc = 0.0
    
    # Process k in tiles
    for tile_start in range(0, k, tile_size):
        # Load U tile
        k_idx = tile_start + local_j
        if i < u.shape[0] and k_idx < k:
            u_tile[local_i, local_j] = u[i, k_idx]
        else:
            u_tile[local_i, local_j] = 0.0
            
        # Load V^T tile
        k_idx = tile_start + local_i
        if k_idx < k and j < vt.shape[1]:
            vt_tile[local_i, local_j] = vt[k_idx, j]
        else:
            vt_tile[local_i, local_j] = 0.0
            
        # Load singular values
        if local_j == 0 and (tile_start + local_i) < k:
            s_tile[local_i] = s[tile_start + local_i]
        
        cuda.syncthreads()
        
        # Compute partial sum for this tile
        for kk in range(min(tile_size, k - tile_start)):
            acc += u_tile[local_i, kk] * s_tile[kk] * vt_tile[kk, local_j]
            
        cuda.syncthreads()
    
    C[i, j] = acc

def profile_shared_memory():
    matrix_size = 1024  # Fixed size for testing different thread configurations
    
    # Block configurations to test - powers of 2, respecting max 1024 threads
    block_configs = [
        (x, y) for x, y in [
            (2**i, 2**j) 
            for i in range(1,6) 
            for j in range(1,9)
        ] if x * y <= 1024 and x <= y
    ]
    
    print(f"\nProfiling shared memory kernel for size {matrix_size}x{matrix_size}")
    
    # Create test data
    u, s, vt, k = create_test_matrix(matrix_size)
    
    # Allocate device memory
    u_dev = cuda.to_device(u)
    s_dev = cuda.to_device(s)
    vt_dev = cuda.to_device(vt)
    C_dev = cuda.device_array((matrix_size, matrix_size), dtype=np.float32)
    
    results = []
    
    for block_size in block_configs:
        # Calculate grid dimensions
        blocks_x = math.ceil(matrix_size / block_size[0])
        blocks_y = math.ceil(matrix_size / block_size[1])
        blocks_per_grid = (blocks_x, blocks_y)
        
        print(f"\nTesting block size: {block_size}")
        print(f"Grid size: {blocks_per_grid}")
        
        # Warmup
        shared_memory_kernel[blocks_per_grid, block_size](u_dev, s_dev, vt_dev, C_dev, k)
        cuda.synchronize()
        
        # Timed run
        start = cuda.event()
        end = cuda.event()
        
        start.record()
        shared_memory_kernel[blocks_per_grid, block_size](u_dev, s_dev, vt_dev, C_dev, k)
        end.record()
        end.synchronize()
        
        kernel_time = cuda.event_elapsed_time(start, end)
        print(f"Execution time: {kernel_time:.2f} ms")
        
        results.append({
            'block_size': block_size,
            'grid_size': blocks_per_grid,
            'time': kernel_time
        })
    
    # Print summary of best configurations
    print("\nTop 5 fastest configurations:")
    sorted_results = sorted(results, key=lambda x: x['time'])
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. Block size: {result['block_size']}, "
              f"Grid size: {result['grid_size']}, "
              f"Time: {result['time']:.2f} ms")
    
    # Clean up
    del u_dev, s_dev, vt_dev, C_dev

if __name__ == "__main__":
    profile_shared_memory() 