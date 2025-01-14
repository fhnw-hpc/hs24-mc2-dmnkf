import argparse
import math
import numpy as np
from numba import cuda, float32


def get_kernel_tiled(BM, BN):
    
    BK = int(BM / 2)
    TM = int(BM / 4)
    TN = int(BN / 4)

    print(f"BM: {BM}, BN: {BN}, BK: {BK}, TM: {TM}, TN: {TN}")

    @cuda.jit
    def svd_reconstruct_tiled(U, S, Vt, out, M, N, K):
        """
        estimated time spent making this work dynamically: 8 hours
        Huge thanks to https://siboehm.com/articles/22/CUDA-MMM
        """

        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        tx = cuda.threadIdx.x
        
        # This thread is responsible for TM x TN sub-results in the final C
        # We also define how many threads per block in x-dimension:
        # if total threads in a block = (BM*BN)/(TM*TN) we can find local row/col
        # But we'll keep it simpler: each block is BM x BN results, each thread produces TM x TN sub-result
        # -> we define "threadRow" & "threadCol" in that 2D sub-block:
        threads_per_block = (BM * BN) // (TM * TN)
        # safety check or assumption:
        if threads_per_block != cuda.blockDim.x:
            return  # mismatch: user must set blockDim.x to threads_per_block

        # The row "tile" in the block for this thread:
        threadRow = tx // (BN // TN)
        threadCol = tx %  (BN // TN)

        # Output tile base index:
        row_start = by * BM
        col_start = bx * BN

        # Shared memory for chunk of U, V^T
        # U-shape: BM x BK, V^T-shape: BK x BN
        U_shared = cuda.shared.array(shape=(BM, BK), dtype=float32)
        Vt_shared = cuda.shared.array(shape=(BK, BN), dtype=float32)

        # each thread accumulates TM x TN partial results
        result = cuda.local.array(shape=(TM, TN), dtype=float32)
        for r in range(TM):
            for c in range(TN):
                result[r, c] = 0.0


        # for each block tile in the K dimension
        for k_tile in range(0, K, BK):
            # each thread loads some portion of the BM x BK chunk
            load_x = tx
            # we'll stride along BM direction in increments of blockDim.x so downward 
            while load_x < BM:
                # go through the BK dimension (across the columns) ---> 
                for kk in range(BK):
                    real_k = k_tile + kk
                    if real_k < K and (row_start + load_x) < M:
                        U_shared[load_x, kk] = U[row_start + load_x, real_k]
                    else:
                        U_shared[load_x, kk] = 0.0
                load_x += threads_per_block

            # load the tile of V^T into Bs
            load_y = tx
            # go through the BN dimension (across the columns) of V^T so to the right
            while load_y < BN:
                # go through the BK dimension (across the rows) of V^T so to the bottom
                for kk in range(BK):
                    real_k = k_tile + kk
                    if real_k < K and (col_start + load_y) < N:
                        # we allocate the S tile here as well to save storage
                        Vt_shared[kk, load_y] = Vt[real_k, col_start + load_y] * S[real_k]
                    else:
                        Vt_shared[kk, load_y] = 0.0
                load_y += threads_per_block

            cuda.syncthreads()

            # compute partial sums for TM x TN sub-block in registers
            #  dotIdx is always across bk so inner dim
            for dotIdx in range(BK):
                # first row wise 
                for i in range(TM):
                    # row in As
                    rowA = threadRow * TM + i
                    aVal = U_shared[rowA, dotIdx]
                    # then col wise
                    for j in range(TN):
                        # col in Bs
                        colB = threadCol * TN + j
                        # calculate the partial sum for all the rows of that col in Bs within a thread
                        result[i, j] += aVal * Vt_shared[dotIdx, colB]

            cuda.syncthreads()

        # store results to global memory
        for i in range(TM):
            out_row = row_start + (threadRow * TM) + i
            if out_row < M:
                for j in range(TN):
                    out_col = col_start + (threadCol * TN) + j
                    if out_col < N:
                        out[out_row, out_col] = result[i, j]
    return svd_reconstruct_tiled



def main():
    parser = argparse.ArgumentParser(
        description="Tile-based GPU partial SVD-like reconstruction with dynamic shared memory + manual indexing."
    )
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension N.")
    # we take the dim based on jupyter notebook best performance
    parser.add_argument("--blockDimX", type=int, default=16, help="Block dimension X.")
    parser.add_argument("--blockDimY", type=int, default=64, help="Block dimension Y.")
    parser.add_argument("--rank", type=int, default=512, help="Partial rank k.")
    args = parser.parse_args()

    N = args.size
    bx = args.blockDimX
    by = args.blockDimY
    k = args.rank

    if bx < 4 or by < 4:
        raise ValueError("Block size must be at least 4x4, but should really be much bigger anyway...")

    U_host = np.random.randn(N, N).astype(np.float32)
    S_host = np.random.randn(N).astype(np.float32)
    VT_host = np.random.randn(N, N).astype(np.float32)

    u_device = cuda.to_device(U_host)
    s_device = cuda.to_device(S_host)
    vt_device = cuda.to_device(VT_host)
    C_device = cuda.device_array((N, N), dtype=np.float32)
    
    M, _ = U_host.shape
    _, N = VT_host.shape
    # Tile sizes: BM=16, BN=16 => we want grid = ceil(M/16, N/16) blocks
    BM = int(bx)
    BN = int(by)
    blocks_x = math.ceil(N / BN)
    blocks_y = math.ceil(M / BM)
    
    threads_per_block = (BM * BN) // (BM / 4 * BN / 4)   # = (16*16)/(4*4) = 16

    print("\n--- Padded Shared-Memory Tile Kernel (Manual Indexing) ---")
    print(f"Matrix size: {N} x {N}")
    print(f"BlockDim  : ({bx}, {by})")
    print(f"Rank (k)  : {k}")
    print(f"Sub block dims: {BM/4} x {BN/4}")
    print(f"Threads per block: {threads_per_block}")
    print("-----------------------------------------------------------")


    kernel = get_kernel_tiled(BM, BN)
    grid_dim = (blocks_x, blocks_y)
    block_dim = (int(threads_per_block),)
    kernel[grid_dim, block_dim](u_device, s_device, vt_device, C_device, M, N, k)
    cuda.synchronize()
    
    start_event = cuda.event()
    stop_event = cuda.event()

    # Timed run
    start_event.record()
    kernel[grid_dim, block_dim](u_device, s_device, vt_device, C_device, M, N, k)
    cuda.synchronize()
    stop_event.record()
    
    # validate the result
    C_cpu = np.dot(np.dot(U_host[:N, :k], np.diag(S_host[:k])), VT_host[:k, :N])
    assert np.allclose(C_device, C_cpu, atol=1e-3), "Result mismatch"

    elapsed_ms = cuda.event_elapsed_time(start_event, stop_event)
    print(f"[GPU] Partial reconstruction time: {elapsed_ms / 1000:.5f} s\n")


if __name__ == "__main__":
    main()