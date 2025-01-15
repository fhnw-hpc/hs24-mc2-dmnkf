import argparse
import math
import numpy as np
from numba import cuda, float32

def get_kernel_tiled_padded(BM, BN):
    # Hard-code the tile dims
    BK = int(BM / 2)
    TM = BM // 8  # => 2
    TN = BN // 8  # => 8

    # Extra columns for padding
    BKpad = BK + 1  # 8 + 1 = 9
    BNpad = BN + 1  # 64 + 1 = 65

    print(f"BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}; Shared arrays have BKpad={BKpad}, BNpad={BNpad}")

    @cuda.jit
    def svd_reconstruct_tiled_padded(U, S, Vt, out, M, N, K):
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        tx = cuda.threadIdx.x

        threads_per_block = (BM * BN) // (TM * TN)
        if threads_per_block != cuda.blockDim.x:
            return

        # 2D sub-block indexing
        threadRow = tx // (BN // TN)  # => tx // 8
        threadCol = tx %  (BN // TN)  # => tx % 8

        row_start = by * BM
        col_start = bx * BN

        # Allocate padded shared arrays:
        # - We'll store up to [BM, BK], leaving the last col as padding
        U_shared  = cuda.shared.array((BM, BKpad), dtype=float32)
        # - We'll store up to [BK, BN], leaving the last col in BN for padding
        Vt_shared = cuda.shared.array((BK, BNpad), dtype=float32)

        # local accumulation
        result = cuda.local.array((TM, TN), dtype=float32)
        for i in range(TM):
            for j in range(TN):
                result[i, j] = 0.0

        # Tiling over the K dimension in increments of BK
        for k_tile in range(0, K, BK):
            # Load BM rows from U
            load_x = tx
            while load_x < BM:
                for kk in range(BK):
                    real_k = k_tile + kk
                    if real_k < K and (row_start + load_x) < M:
                        U_shared[load_x, kk] = U[row_start + load_x, real_k]
                    else:
                        U_shared[load_x, kk] = 0.0
                # For the padded column
                U_shared[load_x, BK] = 0.0
                load_x += threads_per_block

            # Load BN columns from Vt
            load_y = tx
            while load_y < BN:
                for kk in range(BK):
                    real_k = k_tile + kk
                    if real_k < K and (col_start + load_y) < N:
                        Vt_shared[kk, load_y] = Vt[real_k, col_start + load_y] * S[real_k]
                    else:
                        Vt_shared[kk, load_y] = 0.0
                # For the padded column
                # Actually we have BN+1, so let's null it out:
                Vt_shared[kk, BN] = 0.0
                load_y += threads_per_block

            cuda.syncthreads()

            # Dot product
            for dotIdx in range(BK):
                for i in range(TM):
                    rowA = threadRow * TM + i
                    aVal = U_shared[rowA, dotIdx]
                    for j in range(TN):
                        colB = threadCol * TN + j
                        result[i, j] += aVal * Vt_shared[dotIdx, colB]

            cuda.syncthreads()

        # Store partial results
        for i in range(TM):
            out_row = row_start + threadRow*TM + i
            if out_row < M:
                for j in range(TN):
                    out_col = col_start + threadCol*TN + j
                    if out_col < N:
                        out[out_row, out_col] = result[i, j]

    return svd_reconstruct_tiled_padded


def main():
    parser = argparse.ArgumentParser(
        description="Tile-based GPU partial SVD-like reconstruction with dynamic shared memory + manual indexing."
    )
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension N.")
    # we take the dim based on jupyter notebook best performance
    parser.add_argument("--blockDimX", type=int, default=32, help="Block dimension X.")
    parser.add_argument("--blockDimY", type=int, default=32, help="Block dimension Y.")
    parser.add_argument("--rank", type=int, default=512, help="Partial rank k.")
    args = parser.parse_args()

    N = args.size
    bx = args.blockDimX
    by = args.blockDimY
    k = args.rank

    if bx < 8 or by < 8:
        raise ValueError("Block size must be at least 8x8, but should really be much bigger anyway...")

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
    
    threads_per_block = (BM * BN) // (BM / 8 * BN / 8)   # = (16*16)/(4) = 4

    print("\n--- Padded Shared-Memory Tile Kernel (Manual Indexing) ---")
    print(f"Matrix size: {N} x {N}")
    print(f"BlockDim  : ({bx}, {by})")
    print(f"Rank (k)  : {k}")
    print(f"Sub block dims: {BM/4} x {BN/4}")
    print(f"Threads per block: {threads_per_block}")
    print("-----------------------------------------------------------")


    kernel = get_kernel_tiled_padded(BM, BN)
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