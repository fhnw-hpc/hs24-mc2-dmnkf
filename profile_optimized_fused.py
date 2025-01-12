import argparse
import math
import numpy as np
from numba import cuda, float32

from numba import cuda, float32

def get_optimized_kernel_s_shared(threads_per_block):
    blockDimX, blockDimY = threads_per_block

    @cuda.jit(fastmath=True)
    def tile_based_svd_kernel_s_shared(u, s, vt, C, k):
        row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if row >= C.shape[0] or col >= C.shape[1]:
            return

        tileU  = cuda.shared.array(shape=(blockDimY, blockDimX), dtype=float32)
        tileV  = cuda.shared.array(shape=(blockDimX, blockDimY), dtype=float32)
        s_tile = cuda.shared.array(shape=(blockDimX,), dtype=float32)

        # Force single-precision accumulator
        acc = float32(0.0)

        tile_count = (k + blockDimX - 1) // blockDimX
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        for t in range(tile_count):
            k_base = t * blockDimX

            s_index = k_base + tx
            if s_index < k:
                s_tile[tx] = s[s_index]
            else:
                s_tile[tx] = float32(0.0)

            cuda.syncthreads()

            if row < u.shape[0] and s_index < k:
                tileU[ty, tx] = u[row, s_index] * s_tile[tx]
            else:
                tileU[ty, tx] = float32(0.0)

            v_index = k_base + ty
            if v_index < vt.shape[0] and col < vt.shape[1]:
                tileV[tx, ty] = vt[v_index, col]
            else:
                tileV[tx, ty] = float32(0.0)

            cuda.syncthreads()

            for n in range(blockDimX):
                # Optionally use an fma intrinsic:
                # acc = fma(tileU[ty, n], tileV[n, tx], acc)
                acc += tileU[ty, n] * tileV[n, tx]

            cuda.syncthreads()

        C[row, col] = acc

    return tile_based_svd_kernel_s_shared


def main():
    parser = argparse.ArgumentParser(description="GPU-only tile-based SVD-like reconstruction with s shared memory.")
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension N.")
    parser.add_argument("--blockDimX", type=int, default=16, help="Block dimension X.")
    parser.add_argument("--blockDimY", type=int, default=16, help="Block dimension Y.")
    parser.add_argument("--rank", type=int, default=512, help="Partial rank k.")
    args = parser.parse_args()

    N = args.size
    blockDimX = args.blockDimX
    blockDimY = args.blockDimY
    k = args.rank

    print("---- GPU S-Shared Tile SVD-Like Reconstruction ----")
    print(f"Matrix size   : {N} x {N}")
    print(f"BlockDim      : ({blockDimX}, {blockDimY})")
    print(f"Partial rank k: {k}")
    print("---------------------------------------------------")

    # 1) Generate random U, S, V^T with shapes:
    #    U: (N, N),  s: (N,),  V^T: (N, N).
    #    No CPU SVD done here, just random data to test the kernel's performance.

    U_host = np.random.randn(N, N).astype(np.float32)
    S_host = np.random.randn(N).astype(np.float32)
    VT_host = np.random.randn(N, N).astype(np.float32)

    # 2) Copy to GPU
    U_dev = cuda.to_device(U_host)
    S_dev = cuda.to_device(S_host)
    VT_dev = cuda.to_device(VT_host)
    out_dev = cuda.device_array((N, N), dtype=np.float32)

    # 3) Retrieve kernel
    tile_kernel = get_optimized_kernel_s_shared((blockDimX, blockDimY))

    # 4) Grid config
    gridX = math.ceil(N / blockDimX)
    gridY = math.ceil(N / blockDimY)
    grid_dims = (gridX, gridY)
    block_dims = (blockDimX, blockDimY)

    # 5) Warm-up
    tile_kernel[grid_dims, block_dims](U_dev, S_dev, VT_dev, out_dev, k)
    cuda.synchronize()

    # 6) Time with CUDA events
    start_event = cuda.event()
    stop_event = cuda.event()

    start_event.record()
    tile_kernel[grid_dims, block_dims](U_dev, S_dev, VT_dev, out_dev, k)
    cuda.synchronize()
    stop_event.record()

    gpu_time_ms = cuda.event_elapsed_time(start_event, stop_event)
    gpu_time_s = gpu_time_ms / 1000.0

    print(f"[GPU] Partial reconstruction time (k={k}): {gpu_time_s:.5f} s\n")


if __name__ == "__main__":
    main()