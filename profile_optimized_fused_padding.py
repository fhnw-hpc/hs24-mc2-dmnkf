import argparse
import math
import numpy as np
from numba import cuda, float32

def get_optimized_kernel_s_shared(threads_per_block):

    blockDimX, blockDimY = threads_per_block

    @cuda.jit(fastmath=True)
    def tile_based_svd_kernel_s_shared(u, s, vt, C, k):
        # Access dynamic shared memory as a 1D buffer
        shared_mem = cuda.shared.array(0, float32)

        # Precompute chunk sizes
        tileU_size  = blockDimY * (blockDimX + 1)  # +1 to reduce bank conflicts
        tileV_size  = blockDimX * (blockDimY + 1)
        s_tile_size = blockDimX
        total_size  = tileU_size + tileV_size + s_tile_size

        # Partition the 1D buffer
        tileU_1d  = shared_mem[0 : tileU_size]
        tileV_1d  = shared_mem[tileU_size : tileU_size + tileV_size]
        s_tile_1d = shared_mem[tileU_size + tileV_size : total_size]

        # Map thread to global row/col
        row = cuda.blockIdx.y * blockDimY + cuda.threadIdx.y
        col = cuda.blockIdx.x * blockDimX + cuda.threadIdx.x

        if row >= C.shape[0] or col >= C.shape[1]:
            return

        # Accumulator
        acc = float32(0.0)
        tile_count = (k + blockDimX - 1) // blockDimX

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        for t in range(tile_count):
            k_base = t * blockDimX
            s_idx  = k_base + tx

            # 1) Load s_tile
            if s_idx < k:
                s_tile_1d[tx] = s[s_idx]
            else:
                s_tile_1d[tx] = float32(0.0)

            cuda.syncthreads()

            # 2) tileU: row-based => ty * (blockDimX+1) + tx
            #    Multiply U by s_tile
            u_index_1d = ty * (blockDimX + 1) + tx
            if (row < u.shape[0]) and (s_idx < k):
                val_s = s_tile_1d[tx]
                tileU_1d[u_index_1d] = u[row, s_idx] * val_s
            else:
                tileU_1d[u_index_1d] = float32(0.0)

            # 3) tileV: col-based => tx * (blockDimY+1) + ty
            v_index_1d = tx * (blockDimY + 1) + ty
            v_idx = k_base + ty
            if v_idx < vt.shape[0] and col < vt.shape[1]:
                tileV_1d[v_index_1d] = vt[v_idx, col]
            else:
                tileV_1d[v_index_1d] = float32(0.0)

            cuda.syncthreads()

            # 4) Dot product across blockDimX
            #    tileU_1d[ty*(blockDimX+1) + n] * tileV_1d[n*(blockDimY+1) + tx]
            for n in range(blockDimX):
                acc += tileU_1d[ty * (blockDimX + 1) + n] * tileV_1d[n * (blockDimY + 1) + tx]

            cuda.syncthreads()

        C[row, col] = acc

    return tile_based_svd_kernel_s_shared


def main():
    parser = argparse.ArgumentParser(description="Tile-based SVD with dynamic shared memory & padding to reduce bank conflicts.")
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension N")
    parser.add_argument("--blockDimX", type=int, default=16, help="Block dimension X")
    parser.add_argument("--blockDimY", type=int, default=16, help="Block dimension Y")
    parser.add_argument("--rank", type=int, default=512, help="Partial rank k")
    args = parser.parse_args()

    N = args.size
    bx = args.blockDimX
    by = args.blockDimY
    k  = args.rank

    print("---- GPU S-Shared Tile SVD with Padding (Dynamic) ----")
    print(f"Matrix size   : {N} x {N}")
    print(f"BlockDim      : ({bx}, {by})")
    print(f"Partial rank  : {k}")
    print("-----------------------------------------")

    # Generate random data
    U_host  = np.random.randn(N, N).astype(np.float32)
    S_host  = np.random.randn(N).astype(np.float32)
    VT_host = np.random.randn(N, N).astype(np.float32)

    U_dev   = cuda.to_device(U_host)
    S_dev   = cuda.to_device(S_host)
    VT_dev  = cuda.to_device(VT_host)
    out_dev = cuda.device_array((N, N), dtype=np.float32)

    # Get kernel
    kernel = get_optimized_kernel_s_shared((bx, by))

    # Compute dynamic shared memory usage
    tileU_size  = by*(bx+1)
    tileV_size  = bx*(by+1)
    s_tile_size = bx
    total_floats = tileU_size + tileV_size + s_tile_size
    shared_mem_bytes = total_floats * 4  # 4 bytes per float32

    # Grid, block
    gridX = math.ceil(N / bx)
    gridY = math.ceil(N / by)
    grid_dims = (gridX, gridY)
    block_dims= (bx, by)

    # Warm-up
    # we need to pass the shared memory size to the kernel to avoid shape errors
    kernel[grid_dims, block_dims, 0, shared_mem_bytes](U_dev, S_dev, VT_dev, out_dev, k)
    cuda.synchronize()

    # Time with CUDA events
    start_event = cuda.event()
    stop_event  = cuda.event()

    start_event.record()
    kernel[grid_dims, block_dims, 0, shared_mem_bytes](U_dev, S_dev, VT_dev, out_dev, k)
    cuda.synchronize()
    stop_event.record()

    elapsed_ms = cuda.event_elapsed_time(start_event, stop_event)
    print(f"[GPU] Partial reconstruction time: {elapsed_ms/1000:.5f} s\n")
    print("Now you can run Nsight Compute on this script, and it shouldn't fail on the shape errors.\n")


if __name__ == "__main__":
    main()