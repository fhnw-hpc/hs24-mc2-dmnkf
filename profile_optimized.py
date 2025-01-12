import argparse
import math
import numpy as np
from numba import cuda, float32


def get_padded_kernel_dynamic():
    """
    Returns a tile-based GPU kernel that:
      • Uses dynamic shared memory (1D buffer)
      • Pads tileU and tileV to reduce shared-memory bank conflicts
      • Manually indexes each element to avoid reshape() issues
      • Applies small fixes to reduce integer overhead
    """

    @cuda.jit(fastmath=True)
    def tile_svd_kernel_s_padded_dynamic(u, s, vt, C, k, bx, by):
        """
        out[row, col] = sum_{r=0..k-1} (u[row, r] * s[r]) * vt[r, col]

        Shared memory layout (1D array):
          tileU_size  = by * (bx + 1)
          tileV_size  = bx * (by + 1)
          s_tile_size = bx
          total_size  = tileU_size + tileV_size + s_tile_size

        Offsets:
          offsetU = 0
          offsetV = tileU_size
          offsetS = tileU_size + tileV_size

        Tile Access:
          tileU[rowU + tx], tileV[colV + ty], etc.
          where rowU = offsetU + ty*(bx+1), colV = offsetV + tx*(by+1).

        The +1 in bx+1 or by+1 provides padding that helps avoid bank conflicts.
        """

        # 1) Access dynamic shared memory as a 1D float array
        shared_mem = cuda.shared.array(0, float32)

        # Compute chunk sizes
        tileU_size = by * (bx + 1)
        tileV_size = bx * (by + 1)

        offsetU = 0
        offsetV = tileU_size
        offsetS = tileU_size + tileV_size

        # Map thread indices to global row/col
        row = cuda.blockIdx.y * by + cuda.threadIdx.y
        col = cuda.blockIdx.x * bx + cuda.threadIdx.x

        # Check OOB
        if row >= C.shape[0] or col >= C.shape[1]:
            return

        # Single-precision accumulator
        acc = float32(0.0)
        tile_count = (k + bx - 1) // bx

        # Thread IDs
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # Precompute partial offsets to reduce integer overhead inside the loop
        # For U access
        rowU_base = offsetU + ty * (bx + 1)
        # For V access (transposed indexing)
        colV_base = offsetV + tx * (by + 1)

        for t in range(tile_count):
            k_base = t * bx

            # 1) Load chunk of s[] into shared memory
            s_idx = k_base + tx
            if s_idx < k:
                shared_mem[offsetS + tx] = s[s_idx]
            else:
                shared_mem[offsetS + tx] = float32(0.0)

            cuda.syncthreads()

            # 2) Load tileU from U[row, s_idx] * s_tile
            if row < u.shape[0] and s_idx < k:
                valS = shared_mem[offsetS + tx]
                shared_mem[rowU_base + tx] = u[row, s_idx] * valS
            else:
                shared_mem[rowU_base + tx] = float32(0.0)

            # 3) Load tileV from vt[s_idx2, col]
            s_idx2 = k_base + ty
            if s_idx2 < vt.shape[0] and col < vt.shape[1]:
                shared_mem[colV_base + ty] = vt[s_idx2, col]
            else:
                shared_mem[colV_base + ty] = float32(0.0)

            cuda.syncthreads()

            # 4) Dot product across this chunk
            for n in range(bx):
                valU = shared_mem[rowU_base + n]
                valV = shared_mem[colV_base + n]
                acc += valU * valV

            cuda.syncthreads()

        # 5) Write the final result to global memory
        C[row, col] = acc

    return tile_svd_kernel_s_padded_dynamic


def main():
    parser = argparse.ArgumentParser(
        description="Tile-based GPU partial SVD-like reconstruction with dynamic shared memory + manual indexing."
    )
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension N.")
    parser.add_argument("--blockDimX", type=int, default=16, help="Block dimension X.")
    parser.add_argument("--blockDimY", type=int, default=16, help="Block dimension Y.")
    parser.add_argument("--rank", type=int, default=512, help="Partial rank k.")
    args = parser.parse_args()

    N = args.size
    bx = args.blockDimX
    by = args.blockDimY
    k = args.rank

    print("\n--- Padded Shared-Memory Tile Kernel (Manual Indexing) ---")
    print(f"Matrix size: {N} x {N}")
    print(f"BlockDim  : ({bx}, {by})")
    print(f"Rank (k)  : {k}")
    print("-----------------------------------------------------------")

    # 1) Generate random data for U, S, VT
    U_host = np.random.randn(N, N).astype(np.float32)
    S_host = np.random.randn(N).astype(np.float32)
    VT_host = np.random.randn(N, N).astype(np.float32)

    # 2) Transfer data to GPU
    U_dev = cuda.to_device(U_host)
    S_dev = cuda.to_device(S_host)
    VT_dev = cuda.to_device(VT_host)
    out_dev = cuda.device_array((N, N), dtype=np.float32)

    # 3) Get the kernel
    kernel = get_padded_kernel_dynamic()

    # 4) Compute dynamic shared memory usage
    tileU_size = by * (bx + 1)
    tileV_size = bx * (by + 1)
    s_tile_size = bx
    total_floats = tileU_size + tileV_size + s_tile_size
    shared_mem_bytes = total_floats * 4  # 4 bytes per float32

    # 5) Grid/block config
    gridX = math.ceil(N / bx)
    gridY = math.ceil(N / by)
    grid_dims = (gridX, gridY)
    block_dims = (bx, by)

    # 6) Warm-up
    kernel[grid_dims, block_dims, 0, shared_mem_bytes](
        U_dev, S_dev, VT_dev, out_dev, k, bx, by
    )
    cuda.synchronize()

    # 7) Time the kernel with CUDA events
    start_event = cuda.event()
    stop_event = cuda.event()

    start_event.record()
    kernel[grid_dims, block_dims, 0, shared_mem_bytes](
        U_dev, S_dev, VT_dev, out_dev, k, bx, by
    )
    cuda.synchronize()
    stop_event.record()

    elapsed_ms = cuda.event_elapsed_time(start_event, stop_event)
    print(f"[GPU] Partial reconstruction time: {elapsed_ms / 1000:.5f} s\n")

    print("Done. Re-run with Nsight (nsys/ncu) to see if bank conflicts & linking errors are resolved.\n")


if __name__ == "__main__":
    main()