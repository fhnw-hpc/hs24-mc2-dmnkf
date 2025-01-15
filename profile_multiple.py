import math
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Dict
import logging

import numpy as np
from argparse import ArgumentParser
from numba import cuda, float32
from numpy.linalg import svd

# --- Configuration ---
LOG_LEVEL = logging.INFO
BLOCK_SIZE_X, BLOCK_SIZE_Y = 32, 32
TOTAL_IMAGES = 10000
IMG_SIZE = 1024
NUM_WORKERS = 4
NUM_STREAMS = 4 
IMG = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

BM = int(BLOCK_SIZE_X)
BN = int(BLOCK_SIZE_Y)
TM = BM // 8  # => 2
TN = BN // 8  # => 8
BK = int(BM / 2)

# Extra columns for padding
BKpad = BK + 1  # 8 + 1 = 9
BNpad = BN + 1  # 64 + 1 = 65

@cuda.jit(fastmath=True)
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


@dataclass
class WorkerResource:
    """
    One CPU worker controlling multiple CUDA streams. We store references
    to device memory and pinned host memory for each stream.
    """
    streams: List[cuda.stream]
    device_bufs: List[Dict[str, cuda.devicearray.DeviceNDArray]]
    host_bufs: List[Dict[str, np.ndarray]]


class Reconstructor:

    def __init__(self, shape_info: tuple):
        self.work_queue = Queue()
        self.shutdown_event = threading.Event()
        self.worker_threads: List[threading.Thread] = []

        self.num_workers = NUM_WORKERS
        self.num_streams = NUM_STREAMS
        self.worker_resources: List[WorkerResource] = []

        for worker_id in range(self.num_workers):
            logger.info(f"Allocating GPU resources for worker {worker_id}")
            resources = self._init_resources(shape_info)
            self.worker_resources.append(resources)

    def _init_resources(self, shape_info: tuple) -> WorkerResource:
        M, N, K = shape_info
        streams = []
        device_bufs = []
        host_bufs = []

        for _ in range(self.num_streams):
            streams.append(cuda.stream())
            device_dict = {
                "U": cuda.device_array((M, K), dtype=np.float32),
                "S": cuda.device_array((K,), dtype=np.float32),
                "V": cuda.device_array((K, N), dtype=np.float32),
                "C": cuda.device_array((M, N), dtype=np.float32),
            }
            host_dict = {
                "U": cuda.pinned_array((M, K), dtype=np.float32),
                "S": cuda.pinned_array((K,), dtype=np.float32),
                "V": cuda.pinned_array((K, N), dtype=np.float32),
                "C": cuda.pinned_array((M, N), dtype=np.float32),
            }
            device_bufs.append(device_dict)
            host_bufs.append(host_dict)

        return WorkerResource(streams, device_bufs, host_bufs)

    def _gpu_worker(self, worker_id: int, shape_info: tuple):
        """
        Worker thread: picks tasks from the queue, launches GPU ops, stores final data in pinned memory.
        """
        M, N, K = shape_info
        resources = self.worker_resources[worker_id]

        def on_gpu_complete(stream, status, arg):
            """
            Called after GPU completes. arg is (task_id, slot_id).
            We won't do disk saving here, only log final results if needed.
            """
            task_id, slot_id = arg
            # Data is already in resources.host_bufs[slot_id]["C"].
            logger.debug(f"[Worker {worker_id}] Task {task_id} finished on slot {slot_id}.")

        counter = 0
        logger.info(f"[Worker {worker_id}] started")
        try:
            while not self.shutdown_event.is_set():
                try:
                    task_id, decomp = self.work_queue.get(timeout=1.0)
                except Empty:
                    continue

                slot_id = counter % self.num_streams
                counter += 1

                stream = resources.streams[slot_id]
                dev = resources.device_bufs[slot_id]
                host = resources.host_bufs[slot_id]

                # Copy decomposition to pinned memory
                host["U"][:] = decomp["u"]
                host["S"][:] = decomp["s"]
                host["V"][:] = decomp["v"]

                # Transfer pinned -> device
                cuda.to_device(host["U"], to=dev["U"], stream=stream)
                cuda.to_device(host["S"], to=dev["S"], stream=stream)
                cuda.to_device(host["V"], to=dev["V"], stream=stream)

                M, _ = host["U"].shape
                _, N = host["V"].shape

                blocks_x = math.ceil(N / BN)
                blocks_y = math.ceil(M / BM)
                
                threads_per_block = (BM * BN) // (BM / 8 * BN / 8)   # = (16*16)/(4) = 4

                grid_dim = (blocks_x, blocks_y)
                block_dim = (int(threads_per_block),)

                svd_reconstruct_tiled_padded[grid_dim, block_dim, stream](dev["U"], dev["S"], dev["V"], dev["C"], M, N, K)

                dev["C"].copy_to_host(host["C"], stream=stream)

                # Callback so we know when GPU work is truly done
                stream.add_callback(on_gpu_complete, (task_id, slot_id))

                self.work_queue.task_done()
                logger.debug(f"[Worker {worker_id}] launched task {task_id} on slot {slot_id}")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] error: {str(e)}", exc_info=True)
        finally:
            logger.info(f"[Worker {worker_id}] shutting down...")
            for st in resources.streams:
                st.synchronize()
            logger.info(f"[Worker {worker_id}] all streams done.")

    def start_workers(self, shape_info: tuple):
        """
        Start worker threads. Each worker does GPU tasks for multiple streams.
        """
        for w_id in range(self.num_workers):
            th = threading.Thread(target=self._gpu_worker, args=(w_id, shape_info))
            th.start()
            self.worker_threads.append(th)

    def enqueue_task(self, decomp: Dict, task_id: int):
        """
        Enqueue a GPU reconstruction task (a dict with 'u', 's', 'v').
        """
        self.work_queue.put((task_id, decomp))

    def stop(self):
        """
        Signal worker(s) to stop and wait for them to finish.
        """
        logger.info("Stop requested. Shutting down workers...")
        self.shutdown_event.set()
        self.work_queue.join()

        for th in self.worker_threads:
            th.join()
        logger.info("All workers shut down successfully.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--sequential", action="store_true", help="Optional: run single-thread GPU approach.")
    args = parser.parse_args()
    logger.info(f"sequential={args.sequential}")

    # For demonstration, do a single CPU SVD of the random image
    u_mat, s_vec, vt_mat = svd(IMG, full_matrices=False)
    M, K = u_mat.shape
    K2, N = vt_mat.shape
    if K2 != K:
        raise ValueError("Mismatch in SVD shapes.")

    shape_info = (M, N, K)
    example_decomp = {"u": u_mat, "s": s_vec, "v": vt_mat}

    if args.sequential:
        # Minimal approach: no concurrency, single stream
        logger.info("Running single-stream, no concurrency approach...")
        U_host = cuda.pinned_array((M, K), dtype=np.float32)
        S_host = cuda.pinned_array((K,),   dtype=np.float32)
        V_host = cuda.pinned_array((K, N), dtype=np.float32)
        C_host = cuda.pinned_array((M, N), dtype=np.float32)
        
        U_host[:] = example_decomp["u"]
        S_host[:] = example_decomp["s"]
        V_host[:] = example_decomp["v"]


        start_t = time.time()
        for i in range(TOTAL_IMAGES):
            # Allocate pinned arrays here (or reuse them)

            # Copy to device
            U_dev = cuda.to_device(U_host)
            S_dev = cuda.to_device(S_host)
            V_dev = cuda.to_device(V_host)
            C_dev = cuda.device_array((M, N), dtype=np.float32)

            M, _ = U_host.shape
            _, N = V_host.shape

            blocks_x = math.ceil(N / BN)
            blocks_y = math.ceil(M / BM)
            
            threads_per_block = (BM * BN) // (BM / 8 * BN / 8)

            grid_dim = (blocks_x, blocks_y)
            block_dim = (int(threads_per_block),)

            svd_reconstruct_tiled_padded[grid_dim, block_dim](U_dev, S_dev, V_dev, C_dev, M, N, K)

            C_dev.copy_to_host(C_host)

        cuda.synchronize()
        elapsed = time.time() - start_t
        logger.info(f"Sequential run took {elapsed:.2f}s => {TOTAL_IMAGES / elapsed:.2f} images/s")
        return

    # Concurrency approach
    start_t = time.time()
    manager = Reconstructor(shape_info)
    manager.start_workers(shape_info)

    # Enqueue tasks
    for i in range(TOTAL_IMAGES):
        manager.enqueue_task(example_decomp, i)

    # Wait for tasks, then stop
    manager.work_queue.join()
    manager.stop()

    end_t = time.time()
    total_time = end_t - start_t
    logger.info(f"Concurrent GPU run: {total_time:.2f}s => {TOTAL_IMAGES / total_time:.2f} images/s")


if __name__ == "__main__":
    main()
