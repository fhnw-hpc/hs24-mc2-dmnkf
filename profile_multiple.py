# gpu_reconstruction_no_save_demo.py
# Purpose: Demonstrate GPU concurrency for random images, no disk saving.
#          We remove all logic that wrote images to "out/".

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
BLOCK_SIZE_X, BLOCK_SIZE_Y = 16, 16
TOTAL_IMAGES = 2
IMG_SIZE = 2**10  # 1024
NUM_WORKERS = 1   # single CPU worker controlling GPU
NUM_STREAMS = 16   # concurrency on GPU
IMG = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@cuda.jit
def gpu_reconstruct_kernel(u, s, vt, out_matrix, k):
    """
    GPU kernel: reconstruct the matrix from U, S, and V^T, storing in 'out_matrix'.
    """
    y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x < u.shape[0] and y < vt.shape[1]:
        accum = float32(0.0)
        for ki in range(k):
            accum += u[x, ki] * s[ki] * vt[ki, y]
        out_matrix[x, y] = accum


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

                gx = math.ceil(N / BLOCK_SIZE_X)
                gy = math.ceil(M / BLOCK_SIZE_Y)
                gpu_reconstruct_kernel[(gy, gx), (BLOCK_SIZE_X, BLOCK_SIZE_Y), stream](
                    dev["U"], dev["S"], dev["V"], dev["C"], K
                )

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
        start_t = time.time()
        for i in range(TOTAL_IMAGES):
            # Allocate pinned arrays here (or reuse them)
            U_host = cuda.pinned_array((M, K), dtype=np.float32)
            S_host = cuda.pinned_array((K,),   dtype=np.float32)
            V_host = cuda.pinned_array((K, N), dtype=np.float32)
            C_host = cuda.pinned_array((M, N), dtype=np.float32)

            U_host[:] = example_decomp["u"]
            S_host[:] = example_decomp["s"]
            V_host[:] = example_decomp["v"]

            # Copy to device
            U_dev = cuda.to_device(U_host)
            S_dev = cuda.to_device(S_host)
            V_dev = cuda.to_device(V_host)
            C_dev = cuda.device_array((M, N), dtype=np.float32)

            gx = math.ceil(N / BLOCK_SIZE_X)
            gy = math.ceil(M / BLOCK_SIZE_Y)
            gpu_reconstruct_kernel[(gy, gx), (BLOCK_SIZE_X, BLOCK_SIZE_Y)](U_dev, S_dev, V_dev, C_dev, K)

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