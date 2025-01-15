### BEGIN SOLUTION
import numpy as np
import multiprocessing
import time
from tqdm import tqdm

# Define colors for process output
process_colors = {
    0: "\033[91m",  # Red
    1: "\033[92m",  # Green
    2: "\033[93m",  # Yellow
    3: "\033[94m",  # Blue
    4: "\033[96m",  # Cyan
}

def init_worker(shared_array_, array_shape_, lock_):
    """Initialize global variables for worker processes"""
    global shared_array, array_shape, lock
    shared_array = shared_array_
    array_shape = array_shape_ 
    lock = lock_

def process_chunk(args):
    """Process a chunk of rows for SVD reconstruction"""
    start_idx, end_idx, u_chunk, s, vt, k, color_idx = args
    
    # Calculate reconstruction for chunk
    chunk_result = np.zeros((end_idx - start_idx, array_shape[1]))
    for i in range(end_idx - start_idx):
        chunk_result[i] = np.dot(u_chunk[i, :k] * s[:k], vt[:k, :])
    
    # Update shared array with lock
    with lock:
        print(f"{process_colors[color_idx]}Process {color_idx}: Writing rows {start_idx} to {end_idx}")
        # Ensure we copy the exact number of elements
        shared_array[start_idx * array_shape[1]:end_idx * array_shape[1]] = chunk_result.flatten()
    
    return start_idx, end_idx

def reconstruct_svd_multiprocessing(u, s, vt, k):
    """SVD reconstruction using multiprocessing
    
    Args:
        u (ndarray): Left singular vectors
        s (ndarray): Singular values 
        vt (ndarray): Right singular vectors transposed
        k (int): Number of components to use
        
    Returns:
        ndarray: Reconstructed matrix
    """
    # Setup shared memory and synchronization
    array_shape = (u.shape[0], vt.shape[1])
    shared_array = multiprocessing.Array('d', array_shape[0] * array_shape[1])
    lock = multiprocessing.Lock()
    
    # Determine chunks for processes
    n_processes = min(multiprocessing.cpu_count(), len(process_colors))
    chunk_size = array_shape[0] // n_processes
    chunks = []
    
    for i in range(n_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_processes-1 else array_shape[0]
        chunks.append((
            start_idx, 
            end_idx,
            u[start_idx:end_idx],
            s,
            vt,
            k,
            i % len(process_colors)
        ))

    # Create and start process pool
    with multiprocessing.Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(shared_array, array_shape, lock)
    ) as pool:
        # Process chunks with progress bar
        list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # Reshape result from shared memory
    result = np.frombuffer(shared_array.get_obj()).reshape(array_shape)
    
    return result

if __name__ == "__main__":
    n, m = 200, 200
    original_matrix = np.random.rand(n, m)
    
    # Perform SVD
    u, s, vt = np.linalg.svd(original_matrix, full_matrices=False)
    
    # Number of components to use
    k = 50
    
    # Time the reconstruction
    start_time = time.time()
    reconstructed = reconstruct_svd_multiprocessing(u, s, vt, k)
    end_time = time.time()
    
    print(f"\nReconstruction completed in {end_time - start_time:.2f} seconds")
    print(f"Reconstruction error: {np.linalg.norm(original_matrix - reconstructed):.2e}")

### END SOLUTION
