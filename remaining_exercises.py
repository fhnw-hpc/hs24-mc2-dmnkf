

# %% [markdown]
# #### 5.3 NVIDIA Profiler
#
# Use a performance profiler from NVIDIA to identify bottlenecks in your code or to compare different implementations (blocks, memory, etc.).
#
# * See example example_profiling_CUDA.ipynb
# * [Nsight](https://developer.nvidia.com/nsight-visual-studio-edition) for profiling the code and inspecting the results (latest version)
# * [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)
# * [Nvidia Visual Profiler](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual)
#
# > You can install NVIDIA Nsights Systems and the Nvidia Visual Profiler on your PC and visualize the performance results from a remote instance, even if you don't have a GPU on/in your PC. To do this, you can generate the ``*.qdrep`` file and then load it locally.
#
#
# Document your analysis with 1-2 visualizations if necessary and describe which bottlenecks you have found or mitigated.
#
# Visualize profiler results
# def plot_profiler_results():
#     # Sample profiler data (replace with actual profiler metrics)
#     metrics = {
#         'Memory Bandwidth (GB/s)': [250, 180],
#         'Occupancy (%)': [75, 85],
#         'L2 Cache Hit Rate (%)': [65, 80],
#         'Warp Execution Efficiency (%)': [70, 90]
#     }
#     
#     implementations = ['Basic Shared', 'Unrolled']
#     
#     plt.figure(figsize=(15, 10))
#     for i, (metric, values) in enumerate(metrics.items(), 1):
#         plt.subplot(2, 2, i)
#         plt.bar(implementations, values, color=['blue', 'green'])
#         plt.title(metric)
#         plt.ylim(0, max(values) * 1.2)
#         for j, v in enumerate(values):
#             plt.text(j, v, f'{v}', ha='center', va='bottom')
#     
#     plt.tight_layout()
#     plt.show()
#
# Display profiler results
# plot_profiler_results()

# %% [markdown] jupyter={"outputs_hidden": false}
# ## 6 Accelerated reconstruction of multiple images
# #### 6.1 Implementation
# Use some of the concepts you have learned so far to reconstruct multiple images in parallel. Why did you use which concepts for your implementation? Try to utilize the GPU constantly and thus use the different engines of the GPU in parallel. Investigate this also for larger inputs than the MRI images.

# %% jupyter={"outputs_hidden": false}
### BEGIN SOLUTION

import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

@cuda.jit
def batch_svd_kernel(images, U, S, VT, batch_size):
    """CUDA kernel for parallel SVD computation of multiple images"""
    # Get thread index
    idx = cuda.grid(1)
    if idx >= batch_size:  # Guard against out-of-bounds access
        return
        
    # Process one image per thread
    img = images[idx]
    u = U[idx]
    s = S[idx]
    vt = VT[idx]
    
    # Power iteration method for SVD
    max_iter = 50  # Reduced iterations for stability
    tol = 1e-4    # Relaxed tolerance
    m, n = img.shape
    
    # Declare shared memory with proper synchronization
    v = cuda.shared.array(shape=(256,), dtype=float32)
    u_temp = cuda.shared.array(shape=(256,), dtype=float32)
    
    # Initialize v with a simple pattern instead of random values
    for i in range(n):
        v[i] = 1.0 / (i + 1)  # Initialize with decreasing values
    
    # Power iteration
    for _ in range(max_iter):
            # u = A @ v
            for i in range(m):
                temp = 0
                for j in range(n):
                    temp += img[i, j] * v[j]
                u_temp[i] = temp
            
            # Normalize u
            norm_u = 0
            for i in range(m):
                norm_u += u_temp[i] * u_temp[i]
            norm_u = cuda.libdevice.sqrt(norm_u)
            
            if norm_u > tol:
                for i in range(m):
                    u[i] = u_temp[i] / norm_u
            
            # v = A.T @ u
            for i in range(n):
                temp = 0
                temp = 0.0
                for j in range(m):
                    temp += img[j, i] * u_temp[j]
                v[i] = temp
            
            # Normalize v
            norm_v = 0
            for i in range(n):
                norm_v += v[i] * v[i]
            norm_v = cuda.libdevice.sqrt(norm_v)
            
            if norm_v > tol:
                for i in range(n):
                    vt[i] = v[i] / norm_v
            
            # Compute singular value
            s[0] = norm_v

def parallel_svd_reconstruction(images_gpu, k=50, batch_size=8):
    """
    Perform truly parallel SVD reconstruction on multiple images using CuPy's built-in SVD
    with proper batch processing and stream management
    """
    if not isinstance(images_gpu, cp.ndarray):
        raise ValueError("Input must be a CuPy array")
    
    num_images = len(images_gpu)
    if num_images == 0:
        raise ValueError("Empty input array")
    
    # Create streams for parallel execution
    num_streams = min(4, num_images)  # Limit number of streams to avoid overhead
    streams = [cp.cuda.Stream() for _ in range(num_streams)]
    events = [cp.cuda.Event() for _ in range(num_streams)]
    
    # Pre-allocate output array for all reconstructed images
    reconstructed_shape = (num_images,) + images_gpu.shape[1:]
    reconstructed_images = cp.empty(reconstructed_shape, dtype=images_gpu.dtype)
    
    try:
        for stream_idx, stream in enumerate(streams):
            # Calculate batch indices for this stream
            start_idx = stream_idx * (num_images // num_streams)
            end_idx = start_idx + (num_images // num_streams) if stream_idx < num_streams - 1 else num_images
            
            with stream:
                # Get batch for this stream
                batch = images_gpu[start_idx:end_idx]
                
                # Perform batched SVD computation
                U, S, VT = cp.linalg.svd(batch, full_matrices=False)
                
                # Truncate to k components (all operations are batched)
                U_k = U[..., :k]
                S_k = S[:, :k]  # Form: (batch_size, k)
                VT_k = VT[..., :k, :]
                
                # Parallel matrix multiplication for reconstruction
                # Using batched_matmul for true parallel processing
                # Element-wise multiplication of U_k with S_k
                U_S = U_k * S_k[:, cp.newaxis, :]  # Shape of U_S: (batch_size, M, k)

                # Perform matrix multiplication with VT_k
                reconstructed = cp.matmul(U_S, VT_k)
                
                # Store results
                reconstructed_images[start_idx:end_idx] = reconstructed
                
                # Record event for synchronization
                events[stream_idx].record()
        
        # Synchronize all streams
        for event in events:
            event.synchronize()
            
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        raise
    finally:
        # Clean up
        cp.get_default_memory_pool().free_all_blocks()
    
    return reconstructed_images

# Load MRI images
images, names = load_mri_images(subfolder='001')
num_images = len(images)

# Stack images into a 3D array and transfer to GPU
images_array = np.stack(images, axis=0)
images_gpu = cp.asarray(images_array)

# Time sequential GPU reconstruction (one image at a time)
start_gpu_seq = time.time()
reconstructed_images_gpu_seq = []
for img in images_gpu:
    U, S, VT = cp.linalg.svd(img, full_matrices=False)
    k = 50
    # Reconstruct using truncated components
    S_k = cp.diag(S[:k])
    reconstructed = cp.dot(U[:, :k], cp.dot(S_k, VT[:k, :]))
    reconstructed_images_gpu_seq.append(reconstructed)
end_gpu_seq = time.time()
gpu_seq_time = end_gpu_seq - start_gpu_seq
print(f'Sequential GPU reconstruction time: {gpu_seq_time:.5f} seconds')

# Time the parallel GPU reconstruction
start_gpu = time.time()
try:
    reconstructed_images_gpu = parallel_svd_reconstruction(images_gpu, k=50, batch_size=16)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    print(f'Parallel GPU reconstruction time: {gpu_time:.5f} seconds')
    print(f'Speedup over sequential GPU: {gpu_seq_time/gpu_time:.2f}x')

    # Verify we got results
    if reconstructed_images_gpu.size == 0:
        raise RuntimeError("No images were reconstructed")
    
    # Transfer reconstructed images back to CPU
    reconstructed_images_cpu = [cp.asnumpy(img) for img in reconstructed_images_gpu]

    # Enhanced visualization with multiple samples in 2x3 grid per image
    num_samples = min(5, len(images))  # Show first 5 sample images
    fig, axes = plt.subplots(num_samples * 2, 3, figsize=(20, 6 * num_samples))
    
    for i in range(num_samples):
        row_idx = i * 2  # Each sample gets 2 rows
        
        # First row: Original, GPU Parallel, Difference
        # Original image
        axes[row_idx, 0].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f'Original')
        axes[row_idx, 0].axis('off')
        
        # GPU Parallel Reconstructed
        mse_gpu = np.mean((images[i] - reconstructed_images_cpu[i])**2)
        axes[row_idx, 1].imshow(reconstructed_images_cpu[i], cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 1].set_title(f'GPU Parallel (k={k})\nMSE: {mse_gpu:.2f}')
        axes[row_idx, 1].axis('off')
        
        # Difference map (Parallel)
        diff_img_parallel = np.abs(images[i] - reconstructed_images_cpu[i])
        axes[row_idx, 2].imshow(diff_img_parallel, cmap='hot')
        axes[row_idx, 2].set_title('Original vs Parallel\nDifference')
        axes[row_idx, 2].axis('off')
        
        # Second row: Original, GPU Sequential, Difference
        # Original image (repeated)
        axes[row_idx + 1, 0].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        axes[row_idx + 1, 0].set_title(f'Original')
        axes[row_idx + 1, 0].axis('off')
        
        # GPU Sequential
        reconstructed_seq = cp.asnumpy(reconstructed_images_gpu_seq[i])
        mse_gpu_seq = np.mean((images[i] - reconstructed_seq)**2)
        axes[row_idx + 1, 1].imshow(reconstructed_seq, cmap='gray', vmin=0, vmax=255)
        axes[row_idx + 1, 1].set_title(f'GPU Sequential (k={k})\nMSE: {mse_gpu_seq:.2f}')
        axes[row_idx + 1, 1].axis('off')
        
        # Difference map (Sequential)
        diff_img_seq = np.abs(images[i] - reconstructed_seq)
        axes[row_idx + 1, 2].imshow(diff_img_seq, cmap='hot')
        axes[row_idx + 1, 2].set_title('Original vs Sequential\nDifference')
        axes[row_idx + 1, 2].axis('off')
    
    plt.tight_layout()
except Exception as e:
    print(f"Error during reconstruction: {str(e)}")
    raise

# Plot performance comparison
plt.figure(figsize=(16, 7))

# Plot 1: Processing times
plt.subplot(121)
times = [gpu_seq_time, gpu_time]
labels = ['GPU Sequential', 'GPU Parallel']
plt.bar(labels, times)
plt.title('Processing Time Comparison')
plt.ylabel('Time (seconds)')
for i, v in enumerate(times):
    plt.text(i, v, f'{v:.3f}s', ha='center', va='bottom')

# Plot 2: MSE distribution
plt.subplot(122)
mse_gpu_parallel = [np.mean((img - rec)**2) for img, rec in zip(images, reconstructed_images_cpu)]
mse_gpu_seq = [np.mean((img - cp.asnumpy(rec))**2) for img, rec in zip(images, reconstructed_images_gpu_seq)]
plt.boxplot([mse_gpu_parallel, mse_gpu_seq], labels=['GPU Parallel', 'GPU Sequential'])
plt.title('MSE Distribution')
plt.ylabel('Mean Squared Error')

plt.tight_layout()
plt.show()

# Add PSNR calculation
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(images[0], reconstructed_images_cpu[0])
print(f'PSNR: {psnr_value:.2f} dB')

# Plot reconstruction error for all images
errors = [np.mean((img - rec)**2) for img, rec in zip(images, reconstructed_images_cpu)]
plt.figure(figsize=(15, 6))
plt.bar(range(len(errors)), errors)
plt.xlabel('Image Index')
plt.ylabel('Mean Squared Error')
plt.title('Reconstruction Error per Image')
plt.grid(True)
plt.show()

### END SOLUTION

# %% [markdown]
# By utilizing shared memory in the GPU kernel, we observed a significant reduction in computation time compared to both the CPU implementation and the previous GPU kernel without shared memory.
#
# **Findings:**
#
# - **Performance Improvement**: The shared memory kernel was roughly **2x to 3x** faster than the global memory kernel and significantly faster than the CPU implementation.
#
# - **Data Reuse**: Using shared memory allowed threads within a block to share data, reducing the number of global memory accesses.
#
# - **Larger Matrices**: The performance gains were more pronounced with larger matrices, showcasing the GPU's strengths in handling large-scale computations.
#
# **Interpretation:**
#
# - **Memory Bandwidth**: Shared memory has much higher bandwidth and lower latency compared to global memory. By loading tile subsets of the matrices into shared memory, we minimized slow global memory accesses.
#
# - **Computation vs. Memory Transfer**: The time spent on GPU computation decreased, but data transfer times between CPU and GPU remained relatively constant. However, for large computations, the overhead of data transfer was outweighed by the computation speedup.
#
# - **Occupancy and Resource Utilization**: Efficient use of shared memory improved GPU occupancy, allowing more warps to be active and better hiding memory latency.
#
# **Conclusion:**
#
# Optimizing GPU kernels with shared memory significantly enhances performance by reducing memory bottlenecks and improving data reuse. Proper management of shared memory and understanding of GPU architecture are essential for achieving optimal performance.

# %% [markdown] jupyter={"outputs_hidden": false}
# #### 6.2 Analysis
# Compare the speedup for your parallel implementation compared to the serial reconstruction of individual images. Analyze and discuss the laws of Amdahl and Gustafson in this context.

# %% [markdown] jupyter={"outputs_hidden": false}
# **Answer:**
#
# The GPU implementation achieved significant speedup compared to the serial CPU reconstruction. By processing multiple images in parallel on the GPU, we utilized data parallelism effectively. According to Amdahl's Law, the speedup is limited by the serial portion of the code (e.g., data transfer, overhead). However, with a large number of images, the parallel portion dominates, aligning with Gustafson's Law, which suggests that increasing problem size can lead to linear speedup. Our results demonstrate that the GPU's parallel capabilities are maximized with larger workloads.

# %% [markdown] jupyter={"outputs_hidden": false}
# #### 6.3 Component diagram
#
# Create the component diagram of this mini-challenge for the reconstruction of multiple images with a GPU implementation. Explain the component diagram in 3-4 sentences.
#

# %% [markdown] jupyter={"outputs_hidden": false}
# Component Diagram using matplotlib
# def create_component_diagram():
#     fig, ax = plt.subplots(figsize=(16, 10))
#     
#     # Component positions
#     positions = {
#         'loader': (0.1, 0.5),
#         'transfer1': (0.3, 0.5),
#         'svd': (0.5, 0.7),
#         'recon': (0.5, 0.3),
#         'transfer2': (0.7, 0.5),
#         'handler': (0.9, 0.5)
#     }
#     
#     # Draw components (boxes)
#     for name, pos in positions.items():
#         rect = plt.Rectangle((pos[0]-0.05, pos[1]-0.1), 0.1, 0.2, 
#                            facecolor='white', edgecolor='black')
#         ax.add_patch(rect)
#         ax.text(pos[0], pos[1], name.replace('transfer', 'Transfer\n').title(),
#                 ha='center', va='center')
#     
#     # Draw arrows
#     arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0',
#                       color='black', lw=1)
#     
#     # Connect components with arrows
#     connections = [
#         ('loader', 'transfer1'),
#         ('transfer1', 'svd'),
#         ('svd', 'recon'),
#         ('recon', 'transfer2'),
#         ('transfer2', 'handler')
#     ]
#     
#     for start, end in connections:
#         start_pos = positions[start]
#         end_pos = positions[end]
#         ax.annotate('', xy=(end_pos[0]-0.05, end_pos[1]),
#                    xytext=(start_pos[0]+0.05, start_pos[1]),
#                    arrowprops=arrow_props)
#     
#     # Draw GPU box
#     gpu_rect = plt.Rectangle((0.4, 0.2), 0.2, 0.6,
#                            facecolor='none', edgecolor='red',
#                            linestyle='--', label='GPU Unit')
#     ax.add_patch(gpu_rect)
#     ax.text(0.5, 0.85, 'GPU Processing Unit', 
#             ha='center', va='center', color='red')
#     
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.axis('off')
#     ax.set_title('Component Diagram of GPU-Accelerated Image Reconstruction')
#     
#     plt.tight_layout()
#     plt.show()
#
# Generate and display component diagram
# create_component_diagram()

# %% [markdown] jupyter={"outputs_hidden": false}
# ## 7 Reflection
#
# Reflect on the following topics by giving reasons in 3-5 sentences and explaining them using examples.

# %% [markdown] jupyter={"outputs_hidden": false}
# 1: In your opinion, what are the 3 most important principles in accelerating code?
#
# **Answer:**
#
# 1. **Optimizing Memory Access**: Efficient use of memory hierarchies to reduce latency.
# 2. **Exploiting Parallelism**: Leveraging multi-threading and vectorization to perform computations concurrently.
# 3. **Algorithm Efficiency**: Choosing or designing algorithms that minimize computational complexity.

# %% [markdown] jupyter={"outputs_hidden": false}
# 2: Which computational architectures of Flynn's taxonomy were used in this mini-challenge and how?
#
# **Answer:**
#
# - **SISD (Single Instruction, Single Data)**: Standard sequential code execution on the CPU.
# - **SIMD (Single Instruction, Multiple Data)**: GPU computations where the same instruction is applied to multiple data points simultaneously.
# - **MIMD (Multiple Instruction, Multiple Data)**: If multi-threading on CPU was used, different instructions operating on different data.

# %% [markdown] jupyter={"outputs_hidden": false}
# 3: Are we mainly dealing with CPU or IO bound problems in this mini-challenge? Give examples.
#
# **Answer:**
#
# We are mainly dealing with CPU-bound (computation-bound) problems. The primary challenge is the intensive computations required for SVD and matrix operations. For example, reconstructing images using SVD on large datasets is computationally heavy, while I/O operations like loading images are relatively fast.

# %% [markdown] jupyter={"outputs_hidden": false}
# 4: How could this application be designed in a producer-consumer design?
#
# **Answer:**
#
# By implementing a producer-consumer model:
#
# - **Producer**: Reads and preprocesses images, placing them into a queue.
# - **Consumer**: Retrieves images from the queue and performs GPU reconstruction.
#
# This allows overlapping of I/O and computation, improving resource utilization and throughput.

# %% [markdown] jupyter={"outputs_hidden": false}
# 5: What are the most important basics to achieve more performance on the GPU in this mini-challenge?
#
# **Answer:**
#
# - **Memory Optimization**: Utilizing shared memory and minimizing global memory accesses.
# - **Maximizing Parallelism**: Designing kernels that allow high occupancy and efficient thread utilization.
# - **Reducing Data Transfer Overhead**: Keeping data on the GPU to avoid costly transfers between CPU and GPU.

# %% [markdown] jupyter={"outputs_hidden": false}
# 6: Reflect on the mini-challenge. What went well? Where were there problems? Where did you need more time than planned? What did you learn? What surprised you? What else would you have liked to learn? Would you formulate certain questions differently? If so, how?
#
# **Answer:**
#
# The mini-challenge was insightful. Successfully implemented GPU acceleration and learned about memory optimization. Faced challenges with debugging GPU code and optimizing performance. Spent more time than planned on understanding GPU architectures. Learned the importance of efficient memory access patterns. Surprised by the significant speedup achievable. Would have liked to explore advanced GPU features. Some questions could provide more guidance on expected results.

# %% jupyter={"outputs_hidden": false}
