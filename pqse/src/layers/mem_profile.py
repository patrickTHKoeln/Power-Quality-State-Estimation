import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from memory_profiler import memory_usage
from tqdm import tqdm

from AdmittanceLayer import prepare_admittance_matrices

"""
Scripts for measuring memory usage of different approaches to matrix multiplications that are 
memory intensive in custom layers.
"""


# @profile
def einsum_approach(x, kernel, y_mats):
    masked_weights = tf.multiply(kernel, y_mats)
    output = tf.einsum('bfci,fcijko->bjko', x, masked_weights)
    return output


# @profile
def reshape_approach(x, kernel, y_mats):
    masked_weights = tf.multiply(kernel, y_mats)
    # reshape x to 2d
    # x_resh = tf.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    x_resh = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    # reshape masked_weights to 2d
    masked_weights_resh = tf.reshape(masked_weights,
                                     (np.prod(masked_weights.shape[0:3]), np.prod(masked_weights.shape[3:])))
    # matmul
    output = tf.matmul(x_resh, masked_weights_resh)
    # reshape output to 4d
    output = tf.reshape(output, (x.shape[0] + masked_weights.shape[3:]))
    return output


def memory_profile(func, *args):
    mem_usage = memory_usage((func, args), interval=1, timeout=1)
    return np.max(mem_usage) - np.min(mem_usage)


def plot_memory_usage_per_batch_size():
    frequency, complex_dim, input_dim, output_dim = 20, 2, 44, 44
    # increments of 500 from 500 to 32000
    num_runs = 10  # Number of runs for averaging
    # batch_sizes = 2 ** np.arange(9, 16)  # powers of 2 starting at 512
    batch_sizes = np.arange(500, 32000, 500)
    # batch_sizes = [100, 200, 500, 1000]
    einsum_memory = []
    reshape_memory = []
    einsum_time = []
    reshape_time = []

    total_combinations = len(batch_sizes) * num_runs
    with tqdm(total=total_combinations) as pbar:
        for batch in batch_sizes:
            mem_usage_einsum = []
            mem_usage_reshape = []
            time_einsum = []
            time_reshape = []

            for _ in range(num_runs):
                x = np.random.rand(batch, frequency, complex_dim, input_dim)
                y_mats = np.random.rand(frequency, complex_dim, output_dim, output_dim)
                y_mats = prepare_admittance_matrices(y_mats, None)
                kernel = np.random.rand(frequency, complex_dim, input_dim, frequency, complex_dim, output_dim)

                start_time = time.time()
                mem_einsum = memory_profile(einsum_approach, x, kernel, y_mats)
                time_einsum.append(time.time() - start_time)
                mem_usage_einsum.append(mem_einsum)

                start_time = time.time()
                mem_reshape = memory_profile(reshape_approach, x, kernel, y_mats)
                time_reshape.append(time.time() - start_time)
                mem_usage_reshape.append(mem_reshape)
                pbar.update(1)

            einsum_memory.append(np.mean(mem_usage_einsum))
            reshape_memory.append(np.mean(mem_usage_reshape))
            einsum_time.append(np.mean(time_einsum))
            reshape_time.append(np.mean(time_reshape))

        # Create a plot with dual y-axes
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Memory Usage (MB)', color='tab:blue')
    ax1.plot(batch_sizes, einsum_memory, label='Einsum Memory', marker='o', linestyle='--', color='tab:blue')
    ax1.plot(batch_sizes, reshape_memory, label='Reshape Memory', marker='x', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (s)', color='tab:orange')
    ax2.plot(batch_sizes, einsum_time, label='Einsum Time', marker='o', linestyle='--', color='tab:orange')
    ax2.plot(batch_sizes, reshape_time, label='Reshape Time', marker='x', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=[0.15, 1], bbox_transform=ax1.transAxes)
    plt.title(f'Memory Usage and Processing Time averaged over {num_runs} runs')
    plt.show()


if __name__ == '__main__':
    plot_memory_usage_per_batch_size()
    # Generate random input data for testing
    # batch, frequency, complex_dim, input_dim, output_dim = 1000, 20, 2, 3, 44
    batch, frequency, complex_dim, input_dim, output_dim = 1000, 20, 2, 44, 44
    x = np.random.rand(batch, frequency, complex_dim, input_dim)
    # x = np.ones((batch, frequency, complex_dim, input_dim))
    y_mats = np.random.rand(frequency, complex_dim, output_dim, output_dim)
    # y_mats = np.zeros((frequency, complex_dim, output_dim, output_dim))
    # Set the diagonal elements to 1 using a loop over the first two axes
    for i in range(y_mats.shape[0]):
        for j in range(y_mats.shape[1]):
            np.fill_diagonal(y_mats[i, j], 1)
            y_mats[i, j, 0, 0] = 0
            y_mats[i, j, 1, 1] = 0
    kernel = np.random.rand(frequency, complex_dim, input_dim, frequency, complex_dim, output_dim)
    # kernel = np.ones((frequency, complex_dim, input_dim, frequency, complex_dim, output_dim))

    # y_mats = prepare_y_mats(y_mats, [3, 19, 22])
    y_mats = prepare_admittance_matrices(y_mats, None)
    # Execute both approaches and print the memory usage
    # matmul_approach(x, kernel, y_mats) # out of memory
    res0 = np.array(reshape_approach(x, kernel, y_mats))
    res1 = np.array(einsum_approach(x, kernel, y_mats))
    print(f"Results are equal: {np.allclose(res0, res1)}")
    pass
