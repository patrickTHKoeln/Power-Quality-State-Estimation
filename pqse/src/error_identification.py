import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def identify_phase_angle_problem(array4d):
    """
    Identify nodes where the phase angle is not close to the most common phase angle
    and the magnitude is close to zero
    :param array4d: 4d data
    :return: dictionary with frequencies as keys and node counts as values, and the most common phase angle
    """
    results = {}
    # List of specific frequencies to consider (0 is 50Hz, 1 is 100Hz, etc.)
    frequencies_to_check = [2, 5, 8, 11, 14, 17]

    # Loop through the specific frequencies
    for f in frequencies_to_check:
        # Initialize a dictionary for each frequency to store node counts
        freq_dict = {}

        # Extract 3D array corresponding to the frequency f
        slice_at_f = array4d[:, f, :, :]

        # Extract magnitude and phase components for all batches and nodes
        magnitude = slice_at_f[:, 0, :]
        phase = slice_at_f[:, 1, :]

        # Calculate the most common phase angle for the current frequency, taking all nodes into account
        mode_angle = stats.mode(phase.ravel())[0]

        # Specify the condition for close-to-zero magnitude (e.g., within a tolerance of 1e-5)
        mag_near_zero_indices = np.abs(magnitude) < 1e-5

        # Extract the phase values where the magnitude is close to zero
        phase_values_for_mag_near_zero = np.extract(mag_near_zero_indices, phase)

        # Find node indices where magnitude is close to zero
        node_indices = np.argwhere(mag_near_zero_indices)

        # Loop through unique nodes to count occurrences where the phase is not as expected
        for node in np.unique(node_indices[:, -1]):
            # Filter node_indices array for the current node
            filtered_indices = node_indices[node_indices[:, -1] == node]

            # Count occurrences where phase value is not close to the mode_angle (within ±0.5)
            # count = np.sum(np.abs(phase_values_for_mag_near_zero[filtered_indices[:, 0]] - mode_angle) > 0.5)
            count = len(filtered_indices[:, 0])
            # Add this count to the frequency's dictionary
            freq_dict[f'node {node}'] = count
        freq_dict['most_common_angle'] = mode_angle
        # Add freq_dict to the results dictionary
        results[f"{(f + 1) * 50}Hz"] = freq_dict
    return results


def plot_heatmap_phase_angle_nodes(results):
    """
    Plot a heatmap of the occurrences of phase angles not close to the most common phase angle
    :param results: dictionary with frequencies as keys and node counts as values, and the most common phase angle
    :return:
    """
    # List of unique frequencies and nodes
    frequencies = sorted([int(key.replace("Hz", "")) for key in results.keys()])
    nodes_in_array = sorted([-1 if key == 'most_common_angle' else int(key.replace("node ", "")) for key in
                             results[f'{frequencies[0]}Hz'].keys()])
    nodes = np.arange(0, 44, 1)

    # Initialize an empty matrix to store the counts
    count_matrix = np.zeros((len(nodes), len(frequencies)))

    # Populate the count_matrix with the occurrences
    for f_idx, f in enumerate(frequencies):
        for n_idx, node in enumerate(nodes):
            if node in nodes_in_array:
                count_matrix[n_idx, f_idx] = results[f"{f}Hz"][f"node {node}"]
            else:
                count_matrix[n_idx, f_idx] = 0

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    c = plt.imshow(count_matrix, interpolation='nearest', cmap='hot', aspect='auto')
    plt.colorbar(c, label='Occurences of magnitude close to zero')

    # Label the axes
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Node')
    plt.title('Occurrences of magnitude close to zero per node and third harmonic')

    # Assign labels to x and y axes
    plt.xticks(np.arange(len(frequencies)), [str(f) for f in frequencies], rotation=45)
    plt.yticks(np.arange(len(nodes)), [str(node) for node in nodes])

    plt.show()
