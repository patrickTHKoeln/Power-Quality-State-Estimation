import os

import matplotlib.pyplot as plt
import numpy as np
from PIL.features import features
from ml_multi_dim_toolbox import SplitComplexMode


# TODO Move the whole class somewhere else and create a unified data format for power grid simulation data
def plot_data_distribution(data, node, frequencies, mode='y', complex_mode=SplitComplexMode.CARTESIAN,
                           normalized=False, save_path=None, show_plot=False, no_columns=0, polar_plot=False):
    """
    Plots the data distribution of power grid simulations.

    :param data: dictionary with keys
        - 'split_complex_modes' list of complex representation modes
        - any of 'x_train', 'x_test', 'x_val', 'y_train', 'y_test', 'y_val' or with suffix '_n' if normalized
        - optional 'mask_x_test', 'mask_y_test' as lists of masked nodes

    :param node: int, specifies which node to plot
    :param frequencies: frequencies present in the data (must be same length as frequencies in data)
    :param mode: 'x' for input 'y' for target
    :param complex_mode: mode to plot
    :param normalized: if True plots normalized data (requires normalized data fields in data)
    :param save_path: path to save plot to
    :param show_plot: if True shows plot
    :return:
    """

    if mode not in ['x', 'y']:
        raise ValueError('Mode must be >x< for input or >y< for target')
    is_3_phase = len(data[mode + '_train'].shape) == 5
    # kinda hacky way to determine where in the last dimension of the array which complex feature is
    feature_x = data['split_complex_modes'].index(complex_mode) * 2
    feature_y = data['split_complex_modes'].index(complex_mode) * 2 + 1
    # colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}
    colors = {'train': 'tab:blue', 'test': 'tab:green'}
    markers = {'train': 'o', 'val': 'x', 'test': 'x'}
    facecolors = {'train': 'None', 'val': 'None', 'test': 'tab:green'}
    node_index_test_set = node
    if mode == 'x' and 'mask_x_test' in data:
        node_index_test_set = data['mask_x_test'].index(node)
    if mode == 'y' and 'mask_y_test' in data:
        node_index_test_set = data['mask_y_test'].index(node)

    # Determine grid size for subplots
    if no_columns > 0:
        no_rows = int(np.ceil(len(frequencies)/no_columns))
    else:
        grid_size = int(np.ceil(np.sqrt(len(frequencies))))
        no_rows = grid_size
        no_columns = grid_size

    range_phases = [0]
    if is_3_phase:
        range_phases = [0, 1, 2]

    for phase in range_phases:
        if polar_plot:
            fig, axes = plt.subplots(no_rows, no_columns, figsize=(4 * no_columns, 4 * no_rows),
                                     subplot_kw={'projection': 'polar'})
        else:
            fig, axes = plt.subplots(no_rows, no_columns, figsize=(4 * no_columns, 4 * no_rows))
        fig.suptitle(f'Voltage Data Distribution [V] -  Node {node} - Phase {phase+1}', fontsize=16)
        for h in range(len(frequencies)):
            row = h // no_columns
            col = h % no_columns
            ax = axes[row, col]
            list_plot_log_data = []
            max_mag = 1e-9
            min_mag = 1e9
            for set_name in colors.keys():
                key = f'{mode}_{set_name}_n' if normalized else f'{mode}_{set_name}'
                node_index = node_index_test_set if set_name == 'test' else node
                data_set = data[key]
                if not is_3_phase:
                    data_set = np.expand_dims(data_set, axis=3)
                if polar_plot:
                    x_data = np.angle(data_set[:, node_index, h, phase, feature_x] + data_set[:, node_index, h, phase, feature_y]*1j, deg=False)
                    y_data = np.sqrt(np.square(data_set[:, node_index, h, phase, feature_y]) +
                                     np.square(data_set[:, node_index, h, phase, feature_x]))
                    list_plot_log_data.append({'x': x_data, 'y': y_data,
                                               'desc': {"alpha": 0.5, "label": set_name, "color": colors[set_name],
                                                        "marker": markers[set_name], "facecolors": facecolors[set_name]}})

                    max_mag = max(np.max(y_data), max_mag)
                    min_mag = min(np.min(y_data), min_mag)
                    if min_mag == 0.0:
                        min_mag = y_data[y_data>0.0].min()
                    print(f'Harmonic {h+1}: {len(y_data<1e-6)} values smaller 1e-6')
                else:
                    x_data = data_set[:, node_index, h, phase, feature_x]
                    y_data = data_set[:, node_index, h, phase, feature_y]
                    ax.scatter(x_data, y_data, alpha=0.5, label=set_name, color=colors[set_name], marker='o', facecolors='none')

            ax.set_title(f'Harmonic {h + 1}')
            if not polar_plot:
                ax.set_xlabel(get_feature_label(feature_x))
                ax.set_ylabel(get_feature_label(feature_y))
            else:
                min_log = format_log_axis(ax, max_mag, min_mag)
                for p in list_plot_log_data:
                    add_log_data(ax, p['x'], p['y'], min_log, **p['desc'])

        # Remove empty subplots
        for h in range(len(frequencies), no_columns * no_rows):
            fig.delaxes(axes.flatten()[h])

        # Add legend to a single subplot (the last one in this case)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'data_dist_{complex_mode.name}_{mode}_n{node}_{feature_x}_{feature_y}_L{phase+1}'))
        if show_plot:
            plt.show()
        plt.close()


def add_log_data(ax, theta, r_, min_log, **kwargs):
    r = np.log10(r_) - min_log
    ax.scatter(theta, r, **kwargs)


def format_log_axis(ax, max_m, min_m):
    min10 = np.log10(min_m)
    max10 = np.log10(max_m)
    min_mag = np.floor(min10)
    max_mag = np.ceil(max10)

    # if only one decade would be shown, extend it to 3
    if min_mag == max_mag - 1:
        min_mag -= 1

    l = np.arange(min_mag, max_mag+1)
    reduce_labels = len(l) > 5
    ax.set_rticks(l - min_mag)
    if not reduce_labels:
        ax.set_yticklabels(['$10^{'+str(int(x))+'}$' for x in l])
    else:
        ax.set_yticklabels(['$10^{' + str(int(x)) + '}$' if idx % 2 == 0 else "" for idx, x in enumerate(l)])
    ax.set_rlim(0, l[-1] - min_mag)
    return min_mag


def get_feature_label(feature):
    # TODO move to split_complex file
    labels = [
        'Re{V}',
        'Im{V}',
        'Voltage Magnitude',
        r'Imaginary Part of $e^{j\phi}$'
    ]
    return labels[feature]


def plot_nodal_voltage_distribution(bus, harmonic_order, magnitude_or_phase, data_lab, data_sim, save_path, show_plot):
    """
    Plots the nodal voltage distribution of power grid simulations.
    :param bus: Current bus
    :param harmonic_order: Current order
    :param magnitude_or_phase: 3 phase np array
    :param data_lab: 3 phase np array from real measurements
    :param data_sim: 3 phase np array  from simulation
    :param save_path: Where to save the plot
    :param show_plot: show the graph
    """
    x_unit = 'Voltage [V]' if magnitude_or_phase == 'magnitude' else 'Angle [°]'
    # Subplots erstellen
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(0, 3):
    # Erster Subplot mit dem ersten Paar von Datengruppen
        axs[i].hist(data_lab[:, i], bins=30, alpha=0.5, label='Lab Data', edgecolor='black')
        axs[i].hist(data_sim[:, i], bins=30, alpha=0.5, label='Sim Data', edgecolor='black')
        axs[i].set_title(f'Phase {i+1}')
        axs[i].set_xlabel(x_unit)
        axs[i].set_ylabel('Occurrences [ ]')
        axs[i].legend()
    fig.suptitle(f'Comparison of {magnitude_or_phase} for bus {bus}, order {harmonic_order}')
    # Layout anpassen
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'bus_{bus}_harm_{harmonic_order}_{magnitude_or_phase}.png'))
    # Diagramm anzeigen
    if show_plot:
        plt.show()
    plt.close()
