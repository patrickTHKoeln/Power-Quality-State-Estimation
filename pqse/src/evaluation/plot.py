import math
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from networkx.algorithms.bipartite.basic import degrees
from scipy.stats import norm

from evaluation.error import get_max_error_split_axis, get_mae_split_axis, calculate_error_metric, calculate_thd
from ml_multi_dim_toolbox import cartesian_to_polar


def scatter_plot_per_frequency(array_4d, n_samples=5000, x_label="Real", y_label="Imag", title="",
                               frequencies_to_plot=None, save_path=None, show_plot=True):
    """
    Plots a scatter plot for each frequency in the given array of shape
    (batch, frequencies, 2, nodes)
    :param array_4d:
    :param n_samples: amount of samples to plot
    :param x_label: label for x axis
    :param y_label: label for y axis
    :param title: figure title
    :param frequencies_to_plot: list of frequencies to plot
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    import matplotlib.pyplot as plt
    if frequencies_to_plot is None:
        frequencies_to_plot = np.arange(array_4d.shape[1])
        num_cols = 4  # number of columns for the multi-plot
        num_rows = len(frequencies_to_plot) // num_cols if len(frequencies_to_plot) % num_cols == 0 else len(
            frequencies_to_plot) // num_cols + 1
    else:
        num_cols = 1
        num_rows = len(frequencies_to_plot)
    # prepare a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axs = axs.flatten()  # make it a 1D array for easy iteration
    i = 0
    for freq in frequencies_to_plot:
        # gather the first n_samples elements, or all if fewer than n_samples
        num_elements = min(n_samples, array_4d.shape[0] * array_4d.shape[3])
        x = array_4d[:num_elements // array_4d.shape[3], freq, 0, :].flatten()
        y = array_4d[:num_elements // array_4d.shape[3], freq, 1, :].flatten()

        # create scatter plot in the current subplot
        axs[i].scatter(x, y, s=1)

        # label axes
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel(y_label)

        # set title to frequency
        # TODO: make dynamic (e.g. for interharmonics and other fundamental frequencies)
        axs[i].set_title(f"{str((int(freq) + 1) * 50)} Hz")
        i += 1

    # handle case where num_frequencies < num_cols * num_rows
    for i in range(len(frequencies_to_plot), num_cols * num_rows):
        fig.delaxes(axs[i])

    # display the multi-plot figure
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, title + '.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_outliers(array_4d, save_path=None, show_plot=True):
    """
    Identify outliers by calculating the IQR and plotting the data with outliers
    in a separate color.
    :param array_4d: 4d input data
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    outlier_indices_count = Counter()

    # Iterate over each 3D array in the 4D array
    for i in range(array_4d.shape[1]):
        for j in range(array_4d.shape[2]):
            fig, axs = plt.subplots(11, 4, figsize=(20, 50))  # create a grid of 44 subplots (11 rows, 4 columns)
            axs = axs.ravel()  # flatten the grid to an array for easier indexing
            for k in range(array_4d.shape[3]):
                # Select a specific 3D array
                data_slice = array_4d[:, i, j, k]

                # Calculate the IQR
                Q1 = np.percentile(data_slice, 25, method='midpoint')
                Q3 = np.percentile(data_slice, 75, method='midpoint')
                IQR = Q3 - Q1

                # Define the outlier bounds
                lower_bound = Q1 - 10 * IQR
                upper_bound = Q3 + 10 * IQR

                # Identify the outliers
                outlier_indices = np.where((data_slice < lower_bound) | (data_slice > upper_bound))
                outlier_values = data_slice[outlier_indices]

                # Count the occurrence of each index
                outlier_indices_count.update(outlier_indices[0])

                # Visualize the data
                axs[k].plot(data_slice, 'bo', label='Data')
                if outlier_values.size > 0:  # If there are outliers, plot them
                    axs[k].plot(outlier_indices[0], outlier_values, 'ro', label='Outliers')
                axs[k].set_title(f"Outliers in data[:, {i}, {j}, {k}]")
                axs[k].set_xlabel('Index')
                axs[k].set_ylabel('Value')
                axs[k].legend()

            # Adjust the spacing between subplots
            plt.tight_layout()
            # Show the plots
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(os.path.join(save_path, f"outliers_{i}_{j}.png"))
            if show_plot:
                plt.show()
            plt.close()

    # Print the 10% most common outlier indices
    print("Most common outlier indices on axis 0:")
    for index, count in outlier_indices_count.most_common(int(len(outlier_indices_count) * 0.1)):
        print(f"Index: {index}, Count: {count}")


def plot_history(data, mode='val_loss', save_path=None, show_plot=True):
    """
    Plot the loss history of a model
    :param data: loss history
    :param mode: 'val_loss' or 'loss'
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    df = pd.DataFrame(data=data)
    ax = df.plot(figsize=(11.7, 8.27), logx=True, logy=True)
    if mode == 'val_loss':
        ax.set_title('Validation Loss over Epochs')
    elif mode == 'loss':
        ax.set_title('Training Loss over Epochs')
    else:
        raise ValueError("mode must be either 'val_loss' or 'loss'")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig = ax.get_figure()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{mode}.png"))
    if show_plot:
        plt.show()
    plt.close()


def plot_train_val_loss_history(df_train, df_val, save_path=None, show_plot=True):
    """
    Plot the training and validation loss history of models
    :param df_val: validation loss history as pandas dataframe with key model name
    :param df_train: training loss history as pandas dataframe with key model name
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """

    ax = None
    colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(df_train.columns)))

    for col, color in zip(df_train.columns, colors):
        ax = df_train[col].plot(figsize=(11.7, 8.27), logx=False, logy=True, ax=ax, color=color, label=col + '_train',
                                linestyle='-')
        df_val[col].plot(logx=False, logy=True, ax=ax, color=color, label=col + '_val', linestyle='--')

    # Add title, labels, and legend
    ax.set_title('Training and Validation Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (Log Scale)')
    ax.legend()

    fig = ax.get_figure()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"train_val_loss.png"))
    if show_plot:
        plt.show()
    plt.close()


def plot_log_histogram(arr, title, xlabel, ylabel, save_path=None, show_plot=True):
    """
    Plot a histogram with logarithmic y-axis
    :param arr: data
    :param title: plot title
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    plt.figure(figsize=(11.7, 8.27))
    plt.hist(arr, bins=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    if show_plot:
        plt.show()
    plt.close()


def plot_histogram_weights(model, x_train, y_train, save_path=None, show_plot=True):
    """
    Plot the histogram of the weights and gradients for each layer by using the GradientTape.
    :param model: tensorflow model
    :param x_train: input training data
    :param y_train: output training data
    :param save_path: if provided, plot will be saved at specified location, also loads gradients from save_path
    :param show_plot: if True plot will be shown
    :return:
    """

    # Initialize GradientTape
    def compute_loss(y_true, y_pred):
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        predictions = model(x_train)
        loss = compute_loss(y_train, predictions)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    if save_path is not None:
        with open(os.path.join(save_path, f"gradients.pkl"), 'wb') as f:
            pickle.dump(gradients, f)
    n_trainable_layers = 0
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]  # extract weights from the layer
            # Plot histogram
            plt.figure(figsize=(11.7, 8.27))
            plt.hist(np.ravel(weights), bins=100)
            plt.title(layer.name + ' - Weight distribution')
            plt.xlabel('Weight value')
            plt.ylabel('Frequency')
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"weight_histogram_{layer.name}.png"))
            if show_plot:
                plt.show()
            plt.close()

            # Plot gradient histogram
            layer_gradient = gradients[n_trainable_layers].numpy()  # Corresponding layer gradient
            plot_log_histogram(np.ravel(layer_gradient), layer.name + ' - Gradient distribution',
                               'Gradient value', 'Frequency (logarithmic)', save_path, show_plot)
            n_trainable_layers += 1


def plot_gradients_model_comparison(gradients0, gradients1, save_path=None, show_plot=True):
    """
    Plot gradients comparing two different models
    :param gradients0:
    :param gradients1:
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    n_layers = len(gradients0)
    n_rows = 4
    n_cols = (n_layers + 1) // n_rows
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, (grad0, grad1, ax) in enumerate(zip(gradients0, gradients1, axes)):
        grad0_flat = np.ravel(grad0)
        grad1_flat = np.ravel(grad1)

        # Determine the bin edges
        bins = np.histogram(np.hstack((grad0_flat, grad1_flat)), bins=500)[1]

        # Calculate the histograms
        hist0, _ = np.histogram(grad0_flat, bins=bins, density=True)
        hist1, _ = np.histogram(grad1_flat, bins=bins, density=True)

        # Manually plot the histograms in the subplot
        ax.step(bins[:-1], hist0, where='post', alpha=0.5, label='with redistribution')
        ax.step(bins[:-1], hist1, where='post', alpha=0.5, label='without redistribution')

        # Set y-axis to log scale
        ax.set_yscale("log")

        ax.set_xlabel('Gradient Value')
        ax.set_title(f'Layer {i + 1}')

    # Add common y-label and title
    fig.text(0.005, 0.5, 'Frequency Density (log)', va='center', rotation='vertical')
    plt.suptitle('Gradient comparison across Admittance Weighted Layers')

    # Add legend
    axes[0].legend()

    # Save the plot if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "log_gradient_comparison_all_layers.png"))
    if show_plot:
        plt.tight_layout()
        plt.show()
    plt.close()


def plot_histogram_residuals(res_n_flat, title="", save_path=None, show_plot=True):
    """
    Plot the histogram of the residuals.
    :param res_n_flat: flattened residuals
    :param title: plot title
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return: None
    """
    std_dev_n = np.std(res_n_flat)
    mean_n = np.mean(res_n_flat)
    error_range = np.linspace(min(res_n_flat), max(res_n_flat), 1000)
    pdf_values = norm.pdf(error_range, mean_n, std_dev_n)
    # Plotting error distribution as histograms for normalized and non-normalized data
    fig, ax = plt.subplots()
    ax.hist(res_n_flat, bins=100, density=True, edgecolor='black', alpha=0.7)
    ax.plot(error_range, pdf_values, color='blue', linestyle='-', linewidth=0.7)
    colors = ['r', 'g', 'b', 'm', 'c']
    for i in range(1, 5):
        ax.axvline(x=mean_n + i * std_dev_n, color=colors[i - 1], linestyle='--', linewidth=0.7, label=f'{i}σ')
        ax.axvline(x=mean_n - i * std_dev_n, color=colors[i - 1], linestyle='--', linewidth=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend(loc='upper right', title='Standard Deviations')
    if save_path is not None:
        save_path = os.path.join(save_path, 'error_distribution_normalized.png')
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_histogram(data, title="", meas='Error', save_path=None, show_plot=True):
    """
    Plot the histogram of the residuals.
    :param data: data to be quantified
    :param title: plot title
    :param meas: The measure for the Error
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return: None
    """

    # Plotting error distribution as histograms for normalized and non-normalized data
    fig, ax = plt.subplots()
    ax.hist(data, bins=100, edgecolor='black', alpha=0.7)

    ax.set_xlabel(meas)
    ax.set_ylabel('Frequency')
    ax.set_title(title)

    if save_path is not None:
        save_path = os.path.join(save_path, f'{title}.png')
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_pdf(data, title="", save_path=None, show_plot=True):
    """
    Plot the histogram of the residuals.
    :param data: data to be quantified
    :param title: plot title
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return: None
    """
    std_dev_n = np.std(data)
    mean_n = np.mean(data)
    error_range = np.linspace(min(data), max(data), 1000)
    pdf_values = norm.pdf(error_range, mean_n, std_dev_n)
    # Plotting error distribution as histograms for normalized and non-normalized data
    fig, ax = plt.subplots()
    ax.plot(error_range, pdf_values, color='blue', linestyle='-', linewidth=0.7)
    colors = ['r', 'g', 'b', 'm', 'c']
    for i in range(1, 5):
        ax.axvline(x=mean_n + i * std_dev_n, color=colors[i - 1], linestyle='--', linewidth=0.7, label=f'{i}σ')
        ax.axvline(x=mean_n - i * std_dev_n, color=colors[i - 1], linestyle='--', linewidth=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend(loc='upper right', title='Standard Deviations')
    if save_path is not None:
        save_path = os.path.join(save_path, 'Histogram.png')
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_cdf(prediction_losses, label='MAPE', name='Different Models', dashed_line_str='input noise', color_str='Gauss Layer 0.02',
             save_path=None, show_plot=True):
    """
    Plot the CDF of the <label> for each model.
    :param prediction_losses: dictionary of model name -> list of flattened MAPEs, MAEs, ...
    :param label: MAPE, MAE or similar
    :param name: for which data is this Comparison?
    :param dashed_line_str: optional parameter that results in dashed lines if in model name
    :param color_str: optional parameter for different color if in model name
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return: None
    """
    # Flag to check if the specific string is encountered at least once
    specific_string_found = False

    for model_name, losses in prediction_losses.items():
        # Sort the losses
        sorted_losses = np.sort(losses)

        # Generate the CDF
        cdf = np.arange(1, len(sorted_losses) + 1) / float(len(sorted_losses))
        if color_str == "":
            color = None
        else:
            if color_str in model_name:
                color = 'r'
            else:
                color = 'b'
        # Check for the specific string and plot accordingly
        if dashed_line_str in model_name:
            plt.plot(sorted_losses, cdf, '--', color=color, label=model_name)
            specific_string_found = True
        else:
            plt.plot(sorted_losses, cdf, color=color, label=model_name)

    # Customize the graph
    plt.title(f'CDF for {name}')
    plt.xscale('log')
    # ax = plt.gca()
    # ax.set_xlim([0, 100])
    if label == "MAE":
        plt.xlabel(f'Mean Absolute Error ({label})')
    elif label == "MAPE":
        plt.xlabel(f'Mean Absolute Percentage Error ({label})')
    else:
        plt.xlabel(f'{label}')

    plt.ylabel('Cumulative Distribution Function (CDF)')

    # Add the legend for the specific string if it is found
    if specific_string_found:
        plt.legend(title=f'Dashed Line = {dashed_line_str}')
    else:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'cdf_{label}_{name}.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_estimations_wrapper(predictions, grid_name, y_test, x_test, mask_x_test, mask_y_test=None, iteration=1,
                             input_noise=True, harmonic=1, save_path=None, show_plot=True, phase=0):
    """

    :param predictions: dictionary of key <Model identifier> and value <predictions dict>
    :param grid_name: name of the grid
    :param y_test: in cartesian form
    :param x_test: in cartesian form
    :param mask_x_test: mask indices of x_test
    :param mask_y_test: mask indices of y_test, default None = all buses
    :param iteration: which iteration to plot
    :param harmonic: which harmonic to plot
    :param input_noise: if measurement noise is added to the input, set to True to get accurate labels in plots
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :param phase: Which phase should be plotted
    :return:
    """
    y_preds = {}

    real_index = int(phase * 2)
    imag_index = [1, 3, 5]

    for key, value in predictions.items():
        y_preds[key] = cartesian_to_polar(value['y_pred'], axis=4)[iteration, :, harmonic - 1, phase, 0]
    y_test = cartesian_to_polar(y_test, axis=4)
    y_test_single_it = y_test[iteration, :, harmonic - 1, phase, 0]
    x_test = cartesian_to_polar(x_test, axis=4)
    x_test_single_it = x_test[iteration, :, harmonic - 1, phase, 0]
    title = f"Estimation harmonic {harmonic} L{phase+1} for {grid_name} at iteration {iteration}"
    if save_path is not None:
        path = os.path.join(save_path, f'estimations_it{iteration}_p{phase+1}_h{harmonic}.png')
        os.makedirs(save_path, exist_ok=True)
    else:
        path = None
    plot_estimations(y_test_single_it, x_test_single_it, y_preds, title, mask_x_test, mask_y_test, save_path=path,
                     show_plot=show_plot, input_noise=input_noise)


def plot_estimations(y_test_polar, x_test_polar, y_pred_polar, title, mask_x_test, mask_y_test=None, input_noise=True,
                     save_path=None, show_plot=True):
    """
    Plot the true values, the predictions and the noiseful input for each bus.
    :param y_test_polar: true values
    :param x_test_polar: noiseful input
    :param y_pred_polar: dictionary of model name -> predictions
    :param title: Plot title
    :param mask_x_test: mask indices of x_test (i.e. measurement locations)
    :param mask_y_test: mask indices for y_test (i.e. true values), default None = all buses
    :param input_noise: if True, noise is added to the input, then the label is changed in this method
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    bus_numbers = list(range(0, len(next(iter(y_pred_polar.values())))))  # somewhat hacky way to get all nodes
    if mask_y_test is None:
        mask_y_test = bus_numbers
    plt.figure(figsize=(10, 6))

    # Plotting y_test with outlined symbols
    plt.scatter(mask_y_test, y_test_polar, label='True value', edgecolors='r', marker='o', facecolors='none')

    # Plotting y_pred_polar considering multiple models with outlined symbols
    markers = ['s', '^', 'D', 'h', '2']  # Additional markers can be added as needed
    edgecolors = ['g', 'orange', 'brown', 'purple', 'darkred']
    for idx, (model_name, y_pred) in enumerate(y_pred_polar.items()):
        plt.scatter(bus_numbers, y_pred, label=f'Prediction ({model_name})', edgecolors=edgecolors[idx],
                    marker=markers[idx % len(markers)], facecolors='none')

    # Plotting x_test considering indices with outlined symbols
    if mask_x_test is not None:
        label = 'Noiseful Input' if input_noise else 'Input'
        plt.scatter(mask_x_test, x_test_polar, label=label, facecolors='b', marker='x')

    # Setting x-axis and y-axis labels
    plt.xlabel('Nodes')
    plt.ylabel('Voltage Magnitude [V]')
    plt.title(title)
    # Setting y-axis limits to show only relevant parts
    all_values = np.concatenate(
        [y_test_polar, x_test_polar if mask_x_test is not None else []] + list(y_pred_polar.values()))
    lower_limit = np.min(all_values) - 0.1 * (np.max(all_values) - np.min(all_values))
    upper_limit = np.max(all_values) + 0.1 * (np.max(all_values) - np.min(all_values))
    plt.ylim([lower_limit, upper_limit])

    # Adding a legend
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.close()


def evaluate(predictions, data, input_noise=True, save_path=None, show_plot=True):
    """
    Evaluate predictions
    :param predictions: dictionary of model predictions with model label as key and prediction data as value
    :param data: dictionary of network data
    :param input_noise: if measurement noise is added to the input, set to True to get accurate labels in plots
    :param save_path: path to save plots to, default None
    :param show_plot: True if plots shall be shown
    :return:
    """
    complex_axis = 4
    # convert to polar
    y_preds = {}
    for key, value in predictions.items():
        y_preds[key] = cartesian_to_polar(value['y_pred'], axis=complex_axis)

    ## plot estimations for any iteration here
    ## to get the time of the day of a given iteration (in 24h format) calculate: (iteration % (4*24))/4
    for i in [0, 6, 21, 50, 63]:
        for j in [1, 3, 5, 7, 9, 15]:
            plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                                     data['mask_y_test'],
                                     iteration=i, harmonic=j, input_noise=input_noise, save_path=save_path,
                                     show_plot=show_plot)


def plot_losses(losses, save_path=None, show_plot=True):
    if losses is not None and len(losses) > 0:
        # plot loss history
        val_loss = pd.DataFrame({key: val['val_loss'] for key, val in losses.items()})
        train_loss = pd.DataFrame({key: val['loss'] for key, val in losses.items()})
        plot_history(val_loss, mode='val_loss', save_path=save_path, show_plot=show_plot)
        plot_history(train_loss, mode='loss', save_path=save_path, show_plot=show_plot)
        plot_train_val_loss_history(train_loss, val_loss, save_path=save_path, show_plot=show_plot)
        print(f"Minimum val loss:{val_loss.min()}")


def plot_heatmaps(predictions, data, complex_axis=3, magnitude_only=False, mask=None, save_path=None, show_plot=True):
    # plots for each model:
    for key, value in predictions.items():
        # convert to polar
        y_pred_polar = cartesian_to_polar(value['y_pred'], axis=complex_axis, wrap_phase=True, angle_type='degrees')
        y_test_polar = cartesian_to_polar(data['y_test'], axis=complex_axis, wrap_phase=True, angle_type='degrees')

        for phase in range(0, 3):
            df_titles = [f'L{phase+1} Mean Absolute Error of Magnitude [V]', f'L{phase+1} Mean Absolute Error of Angle [°]']
            # MAE plots
            dfs = get_mae_split_axis(y_test_polar, y_pred_polar, data['frequencies'], feature_axis=complex_axis,
                                     error_axis=0, phase=phase)
            if magnitude_only:
                dfs = (dfs[0],)

            plot_heatmap_subplots(list(dfs)[0], df_titles,
                          title=f"{key} MAE L{phase+1}",
                          save_path=save_path, show_plot=show_plot, major_interval_count=5, mask=mask)

            plot_heatmap(list(dfs), df_titles,
                         title=f"{key} MAE L{phase+1}",
                         save_path=save_path, show_plot=show_plot, mask=mask)
            plot_heatmap(list(dfs), df_titles,
                         title=f"{key} MAE L{phase+1}", drop_fundamental=True,
                         save_path=save_path, show_plot=show_plot, mask=mask)

            # Max error plots
            dfs = get_max_error_split_axis(y_test_polar, y_pred_polar, data['frequencies'], feature_axis=complex_axis,
                                           error_axis=0, phase=phase, phase_angle='degrees')
            df_titles = [f'L{phase+1} Maximum Absolute Error of Magnitude [V]', f'L{phase+1} Maximum Absolute Error of Angle [°]']
            if magnitude_only:
                dfs = (dfs[0],)
            plot_heatmap_subplots(list(dfs)[0], df_titles,
                          title=f"{key} Max Error L{phase+1}",
                          save_path=save_path, show_plot=show_plot, major_interval_count=5, mask=mask)
            plot_heatmap(dfs, df_titles,
                         title=f"{key} Max Error L{phase+1}",
                         save_path=save_path, show_plot=show_plot, mask=mask)
            plot_heatmap(dfs, df_titles,
                         title=f"{key} Max Error L{phase+1}", drop_fundamental=True,
                         save_path=save_path, show_plot=show_plot, mask=mask)


def plot_heatmap(dfs, df_titles, title="", drop_fundamental=False, mask=None, save_path=None, show_plot=True):
    """
    Plot dataframes of shape (batch, frequencies, nodes) as heatmaps
    :param dfs: list of pandas dataframes
    :param df_titles: list of strings, titles for each dataframe
    :param title:
    :param drop_fundamental: if True fundamental frequency is dropped
    :param mask: node mask if not all nodes are in evaluation dataframe or nodes are not ordered, else nodes are ordered numerically
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(nrows=len(dfs), figsize=(11.7, 8.27))
    for i, df in enumerate(dfs):
        if drop_fundamental:
            column_max_val = df.max().idxmax()
            if np.isclose(float(column_max_val), 50, rtol=1e-3) or np.isclose(float(column_max_val), 60, rtol=1e-3):
                df = df.drop(column_max_val, axis=1)
        # df.columns = df.columns.astype(int)  # parse columns as integers
        if mask is not None:
            df.index = mask  # fix node numbers
        if len(dfs) == 1:
            s = sns.heatmap(df, ax=axs)
        else:
            s = sns.heatmap(df, ax=axs[i])
        s.set(xlabel='Frequency [Hz]', ylabel='Node', title=df_titles[i])
    # fig.suptitle(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if drop_fundamental:
            save_path = os.path.join(save_path, title + "_drop-fund_" + '.png')
        else:
            save_path = os.path.join(save_path, title + '.png')
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_heatmap_subplots(df, df_titles, title="", save_path=None, show_plot=True, major_interval_count=5, mask=None):
    """
    Plot fundamental and harmonics heatmaps on two scales in one plot
    :param df: pandas dataframe with the dataset
    :param df_titles: list of strings, titles for each dataframe
    :param title:
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :param major_interval_count: how many intervalls shall the new colorbar have
    :param mask: node mask if not all nodes are in evaluation dataframe or nodes are not ordered, else nodes are ordered numerically
    :return:
    """
    if mask is not None:
        df.index = mask  # fix node numbers
    plt.rcParams.update({'font.size': 16})
    fig_height = 9 if len(df.index) == 44 else 7
    fig, axs = plt.subplots(ncols=3, figsize=(14, fig_height), gridspec_kw={'width_ratios': [1, 1, 19], 'wspace': 0.25})

    column_max_val = df.max().idxmax()
    df_only_fundamental = df[[column_max_val]]
    df_drop_fundamental = df.drop(column_max_val, axis=1)

    df_min = df_only_fundamental.min().min()
    df_max = df_only_fundamental.max().max()

    # Interpolierte Werte für eigene Skala erstellen
    interpolated_values = np.linspace(df_max, df_min, 1000)
    interpolated_series = np.linspace(1000, 1, 1000)
    df_scale = pd.DataFrame(interpolated_series)
    df_scale.index = interpolated_values

    # Add a heatmap as custom colorbar
    s_f_scale = sns.heatmap(df_scale, ax=axs[0], cbar=False, cmap='viridis')
    new_ticks = []
    new_tick_labels = []
    old_ticks = s_f_scale.get_yticks()
    delta_ticks = (old_ticks[-1] - old_ticks[0])/major_interval_count
    order_of_mag_max = math.floor(math.log10(df_max))
    if order_of_mag_max > 0:
        df_max_start = (df_max - (df_max % 5))
        delta_tick_labels = (df_max_start - (df_min - (df_min % 5)))/major_interval_count
    else:
        df_max_start = df_max # round(df_min + (df_max-df_min)*0.9, abs(order_of_mag_max)+1)
        df_min_start = df_min # round(df_min + (df_max-df_min)*0.1, abs(order_of_mag_max)+1)
        delta_tick_labels = (df_max_start - df_min_start)/major_interval_count

    for i in range(0, major_interval_count+1):
        tick = old_ticks[0] + delta_ticks * i
        new_ticks.append(tick)
        label = df_max_start - delta_tick_labels*i
        if order_of_mag_max > 0:
            new_tick_labels.append(str(int(math.trunc(label))))
        else:
            new_tick_labels.append(str(round(label, abs(order_of_mag_max)+1)))
    axs[0].set_yticks(new_ticks, new_tick_labels)

    axs[0].set_ylabel('')  # Y-Achsenbeschriftung entfernen
    axs[0].set_xlabel('')  # X-Achsenbeschriftung entfernen
    axs[0].set_xticks([])  # X-Achsenticks entfernen

    # Actual data
    s_f = sns.heatmap(df_only_fundamental, ax=axs[1], cbar=False, cmap='viridis')
    s_h = sns.heatmap(df_drop_fundamental, ax=axs[2])

    axs[2].set_yticklabels([])
    s_f.set(ylabel='Node')
    plt.setp(s_f.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(s_f.yaxis.get_majorticklabels(), rotation=0)
    plt.title(df_titles[0])
    # plt.text(1.0, 900, 'Frequency [Hz]')  # 0.5 0.05
    s_h.set(xlabel='Frequency [Hz]')
    plt.yticks(rotation=0)
    s_h.tick_params(labelrotation=45)
    # plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, title + "_combined_" + '.png')

        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_thd_bus(y_true:np.ndarray, y_pred:np.ndarray, orders_to_evaluate:list[int], plot_buses:list[int], time_span='day',
                 data_start_time=datetime(2024, 1, 1), data_resolution=(15, 'm'),
                 plot_start_time=datetime(2024, 1, 4), plot_end_time=datetime(2024, 1, 5),
                 mask=None, save_path=None, show_plot=True, f_name='Model'):
    """
    Plot the THD of the true and predicted data for the specified buses.
    :param y_true: True data
    :param y_pred: Predicted data
    :param orders_to_evaluate: harmonic orders to evaluate
    :param plot_buses: the buses to plot
    :param time_span: one of 'day', 'week', 'year', or None (where None plots all data on the x-axis)
    :param data_start_time: datetime object representing the start time of the whole data
    :param data_resolution: resolution of the data as tuple of (time_interval, time_unit)
    for now only minutes and seconds are supported ['m', 's'] # TODO: support more time units
    :param plot_start_time: datetime object representing the start time of the plot
    :param plot_end_time: datetime object representing the end time of the plot
    :param mask: bus mask where busses exist in y true
    :param save_path: path to save the plot
    :param show_plot: if True, plot will be shown
    :param f_name: filename
    :return:
    """
    thd_true = calculate_thd(y_true, orders_to_evaluate)
    thd_est = calculate_thd(y_pred, orders_to_evaluate)
    num_phases = y_true.shape[3]
    for i, bus in enumerate(plot_buses):
        if time_span is not None:
            # Determine total number of data points based on data resolution
            total_minutes = (plot_end_time - plot_start_time).total_seconds() / 60
            resolution_minutes = data_resolution[0] if data_resolution[1] == 'm' else data_resolution[0] * 60
            num_points = int(total_minutes / resolution_minutes)

            # Time array
            time_of_day = [data_start_time + timedelta(minutes=i * resolution_minutes) for i in range(num_points)]

            # Create subplots
            fig, axs = plt.subplots(num_phases, 1, figsize=(10, 8), sharex=True)

            start_idx = int((plot_start_time - data_start_time).total_seconds() / 60 / resolution_minutes)
            end_idx = start_idx + num_points

            for p_idx in range(0, num_phases):
                bus_idx = bus if mask is None else plot_buses.index(bus)
                thd_data_true = thd_true[start_idx:end_idx, bus_idx, p_idx]
                thd_data_est = thd_est[start_idx:end_idx, bus, p_idx]

                # Plotting the data for each subplot
                axs[p_idx].plot(time_of_day, thd_data_true, label=f'True')
                axs[p_idx].plot(time_of_day, thd_data_est, label=f'Est.')

                axs[p_idx].set_ylabel('$THD_{V} [\\%]$')
                axs[p_idx].legend()
                axs[p_idx].set_title(f'Node {bus} L{p_idx+1}')

            # Formatting x-axis
            if time_span == 'day':
                axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
            elif time_span == 'week':
                axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
            elif time_span == 'year':
                axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.xticks(rotation=45)
            plt.xlabel('Time of Day')
            plt.tight_layout()
        else:
            # plot all data without time component
            fig, axs = plt.subplots(num_phases, 1, figsize=(10, 8), sharex=True)

            bus_idx = bus if mask is None else plot_buses.index(bus)
            colors = ['b', 'r', 'violet']
            for p_idx in range(0, num_phases):
                axs[p_idx].plot(thd_true[:, bus_idx, p_idx], label=f'L{p_idx+1} - True', color=colors[p_idx])
                axs[p_idx].plot(thd_est[:, bus, p_idx], label=f'L{p_idx+1} - Est.', color=colors[p_idx], linestyle='dashed')
                axs[p_idx].grid()
                axs[p_idx].set_ylabel('$THD_{V} [\\%]$')
                axs[p_idx].set_title(f'Node {bus} L{p_idx+1}')
                axs[p_idx].legend(fontsize="8", loc ="upper left")
            plt.xticks(rotation=45)
            plt.xlabel('Data Point')
            plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path,f'THD_{f_name}_Bus{bus}.png'), dpi=300)

        if show_plot:
            plt.show()

        plt.close()


def alpha_plot_loss(directory):
    # directory = 'C:\\Users\\kraem_k\\Documents\\Git\\power-quality-state-estimation\\data\\saved_data\\'

    files_with_loss = []

    for filename in os.listdir(directory):
        if 'loss' in filename:
            files_with_loss.append(directory + filename)

    plt.figure(figsize=(10, 6))

    for file_path in files_with_loss:
        with open(file_path, 'rb') as file:  # 'rb' für das Lesen von Binärdateien
            data = pickle.load(file)
            loss = data['DNN_flex_freq']['loss']
            data_name = os.path.basename(file_path).replace('.pkl', '')  # Dateinamen als Legendenbeschriftung
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, label=f'{data_name}')

    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_all_activations(activations, model, cmap='viridis', dpi=150):
    num_layers = len(activations)
    cols = 4
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5),
                             dpi=dpi)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, ax in enumerate(axes.flatten()):
        if i < num_layers:
            layer_activations = activations[i]
            layer_name = model.layers[i].name
            layer_type = type(model.layers[i]).__name__
            img = None

            if len(layer_activations.shape) == 2:
                num_neurons = layer_activations.shape[-1]
                img = ax.imshow(layer_activations[0].reshape(1, -1), aspect='auto', cmap=cmap)
                ax.set_title(f'Layer {i}: {layer_name} ({layer_type}, {num_neurons} Neurons)')
            elif len(layer_activations.shape) == 4:
                feature_map = layer_activations[0, :, :, 0]
                img = ax.imshow(feature_map, aspect='auto', cmap=cmap)
                ax.set_title(f'Layer {i}: {layer_name} ({layer_type}, Feature-Map 1)')

            if img:
                fig.colorbar(img, ax=ax, orientation='vertical')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def box_plots(title, data: pd.DataFrame, yaxis_scale=None, measure='MAE [V]', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ticklabel_format(axis='y', style='sci')
    bplot = ax.boxplot(data, patch_artist=True, notch=True, labels=data.columns.values,
                       flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'fuchsia'},
                       whiskerprops={'linestyle': '-', 'linewidth': 2.0},)

    repeat_colors = ['r', 'g', 'b', 'c', 'y']
    c_count = 0
    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor(repeat_colors[c_count])
        c_count += 1
        if c_count == len(bplot['boxes']):
            c_count = 0
    plt.yscale('log')
    ax.set_yticks(yaxis_scale)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))  # Stellt sicher, dass die Hauptticks Zehnerpotenzen sind
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.title(title)
    plt.ylabel(measure)
    plt.xlabel("Metric")

    plt.grid(True, which='both', ls='--', lw=0.5)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{title}.png'))
        plt.close()
    else:
        # Plot anzeigen
        plt.show()

    # calculate data explicitly
    results = {}

    for i in range(0, len(data.columns)):
        # Extrahiere Spaltengruppen
        group_cols = data.columns[i]
        group_data = data[group_cols]

        # Berechne die Statistiken
        medians = group_data.median()
        first_quartiles = group_data.quantile(0.25)
        third_quartiles = group_data.quantile(0.75)

        # Ergebnisse speichern
        results[f'Group {group_cols}'] = {
            'Medians': medians,
            '1st Quartiles': first_quartiles,
            '3rd Quartiles': third_quartiles
        }
    return results

