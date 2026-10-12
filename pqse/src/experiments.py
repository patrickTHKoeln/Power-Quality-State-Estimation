import configparser
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from OpenDSSWrapper.converter.grids import CigreLVDistPV, IEEE33TestFeeder, MicrogridTHKoeln
from evaluation.plot import evaluate, plot_heatmaps, plot_losses, plot_cdf, plot_thd_bus
from evaluation.error import error_metrics_per_harmonic
from layers import AdjacencyPrunedLayer
from ml_multi_dim_toolbox import DimMinMaxScaler, SplitComplexMode, SplitComplexConverter, mean_absolute_error, \
    mean_squared_error, normalized_root_mean_squared_error, FFTMinMaxScaler, Scaler
from models import PANNHybrid, DNN, PANN, CNN, Transformer, DNNFlex
from models.callbacks import CyclicLR
from preprocessing import read_data, add_measurement_noise, read_postgresql_db, read_microgrid_data, EstimationModes

# Set random seed for reproducibility
# full reproducibility also requires setting shuffle and use_multiprocessing to False in model fit
np.random.seed(1)
tf.random.set_seed(2)
random.seed(3)

config = configparser.ConfigParser()
config.sections()
config_path = os.path.dirname(os.path.dirname(__file__))
config.read(os.path.join(config_path, 'config.ini'))
data_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['data_path'])))

# Alter the following parameters
EPOCHS = 1000  # Epochs to train
USE_TRAINED_MODEL = True  # use existing weights of trained model?
# If EPOCHS is set to a number > 0 and USE_TRAINED_MODEL is set, the model will continue training on the loaded weights

# GRID = "ieee33"
GRID = "cigrelv"
# GRID = "thkoelnnetwork"
ANALYZE_DATA = False
NOISY_MODELS = False
BASE_MODELS = True
TRUE_DATA = True
SQL_DATA = False
EST_MODE = EstimationModes.TSE


def state_estimation(hyperparams: dict, network_data: dict, model_class, callbacks: list, exp_path: str, label: str,
                     custom_layer=None, load_weights: bool = False):
    """
    Create and train a model for state estimation, and return the loss and predictions
    :param hyperparams:
    :param network_data:
    :param model_class:
    :param callbacks:
    :param exp_path:
    :param label:
    :param custom_layer:
    :param load_weights:
    :return:
    """
    loss = {}
    predictions = {}
    if model_class == PANN or model_class == PANNHybrid:
        m = model_class(hyperparams, network_data, callbacks, custom_layer=custom_layer)
    else:
        m = model_class(hyperparams, network_data, callbacks)
    model = m.create_model()
    model, loss[label] = m.train(model, exp_path, load_weights=load_weights)
    y_pred, y_pred_n = m.predict(model)
    pred = {'y_pred': y_pred, 'y_pred_n': y_pred_n}
    predictions[label] = pred
    return loss, predictions


def exp_dnn_config():
    label = "DNN"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 16384,  # 16384,
        'num_hidden_layers': 2,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,  # set scaling factor to 2 to achieve the same amount of neurons per layer as in PANN
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
        # 'kernel_regularizer': tf.keras.regularizers.L2(1e-5)
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200., mode='triangular')]
    model_class = DNN
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_dnn_gauss_config():
    label = "DNN Gauss 0.01"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 2,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.01,  # standard deviation
        'batch_normalization': False,
        # 'kernel_regularizer': tf.keras.regularizers.L2(1e-4)
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200., mode='triangular')]
    model_class = DNN
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_pann_config():
    label = "PANN"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 6 if GRID == 'cigrelv' else 5 if GRID == "ieee33" else 4,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
        # 'kernel_regularizer': tf.keras.regularizers.L2(1e-4)
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200., mode='triangular')]
    model_class = PANN
    custom_layer = AdjacencyPrunedLayer
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_pann_gauss_config():
    label = "PANN Gauss 0.01"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 6 if GRID == 'cigrelv' else 5 if GRID == "ieee33" else 4,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.01,  # standard deviation
        'batch_normalization': False,
        # 'kernel_regularizer': tf.keras.regularizers.L2(1e-4)
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200., mode='triangular')]
    model_class = PANN
    custom_layer = AdjacencyPrunedLayer
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_transformer_config():
    label = "Transformer"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': int(16384 / 16),  # batch size limited by GPU memory size
        'num_hidden_layers': 12,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 1e-5,
        'layer_scaling_factor': 1,
        'epochs': int(EPOCHS / 8), # more gradient updates due to smaller batch size --> reduce amount of epochs
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
        'ff_dim': 256,
        'key_dim': 80,
        'input_len': 44,
        'num_heads': 8,
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 100, max_lr=hyperparams['learning_rate'],
                          step_size=200.)]
    model_class = Transformer
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_cnn_config():
    label = "CNN"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 3,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,  # set scaling factor to 2 to achieve the same amount of neurons per layer as in PANN
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = CNN
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_dnnflex_config():
    label = "DNNFlex"
    exp_path = os.path.join(data_path, GRID, label)
    hyperparams = {
        'batch_size': 2400,
        'num_hidden_layers': 8,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'layer_scaling_factor': 1,  # set scaling factor to 2 to achieve the same amount of neurons per layer as in PANN
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [ModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                 initial_value_threshold=1e-2),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = DNNFlex
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def load_data_cigrelv(estimation_mode):
    # Load data
    from OpenDSSWrapper.tools.config import postgres_config
    modes = [SplitComplexMode.CARTESIAN] if estimation_mode == EstimationModes.HSE else [SplitComplexMode.POLAR]
    scaler = DimMinMaxScaler() if estimation_mode == EstimationModes.HSE else FFTMinMaxScaler(data_path)
    transpose_data_format = EstimationModes.HSE == estimation_mode
    network_data = {
        'grid_name': "CigreLVDist",
        'base_table_name': "CigreLVDist",
        'scaler': scaler,
        'test_size': 0.1,
        'validation_size': 0.2,
        'data_format': 'channels_last',
        'mask_x_test': [6, 22, 28],
        'mask_y_test': None, # [0, 19, 22, 35],  # mask of target values
        'split_complex_modes': modes,
        'estimation_mode': estimation_mode
    }
    grid = CigreLVDistPV()
    # from files
    if not SQL_DATA:
        input_target_axis = 2# 3 if estimation_mode == EstimationModes.TSE else 2
        input_converter = SplitComplexConverter(modes, target_axis=input_target_axis)
        target_converter = SplitComplexConverter(modes[0], target_axis=3)
        data = read_data(data_path, network_data, grid, input_converter, target_converter, transpose_data_format=transpose_data_format)
    else:
        frequencies = list(np.arange(50, 1050, 50))
        data = read_postgresql_db(postgres_config, network_data, grid,
                                  columns=['v1', 'vangle1', 'v2', 'vangle2', 'v3', 'vangle3'],
                                  separate_datasets=True, unify_voltage_level=True,
                                  voltage_level_kv=20, frequencies=frequencies, fix_unmeasurable_angles=True,
                                  use_lab_data_test=TRUE_DATA, lab_data_path=data_path)

    # # save data as pickle
    with open(os.path.join(data_path, 'data.pic'), 'wb') as pic:
        pickle.dump(data, pic)
    # load data from pickle
    with open(os.path.join(data_path, 'data.pic'), 'rb') as pic:
        data = pickle.load(pic)
    return data


def load_data_ieee33():
    # Load data
    modes = [SplitComplexMode.CARTESIAN, SplitComplexMode.EXPONENTIAL]
    network_data = {
        'grid_name': "IEEE33",
        'scaler': DimMinMaxScaler(),
        'test_size': 0.1,
        'validation_size': 0.2,
        'data_format': 'channels_last',
        'mask_x_test': [2, 12, 28],
        'mask_y_test': None,  # mask of target values
        'split_complex_modes': modes
    }
    grid = IEEE33TestFeeder()
    from OpenDSSWrapper.tools.config import postgres_config
    # data = read_postgresql_db(postgres_config, network_data, grid, separate_datasets=True,
    #                           reference_rotate=None, columns=['v1', 'vangle1'], unify_voltage_level=False,
    #                           voltage_level_kv=10., frequencies=list(np.arange(60, 1201, 60)))
    # with open(os.path.join(data_path, 'ieee33_data.pic'), 'wb') as f:
    #     pickle.dump(data, f)
    with open(os.path.join(data_path, 'ieee33_data.pic'), 'rb') as f:
        data = pickle.load(f)
    return data

def load_data_microgrid_thk():

    from OpenDSSWrapper.tools.config import postgres_config
    modes = [SplitComplexMode.CARTESIAN]

    grid = MicrogridTHKoeln(None, None, True, False)
    network_data = {
        'grid_name': "thkoelnnetwork",
        'base_table_name': "THKoelnNetwork",
        'scaler': DimMinMaxScaler(),
        'test_size': 0.1,
        'validation_size': 0.2,
        'data_format': 'channels_last',
        'mask_x_test': [0, 7], # [0, 3, 6, 7],  # mask of target values
        'mask_y_test': [0, 3, 6, 7],
        #'mask_y_test': None,
        'split_complex_modes': modes,
        'estimation_mode': EstimationModes.HSE
    }
    frequencies = list(np.arange(50, 1300, 50))

    data = read_postgresql_db(postgres_config, network_data, grid,
                              columns=['v1', 'vangle1', 'v2', 'vangle2', 'v3', 'vangle3'],
                              separate_datasets=True, unify_voltage_level=False,
                              voltage_level_kv=.4, frequencies=frequencies,
                              use_lab_data_test=TRUE_DATA, lab_data_path=data_path)

    # # save data as pickle
    with open(os.path.join(data_path, 'data.pic'), 'wb') as pic:
        pickle.dump(data, pic)
    # load data from pickle
    with open(os.path.join(data_path, 'data.pic'), 'rb') as pic:
        data = pickle.load(pic)
    return data


def compare_experiments(configs, data, measurement_noise=0.01, save_path=None):
    """
    Compare multiple experiments to each other
    :param configs: experiment configurations
    :param data: network data
    :param measurement_noise: add measurement noise to the data (if None, no measurement noise is added)
    :param save_path: path to save the plots
    :return: predictions and losses
    """
    predictions = {}
    losses = {}
    if measurement_noise is not None:
        from copy import deepcopy
        data_updated = deepcopy(data)
        data_updated['x_test_n'] = add_measurement_noise(data_updated['x_test_n'], 0.01, feature_axis=2, scale_on_max=True)
        data_updated['x_test'] = data_updated['input_scaler'].revert(
            data_updated['x_test_n'])  # apply same noise to non-scaled data as well
        load_weights = False  # no training necessary, only prediction
    else:
        data_updated = data
        load_weights = USE_TRAINED_MODEL
    for cfg in configs:
        hyperparams, callbacks, model_class, custom_layer, exp_path, label = cfg
        if measurement_noise:
            hyperparams['epochs'] = 0  # no training necessary, only prediction
        loss, preds = state_estimation(hyperparams, data_updated, model_class, callbacks, exp_path, label,
                                       custom_layer=custom_layer, load_weights=load_weights)
        losses.update(loss)
        predictions.update(preds)
    input_noise = True if measurement_noise else False
    if data['estimation_mode'] == EstimationModes.HSE :
        evaluate(predictions, data_updated, input_noise=input_noise, save_path=save_path, show_plot=False)
    # Save predictions
    with open(os.path.join(data_path, 'pred_y.pic'), 'wb') as pic:
        pickle.dump(predictions, pic)
    # plot residual density
    mae = {}
    from copy import deepcopy
    predictions_masked = deepcopy(predictions)
    for key, value in predictions.items():
        y_pred_n_masked = apply_mask(value['y_pred_n'], data['mask_y_test'])
        mse_overall = mean_squared_error(data['y_test_n'], y_pred_n_masked, axis=None)
        mae_overall = mean_absolute_error(data['y_test_n'], y_pred_n_masked, axis=None)
        print(f"Model {key} - MSE on test set: {mse_overall} - MAE on test set: {mae_overall}")
        nrmse = normalized_root_mean_squared_error(data['y_test_n'], y_pred_n_masked, axis=(0,1,3))
        for h in range(len(nrmse)):
            print(f"Model {key} - NRMSE Harmonic {h+1}: {np.round(nrmse[h], 4)}")
        mse = mean_squared_error(data['y_test_n'], y_pred_n_masked, axis=(0, 1, 3))
        for h in range(len(mse)):
            print(f"Model {key} - MSE Harmonic {h + 1}: {np.round(mse[h], 4)}")
        mae[key] = mean_absolute_error(data['y_test_n'], y_pred_n_masked, axis=0).flatten()
        predictions_masked[key]['y_pred_n'] = y_pred_n_masked
        y_pred_masked = apply_mask(value['y_pred'], data['mask_y_test'])
        predictions_masked[key]['y_pred'] = y_pred_masked
    plot_cdf(mae, label='MAE', color_str='DNN', save_path=save_path, show_plot=False)
    return predictions, losses, predictions_masked


def apply_mask(array, mask):
    mask_slice = mask if mask is not None else slice(None)
    mask_y_test = (slice(None), mask_slice, slice(None), slice(None)) # for now always on the same axis
    indices = [np.array(m) if not isinstance(m, slice) else m for m in mask_y_test]
    return array[tuple(indices)]


def pca_analysis(data):
    import sklearn
    from sklearn.decomposition import PCA
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    aeberle_mask = [0,3,7,6]
    umd_mask = [9,10,11,12,13,14]
    aeberle_mask_test = [0,3,4,5]
    umd_mask_test = [6,7,8,9,10,11]
    mask = data['mask_x_test_meas']
    y_train = data['y_train'][:,mask,:,...][...,0]
    y_test = data['y_test'][:,:,:,...][...,0]
    y_train = y_train.reshape(len(y_train),-1)
    y_test = y_test.reshape(len(y_test),-1)
    pca = PCA(n_components=3)
    y_train_pca = pca.fit_transform(y_train)
    y_test_pca = pca.transform(y_test)
    # plot results
    ex_variance = np.var(y_train_pca, axis=0)
    print(ex_variance)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)
    print(ex_variance_ratio)
    n_samples = 1000
    y_total = np.concatenate((y_train_pca[:n_samples], y_test_pca[:n_samples]))
    Xax = y_total[:, 0]
    Yax = y_total[:, 1]
    Zax = y_total[:, 2]

    cdict = {0: 'red', 1: 'green'}
    labl = {0: 'y_train', 1: 'y_test'}
    marker = {0: '*', 1: 'o'}
    alpha = {0: .2, 1: .5}

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')

    labels = [0]*len(y_train_pca[:n_samples]) + [1]*len(y_test[:n_samples])

    for ix, l in enumerate(labels):
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
                   marker=marker[l], alpha=alpha[l])
    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=14)
    ax.set_ylabel("Second Principal Component", fontsize=14)
    ax.set_zlabel("Third Principal Component", fontsize=14)

    ax.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def analyze_training_data(data, set='y_train', save_path=None, axes_descr=None):
    """
    For each frequency and complex component, compute and plot the correlation matrix (across nodes)
    and perform PCA on the training samples. The training data is assumed to have shape
    (batch, nodes, frequencies, complex component). If a 'mask_y_test' key exists in data,
    then only the specified node indices are used.
    :param data: Dictionary containing sets
    """
    if axes_descr is None:
        axes_descr = {'batch':0, 'nodes':1, 'frequencies':2, 'complex_component':3}
    target_set = data[set]
    if set == 'y_train' or set == 'y_val':
        labels = np.arange(target_set.shape[axes_descr.get('nodes')])
    elif set == 'y_test':
        labels = data['mask_y_test']
    elif set == 'x_train' or set == 'x_test' or set == 'x_val':
        labels = data['mask_x_test']
    else:
        raise UserWarning('Chose valid set name')
    # Loop over all frequencies and complex components
    for f in range(target_set.shape[axes_descr.get('frequencies')]):
        for c in range(target_set.shape[-1]):
            # Extract training data for frequency f and component c (selecting common nodes)
            target_data = target_set[:, :, f, c]  # shape: (n_train, len(nodes_idx))

            # --- Correlation Matrix ---
            # Compute correlation matrix across nodes (features) using samples as observations.
            corr_matrix = np.corrcoef(target_data, rowvar=False)

            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", yticklabels=labels, xticklabels=labels)
            plt.title(f'Correlation Matrix - Harmonic {f+1} - c:{c}')
            plt.xlabel('Node Index')
            plt.ylabel('Node Index')
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'corr_{f+1}_{set}_c{c}.png'))
            plt.show()

            # --- Principal Component Analysis (PCA) ---
            # Apply PCA to the training samples (each sample is a vector in the node space).
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(target_data)

            plt.figure(figsize=(8, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA Scatter Plot - Harmonic {f+1}, Component - c;{c}')
            plt.grid(True)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, f'pca_{f+1}_{set}_c{c}.png'))
            plt.show()

            # Output the explained variance ratio for insight into the dimensionality reduction
            print(
                f'Frequency {f}, Component - c_{c} - Explained Variance Ratio: {pca.explained_variance_ratio_}')


if __name__ == '__main__':
    # Load data from pickle files
    if GRID == 'cigrelv':
        data = load_data_cigrelv(EST_MODE)
        plot_buses = [16, 25, 37]
        base_kvs = dict.fromkeys(np.arange(0, 44, 1), 0.4)  # most in LV
        base_kvs.update(dict.fromkeys([0, 1, 20, 23], 20))  # some in MV
    elif GRID == 'ieee33':
        data = load_data_ieee33()
        plot_buses = [13, 23, 29]  # IEEE33
        base_kvs = dict.fromkeys(np.arange(0, 33, 1), 12.66)  # all in MV
    elif GRID == 'thkoelnnetwork':
        data = load_data_microgrid_thk()
        # data = load_data_microgrid_pca()
        plot_buses = [0, 3, 7]
        base_kvs = dict.fromkeys(np.arange(0, 15, 1), 0.4)  # all in LV
    else:
        raise UserWarning("Choose a valid grid name")
    if ANALYZE_DATA:
        # Plot data distribution
        save_path = os.path.join(data_path, GRID, 'data_dist')
        from OpenDSSWrapper.simulation.plot_data import plot_data_distribution

        for node in data['mask_y_test']:
            plot_data_distribution(data, node, data['frequencies'], mode='y', complex_mode=SplitComplexMode.CARTESIAN,
                                       normalized=False, save_path=save_path, show_plot=False, no_columns=4, polar_plot=True)

        #for node in data['mask_y_test']:
        #    plot_data_distribution(data, node, frequencies, mode='y', complex_mode=SplitComplexMode.POLAR,
        #                               normalized=False, save_path=save_path, show_plot=False)

        # pca_analysis(data)
        save_path = os.path.join(data_path, GRID, 'correlation')
        analyze_training_data(data, 'y_test', save_path=save_path, axes_descr={'batch':0, 'nodes':1, 'frequencies':2, 'phase':3})

    # Run experiments
    if BASE_MODELS:
        ## Base models
        save_path = os.path.join(data_path, GRID, 'plots')
        base_model_configs = [exp_dnnflex_config()]# [exp_dnn_config(), exp_pann_config(), exp_cnn_config()]
        predictions, losses, predictions_masked = compare_experiments(base_model_configs, data, measurement_noise=None, save_path=save_path)
        # plot_heatmaps(predictions_masked, data, complex_axis=3, save_path=save_path, show_plot=False,
        #              magnitude_only=True)
        plot_losses(losses, save_path=save_path, show_plot=False)
        ### with input noise
        save_path = os.path.join(data_path, GRID, 'plots_noise')
        # compare_experiments(base_model_configs, data, measurement_noise=0.01, save_path=save_path)

    if NOISY_MODELS:
        # ## Gaussian noise models
        save_path = os.path.join(data_path, GRID, 'plots_gauss')
        gauss_model_configs = [exp_dnn_gauss_config(), exp_pann_gauss_config()]
        predictions, losses, predictions_masked = compare_experiments(gauss_model_configs, data, measurement_noise=None, save_path=save_path)
        plot_heatmaps(predictions_masked, data, complex_axis=4, save_path=save_path, show_plot=False,
                      magnitude_only=False, mask=data['mask_y_test'])

        plot_losses(losses, save_path=save_path, show_plot=False)
        for m_name in predictions.keys():
            plot_thd_bus(data['y_test'], predictions[m_name]['y_pred'], list(range(2, 25)), plot_buses,
                         save_path=save_path, show_plot=False, f_name=m_name, time_span=None, mask=True)
            # excluded_buses = [0, 1, 20, 23, 21, 22]  # MV busses and those in industrial subnetwork
            error_metrics_per_harmonic(y_true=data['y_test'], y_pred=predictions_masked[m_name]['y_pred'],
                                       base_kvs=base_kvs)
        ### with input noise
        # save_path = os.path.join(data_path, GRID, 'plots_noise_gauss')
        # compare_experiments(gauss_model_configs, data, measurement_noise=0.01, save_path=save_path)
    split_complex_conv = SplitComplexConverter