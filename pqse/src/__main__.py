import configparser
import os
import pickle

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as kl

from ml_multi_dim_toolbox import DimMinMaxScaler
from ml_multi_dim_toolbox.split_complex import SplitComplexConverter, SplitComplexMode, polar_to_cartesian
from models import PANN, PANNHybrid, DNN
from preprocessing import read_data
import mlflow
import mlflow.tensorflow

# mlflow.tensorflow.autolog()
# mlflow.set_tracking_uri("http://139.6.94.20:5005")
config = configparser.ConfigParser()
config.sections()
config_path = os.path.dirname(os.path.dirname(__file__))
config.read(os.path.join(config_path, 'config.ini'))
data_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['data_path'])))
postgresql_config = {
    "dbname": config['POSTGRESQL']['dbname'],
    "user": config['POSTGRESQL']['user'],
    "password": config['POSTGRESQL']['password'],
    "host": config['POSTGRESQL']['host'],
    "port": config['POSTGRESQL']['port']
}


def load_loss_history(path):
    with open(os.path.join(path, 'trainHistoryDict'), 'rb') as file_pi:
        return pickle.load(file_pi)


def graph_nn(network_data):
    from graphs_on_grids.structure import StaticGraphDataset
    from graphs_on_grids.layers import GraphAttention
    from graphs_on_grids.preprocessing import create_train_test_split, mask_features, create_validation_set, \
        apply_scaler
    x_train = network_data['y_train']
    x_transposed = x_train.transpose((0, 3, 1, 2))
    x_reshaped = x_transposed.reshape((x_transposed.shape[0], x_transposed.shape[1], -1))
    x_stacked = np.vstack(x_reshaped)
    df = pd.DataFrame(x_stacked)

    # Create edge list from admittance matrix
    y_mat = network_data['Y_matrices_per_frequency'][50]
    adj_mask = y_mat[0] >= 1e-8
    edges = []
    for i in range(len(adj_mask)):
        for j in range(len(adj_mask)):
            if adj_mask[i, j]:
                # Adjust to file naming convention
                edges.append((i, j))

    # create edge features from admittance matrix
    edge_features = np.zeros((len(edges), 20, 2))
    y_mats = network_data['Y_matrices_per_frequency']
    # convert dict to list take only real and imaginary
    y_mats_cart = np.array(list(y_mats.values()))[:, :2, :, :]
    # convert matrix to dataframe of edge features
    for i, edge in enumerate(edges):
        edge_features[i, ...] = y_mats_cart[:, :, edge[0], edge[1]]  # Real part
    edge_features_reshaped = edge_features.reshape((edge_features.shape[0], -1))
    edge_features_expanded = np.repeat(edge_features_reshaped[np.newaxis, ...], len(x_reshaped), axis=0)
    edge_features_stacked = np.vstack(edge_features_expanded)
    df_edge = pd.DataFrame(edge_features_stacked)
    n_edges = len(edges)
    n_nodes = len(y_mat[0])
    graph = StaticGraphDataset.pandas_to_graphs(df, n_nodes, list(df.columns),
                                                df_edge, n_edges, list(df_edge.columns))
    dataset = StaticGraphDataset(edges, graph)
    create_train_test_split(dataset, 0.8)
    _, _ = apply_scaler(dataset, method="min_max", target="node")
    train, test = apply_scaler(dataset, method="min_max", target="edge")
    mask_indices = [5, 28, 34]
    masked_nodes = np.setdiff1d(np.arange(0, 44), mask_indices)
    masked_train, masked_test = mask_features(train, test, list(df.columns), masked_nodes)
    x_train, x_val, y_train, y_val = create_validation_set(masked_train, train)
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    # actually remove indices
    x_train[0] = np.moveaxis(np.array([x_train[0][:, i, :] for i in mask_indices]), 0, 1)
    x_val[0] = np.moveaxis(np.array([x_val[0][:, i, :] for i in mask_indices]), 0, 1)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    tf.keras.backend.clear_session()
    node_input = kl.Input(shape=(len(mask_indices), len(df.columns)))
    node_flat = kl.Flatten()(node_input)
    node_dense = kl.Dense(n_nodes * len(df.columns))(node_flat)
    node_reshape = kl.Reshape((n_nodes, len(df.columns)))(node_dense)
    # Edge features input
    edge_input = kl.Input(shape=(n_edges, len(df_edge.columns)))
    embedding_size = 25
    hidden_unit_nodes = [96]
    hidden_units_attention = [10]
    n_layers = 6
    skip_connections = True
    x = GraphAttention(dataset.adjacency_matrix, embedding_size, hidden_unit_nodes, hidden_units_attention)(
        [node_reshape, edge_input])
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    for i in range(0, n_layers):
        if i % 2 == 0 and skip_connections:
            skip_x = x
        x = GraphAttention(dataset.adjacency_matrix, embedding_size, hidden_unit_nodes, hidden_units_attention)(x)
        x = kl.BatchNormalization()(x)
        x = kl.ReLU()(x)
        if i % 2 == 1 and skip_connections:
            x = kl.Add()([x, skip_x])
    out = kl.Dense(40)(x)
    model = keras.Model(inputs=[node_input, edge_input], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001)
                  , loss=keras.losses.MeanSquaredError())
    model.summary()
    model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), batch_size=2120)

    pass


def load_gradients_plot(path0, path1):
    with open(path0, 'rb') as file_pi:
        gradients0 = pickle.load(file_pi)
    with open(path1, 'rb') as file_pi:
        gradients1 = pickle.load(file_pi)
    from evaluation.plot import plot_gradients_model_comparison
    plot_gradients_model_comparison(gradients0, gradients1, None)


def transpose_to_channels_last(data, predictions, keys=None):
    from copy import deepcopy
    data = deepcopy(data)
    predictions = deepcopy(predictions)
    if keys is None:
        keys = ['x_test', 'y_test', 'x_val', 'y_val', 'x_train_n', 'x_test_n', 'x_val_n', 'y_train_n', 'y_test_n',
                'y_val_n']
    # transpose array to always follow channels_last for evaluation
    for key, value in predictions.items():
        for key1, value1 in value.items():
            predictions[key][key1] = np.transpose(value1, (0, 3, 1, 2))
    for key in keys:
        data[key] = np.transpose(data[key], (0, 3, 1, 2))
    return data, predictions


if __name__ == '__main__':
    pass
    ### from Postgres
    ### Custom loss
    # from custom_loss import kirchhoff_loss, KirchhoffLoss, CombinedLoss

    # res = kirchhoff_loss(data['y_train_n'], data['Y_matrices_per_frequency'], data['scaler'])
    # mse_loss = tf.keras.losses.MeanSquaredError()
    # custom_loss = KirchhoffLoss(data['Y_matrices_per_frequency'], data['scaler'])
    # combined_loss = CombinedLoss(mse_loss, custom_loss, 0.5)
    # hyperparams['loss_function'] = combined_loss

    # Map custom layers and custom losses to their respective classes
    # custom_objects = {'KirchhoffLoss': KirchhoffLoss, 'CombinedLoss': CombinedLoss,
    #                   'AdjacencyPrunedLayer': AdjacencyPrunedLayer,
    #                   'AdmittanceLayer': AdmittanceLayer, 'AdmittanceWeightedLayer': AdmittanceWeightedLayer,
    #                   'leaky_relu': tf.nn.leaky_relu}
    # reconstructed_model = tf.keras.models.load_model('my_model.keras', custom_objects=custom_objects)
    # reconstructed_model.summary()

    # graph_nn(data)

    # from error_identification import identify_phase_angle_problem, plot_heatmap_phase_angle_nodes
    # results = identify_phase_angle_problem(data['y_train'])
    # plot_heatmap_phase_angle_nodes(results)
    # data_evaluation(data, n_samples=1000000)
