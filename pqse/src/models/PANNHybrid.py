import numpy as np
import tensorflow.keras.layers as kl

from layers.AdmittanceWeightedLayer import AdmittanceWeightedLayer
from models.DNN import DNN


class PANNHybrid(DNN):
    """
    Physics Aware Deep Neural Network (Keras)
    Hybrid model with PANN layers on multidimensional Input and Dense Layers on flattened input
    """

    def __init__(self, hyperparams, network_data, callbacks, custom_layer=AdmittanceWeightedLayer):
        super().__init__(hyperparams, network_data, callbacks)
        self.custom_layer = custom_layer

    def create_layers(self):

        inp = kl.Input(shape=self.input_dims)
        if self.gaussian_noise > 0:
            x = kl.GaussianNoise(self.gaussian_noise)(inp)
        else:
            x = inp
        if self.batch_normalization:
            x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                  mask_indices=self.mask_indices,
                                  layer_scaling_factor=self.layer_scaling_factor)(x)
            x = kl.BatchNormalization(axis=self.feature_axes)(x)
            x = kl.Activation(self.activation)(x)
        else:
            x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                  mask_indices=self.mask_indices, activation="leaky_relu",
                                  layer_scaling_factor=self.layer_scaling_factor)(x)
        skip_x = x
        for i in range(0, self.amount_of_hidden_layers - 1):
            if self.batch_normalization:
                x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                      mask_indices=None,
                                      layer_scaling_factor=self.layer_scaling_factor)(x)
                x = kl.BatchNormalization(axis=self.feature_axes)(x)
                x = kl.Activation(self.activation)(x)
            else:
                x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                      mask_indices=None, activation="leaky_relu",
                                      layer_scaling_factor=self.layer_scaling_factor)(x)
            if self.skip_connections and i % 2 == 1:
                x = kl.Add()([x, skip_x])
                skip_x = x
        x = kl.Flatten(data_format=self.data_format)(x)
        for i in range(0, 2):  # add two dense layers, possibly with dropout
            x = kl.Dense(np.prod(self.output_dims),
                         activation=self.activation)(x)
            if self.dropout > 0:
                x = kl.Dropout(self.dropout)(x)
        out = kl.Reshape(self.output_dims)(x)
        return inp, out
