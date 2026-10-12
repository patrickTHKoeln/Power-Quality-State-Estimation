import numpy as np
import tensorflow as tf
from layers.AdmittanceLayer import AdmittanceLayer


@tf.keras.saving.register_keras_serializable(package="Custom", name=None)
class AdjacencyPrunedLayer(AdmittanceLayer):
    """
    Prune connections based on adjacency information.
    For this, the absolute value of the admittance matrix is taken. Values below threshold are set to zero else to one
    """

    def __init__(self, y_mats, mask_indices=None, activation=None, threshold=1e-8, data_format='channels_last',
                 **kwargs):
        super(AdjacencyPrunedLayer, self).__init__(y_mats, mask_indices, activation, data_format=data_format, **kwargs)
        self.threshold = threshold

    def get_config(self):
        """
        Required for serialization purposes
        :return: config
        """
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Required for deserialization purposes
        :param config: config
        :return: AdjacencyPrunedLayer
        """
        return cls(**config)

    def build(self, input_shape):
        # set values below threshold to zero else to one
        self.y_mats = np.where(self.y_mats < self.threshold, 0, 1)
        super().build(input_shape)
