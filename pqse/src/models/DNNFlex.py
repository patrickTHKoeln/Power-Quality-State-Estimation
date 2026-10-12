import numpy as np
import tensorflow.keras.layers as kl
import time
from models.DNN import DNN


class DNNFlex(DNN):
    def __init__(self, hyperparams, network_data, callbacks=None):
        super().__init__(hyperparams, network_data, callbacks)
        self.y_labels_test = network_data['labels_test']

    def predict(self, model, mask_y_test=None, batch_size=None):
        """
        Evaluate model performance on given test set

        :param model: model
        :param mask_y_test: mask of test set target values
        :param batch_size: [optional]
        :return:
        """
        start = time.process_time()
        y_pred_n = model.predict(self.x_test_n, batch_size)
        print(f"CPU Time for prediction of test set:{time.process_time() - start}")

        # revert normalization
        y_pred = self.scaler.revert(y_pred_n, self.y_labels_test )
        return y_pred, y_pred_n