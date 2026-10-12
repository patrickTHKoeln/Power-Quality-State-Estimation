import numpy as np
from tensorflow.keras.losses import Loss
import tensorflow as tf
from ml_multi_dim_toolbox.scaling import Scaler
from ml_multi_dim_toolbox.split_complex import cartesian_to_polar


# TODO implement with tensors

@tf.keras.saving.register_keras_serializable(package="Custom", name=None)
class CombinedLoss(Loss):
    def __init__(self, first_loss, second_loss, alpha=0.5):
        """
        :param first_loss:
        :param second_loss:
        :param alpha: ratio of the first loss (0.3 means 30% of the first loss and 70% of the second loss)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.first_loss = first_loss
        self.second_loss = second_loss

    def call(self, y_true, y_pred):
        """
        :param y_true: true values
        :param y_pred: predicted values
        :return:
        """
        # TODO: read this: https://stackoverflow.com/questions/63390725/should-the-custom-loss-function-in-keras-return-a-single-loss-value-for-the-batc
        return self.alpha * self.first_loss(y_true, y_pred) + (1 - self.alpha) * self.second_loss(y_true, y_pred)


@tf.keras.saving.register_keras_serializable(package="Custom", name=None)
class KirchhoffLoss(Loss):
    def __init__(self, admittance_matrix: np.ndarray, scaler: Scaler, threshold: float = 1e-20):
        super(KirchhoffLoss, self).__init__()
        self.admittance_matrix = admittance_matrix
        self.scaler = scaler
        self.threshold = threshold

    def call(self, y_true, y_pred):
        """
        :param y_true: will be ignored as the physical loss is calculated from the predicted values
        :param y_pred:
        :return:
        """

        return kirchhoff_loss(y_pred, self.admittance_matrix, self.scaler)

    def get_config(self):
        """
        For saving the model (serialization)
        :return: dictionary with the parameters
        """
        return {
            'admittance_matrix': self.admittance_matrix,
            'scaler': self.scaler,
            'threshold': self.threshold
        }


def kirchhoff_loss(y_pred_n, admittance_matrix, scaler: Scaler):
    """
    Kirchhoff's law loss
    :param y_pred_n: scaled, predicted values (normalized), assuming on axis 3
    voltages and currents lie with their respective real and imaginary part
    shape: (batch, nodes, freq, 4)
    :param admittance_matrix: admittance matrix in cartesian form with shape (nodes, nodes, freq, 2)
    :param scaler: scaler used for data transformation
    :return: residuals per node
    """
    # 1. revert scaling
    y_pred = scaler.revert(y_pred_n)
    # 2.cartesian to polar on predicted values
    y_pred_polar = cartesian_to_polar(y_pred, axis=3, real_index=(0, 2), imag_index=(1, 3))
    # 3. get real and imaginary part of admittance matrix
    admittance_matrix_polar = cartesian_to_polar(admittance_matrix, axis=3, real_index=0, imag_index=1)
    y_mag = admittance_matrix_polar[..., 0]
    y_ang = admittance_matrix_polar[..., 1]
    # 4. extract voltages and currents
    v_mag = y_pred_polar[..., 0]
    v_ang = y_pred_polar[..., 1]
    i_mag = y_pred_polar[..., 2]
    i_ang = y_pred_polar[..., 3]
    res = np.zeros((v_mag.shape[1], v_mag.shape[2]))  # residuals per node and frequency
    # 5. calculate power
    for t in range(v_mag.shape[0]):  # iterate over batch / time steps
        p_phys, q_phys = pbe(v_mag[t], v_ang[t], y_mag, y_ang)
        p = v_mag[t] * i_mag[t] * np.cos(v_ang[t] - i_ang[t])
        q = v_mag[t] * i_mag[t] * np.sin(v_ang[t] - i_ang[t])
        # 6. calculate residuals
        # TODO: both p and p_phys are multiplied by v_mag, therefore this could be omitted to save computation time
        # same for q and q_phys
        # Small inaccuracies in admittance result in large errors in p_phys and q_phys
        res += np.abs(-p - p_phys) + np.abs(-q - q_phys)
    return res


def pbe(v_mags, v_ang, y_mags, y_ang):
    """
    Power balance equation
    Node dimensions first
    :param v_mags: voltage magnitudes, shape: (n) / (n, freq)
    :param v_ang: voltage angles (radians)
    :param y_mags: admittance magnitudes, shape: (n, n) / (n, n, freq)
    :param y_ang: admittance angles (radians)
    :return: P, Q
    """
    n = len(v_mags)
    if len(y_mags.shape) > 2:  # calculate for each frequency
        freqs = y_mags.shape[-1]
        p = np.zeros((n, freqs))
        q = np.zeros((n, freqs))
    else:
        p = np.zeros(n)
        q = np.zeros(n)
    for i in range(n):
        for j in range(n):
            p[i] += v_mags[i] * v_mags[j] * y_mags[i][j] * np.cos(v_ang[i] - v_ang[j] - y_ang[i][j])
            q[i] += v_mags[i] * v_mags[j] * y_mags[i][j] * np.sin(v_ang[i] - v_ang[j] - y_ang[i][j])
    return p, q


def create_adjacency_matrix(admittance_matrix: np.ndarray, threshold: float = 1e-20):
    """
    Creates an adjacency matrix from the admittance matrix
    :param admittance_matrix:
    :param threshold: cutoff ratio for small values. Values less than threshold will be set to 0
    :return:
    """
    # Create a boolean matrix where True indicates the values are greater than or equal to the threshold
    bool_matrix = np.abs(admittance_matrix) >= threshold

    # Convert the boolean matrix to a matrix of integers (1 where True, 0 where False)
    adjacency_matrix = bool_matrix.astype(int)

    return adjacency_matrix
