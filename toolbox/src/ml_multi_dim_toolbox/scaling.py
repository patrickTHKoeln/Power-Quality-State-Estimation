import numpy as np
from ml_multi_dim_toolbox.split_complex import SplitComplexConverter, polar_to_cartesian, cartesian_to_polar
import copy

class Scaler:
    """
    Base class for Scalers along specific axis.
    """

    def __init__(self, eps: float = 1e-8):
        """

        :param eps: Small value to avoid division by zero.
        """
        self.transformation_params = None
        self.eps = eps
        pass

    def fit_transform(self, data: np.ndarray, axis=None) -> np.ndarray:
        """
        Fit to data and transform a multidimensional numpy array over the specified axis.
        :param data: numpy array
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return: transformed array
        """
        pass

    def transform(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Transform data using the transformation parameters and axis used in fit_transform
        :param data: numpy array
        :param meta_data: optional data description
        :return: transformed array
        """
        pass

    def revert(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Reverts the transformation of a numpy array using the transformation parameters and axis from fit_transform.
        :param data: transformed numpy array
        :param meta_data: optional data description
        :return: numpy array with reverted transformation
        """
        pass

    def get_params(self) -> dict:
        return self.transformation_params


class DimMinMaxScaler(Scaler):
    """
    Min-Max-Scaler for multidimensional numpy arrays.
    """

    def __init__(self):
        super().__init__()
        self.axis = None

    def fit_transform(self, data: np.ndarray, axis=None) -> np.ndarray:
        """
        Fit to min max values, then transform a multidimensional numpy array over the specified axis.
        :param np.ndarray data: The input array to be normalized.
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return np.ndarray: the normalized array
        """
        if not axis:
            axis = tuple(range(data.ndim))
        self.axis = axis
        # If axis is a single integer, convert it into a tuple
        if isinstance(axis, int):
            axis = (axis,)

        # Check if axis is a tuple or list of integers
        if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
            raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")
        if not all(0 <= a < data.ndim for a in axis):
            raise ValueError("All axes must be valid for the input array.")
        # Compute the minimum and maximum values along the specified axis
        min_vals = data.min(axis=axis, keepdims=True)
        max_vals = data.max(axis=axis, keepdims=True)

        # Normalize the data
        normalized_data = (data - min_vals) / (max_vals - min_vals + self.eps)

        # Store the normalization parameters in a dictionary
        self.transformation_params = {'min_vals': min_vals, 'max_vals': max_vals}
        return normalized_data

    def transform(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Call fit_transform first!
        transform data using the transformation parameters and axis used in fit_transform
        :param np.ndarray data: The input array to be normalized.
        :param meta_data: optional data description
        :return np.ndarray: The normalized array.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'min_vals' not in self.transformation_params or 'max_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        # Use given min and max values
        min_vals = self.transformation_params['min_vals']
        max_vals = self.transformation_params['max_vals']

        # Check if the shapes of the min and max values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if min_vals.shape != expected_shape or max_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")
        # Normalize the data
        normalized_data = (data - min_vals) / (max_vals - min_vals + self.eps)
        return normalized_data

    def revert(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Reverts the normalization of a numpy array using the normalization parameters and axis from fit_transform.
        :param np.ndarray data: The normalized array to be reverted.
        :param meta_data: optional data description
        :return np.ndarray: The original array after reverting normalization.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'min_vals' not in self.transformation_params or 'max_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        # Use given min and max values
        min_vals = self.transformation_params['min_vals']
        max_vals = self.transformation_params['max_vals']

        # Check if the shapes of the min and max values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if min_vals.shape != expected_shape or max_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")

        # Revert the normalization
        original_arr = data * (max_vals - min_vals) + min_vals

        return original_arr


class DimStandardScaler(Scaler):
    """
    Standard-Scaler for multidimensional numpy arrays.
    Performs Standardization also known as z-score-normalization.
    """

    def __init__(self):
        super().__init__()
        self.axis = None

    def fit_transform(self, data: np.ndarray, axis=None) -> np.ndarray:
        """
        Fit to mean and standard deviation, then transform a multidimensional numpy array over the specified axis.
        :param np.ndarray data: The input array to be normalized.
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return np.ndarray: normalized array
        """
        if not axis:
            axis = tuple(range(data.ndim))
        self.axis = axis
        # If axis is a single integer, convert it into a tuple
        if isinstance(axis, int):
            axis = (axis,)

        # Check if axis is a tuple or list of integers
        if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
            raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")

        if not all(0 <= a < data.ndim for a in axis):
            raise ValueError("All axes must be valid for the input array.")

        # Compute the mean and standard deviation along the specified axis
        mean_vals = data.mean(axis=axis, keepdims=True)
        std_vals = data.std(axis=axis, keepdims=True)

        # Normalize the data
        normalized_data = (data - mean_vals) / (std_vals + self.eps)

        # Store the normalization parameters in a dictionary
        self.transformation_params = {'mean_vals': mean_vals, 'std_vals': std_vals}
        return normalized_data

    def transform(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Call fit_transform first!
        transform data using the transformation parameters and axis used in fit_transform
        :param np.ndarray data: The input array to be normalized.
        :param meta_data: optional data description
        :return np.ndarray: The normalized array.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'mean_vals' not in self.transformation_params or 'std_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'mean_vals' and 'std_vals'.")

        # Use given mean and std values
        mean_vals = self.transformation_params['mean_vals']
        std_vals = self.transformation_params['std_vals']

        # Check if the shapes of the mean and std values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if mean_vals.shape != expected_shape or std_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")
        # Normalize the data
        normalized_data = (data - mean_vals) / (std_vals + self.eps)
        return normalized_data

    def revert(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Reverts the standardization of a numpy array using the transformation parameters and axis from fit_transform.
        :param np.ndarray data: The standardized array to be reverted.
        :param meta_data: optional data description
        :return np.ndarray: The original array after reverting standardization.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'mean_vals' not in self.transformation_params or 'std_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'mean_vals' and 'std_vals'.")

        # Use given mean and std values
        mean_vals = self.transformation_params['mean_vals']
        std_vals = self.transformation_params['std_vals']

        # Check if the shapes of the mean and std values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if mean_vals.shape != expected_shape or std_vals.shape != expected_shape:
            raise ValueError("Transformation parameters must have the correct shape.")

        # Revert the normalization
        original_data = data * std_vals + mean_vals

        return original_data


class FFTMinMaxScaler(Scaler):
    """
    Scales Fourier transformed data using MinMaxScaler within range 0.1..0.9.
    Each Frequency component (voltages/ admittances) needs individual scaling.
    """

    def __init__(self, base_store_path: str):
        super().__init__()
        self.store_path = base_store_path
        self.axis = None
        self.transformation_params_admittances = {}
        self.data_labels_per_row = None
        self.no_of_admittances = 0
        self.no_of_voltages = 0
        self.no_of_complex_voltages = 0
        self.is_input_scaler = True
        self.no_max_connections = 0
        self.converter = None

    def _prepare_fit_fft_data(self, data_labels_per_row, grid_name, converter: SplitComplexConverter,
                              known_admittances_dict=None):

        import os
        self.converter = converter
        path_admittances = os.path.join(self.store_path, grid_name, "defaultAdmittances.pic")
        if os.path.exists(path_admittances):
            import pickle
            with open(path_admittances, 'rb') as pic:
                known_admittances_dict = pickle.load(pic)

        self.data_labels_per_row = data_labels_per_row
        # convert and remove zeros
        all_matrices_as_vector = [known_admittances_dict[m][np.nonzero(np.triu(known_admittances_dict[m], k=1))] for
                                  m in known_admittances_dict.keys()]
        # analyze dimensions of admittances

        self.no_max_connections = np.count_nonzero(np.abs(next(iter(known_admittances_dict.values())))[:-1], axis=1).max()
        self.no_of_admittances = len(all_matrices_as_vector[0])
        list_length = len(all_matrices_as_vector)
        vector_admittances = np.array(all_matrices_as_vector).reshape(list_length, 1, self.no_of_admittances)
        vector_admittances = converter.convert(vector_admittances)
        # analyze all precalculated admittances
        min_vals = vector_admittances.min(axis=(0, 1, 2), keepdims=True)
        max_vals = vector_admittances.max(axis=(0, 1, 2), keepdims=True)

        self.transformation_params_admittances = {'min_vals': min_vals, 'max_vals': max_vals}

    def fit_transform(self, data: np.ndarray, axis=None, data_labels_per_row=None, grid_name="",
                      converter: SplitComplexConverter = None, known_admittances_dict=None) -> np.ndarray:
        """
        Fit voltages to the min and max voltage and admittances to the known admittances.
        :param np.ndarray data: The input array to be normalized.
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :param data_labels_per_row:
        :param grid_name:
        :param data_labels_per_row:
        :param converter:
        :param known_admittances_dict:
        :return np.ndarray: normalized array
        """
        if data_labels_per_row is not None:
            self._prepare_fit_fft_data(data_labels_per_row, grid_name, converter, known_admittances_dict)
            self.no_of_voltages = data.shape[-1] - self.no_max_connections
            self.no_of_complex_voltages = int(self.no_of_voltages / 2)
            new_shape = data.shape[:-1] + (data.shape[-1] - self.no_of_complex_voltages,)
        else:
            self.no_of_voltages = data.shape[-1]
            self.no_of_complex_voltages = int(self.no_of_voltages / 2)
            self.is_input_scaler = False
            new_shape = data.shape[:-1] + (self.no_of_complex_voltages, )
            self.no_max_connections = 0
        if not axis:
            axis = tuple(range(data.ndim))
        self.axis = axis
        self.converter = converter
        # If axis is a single integer, convert it into a tuple
        if isinstance(axis, int):
            axis = (axis,)

        # Check if axis is a tuple or list of integers
        if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
            raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")
        if not all(0 <= a < data.ndim for a in axis):
            raise ValueError("All axes must be valid for the input array.")

        # Compute the minimum and maximum values along the specified axis
        # prepare voltages per frequencies
        all_data_sets = [f[1] for f in self.data_labels_per_row]
        unique_frequencies = np.unique(all_data_sets) # ToDo integrate in data split

        # pre init data_n
        data_n = np.zeros(new_shape, dtype=complex)
        data_buffer = copy.deepcopy(data)

        # indices for conversion
        magnitude_indices = np.arange(0, self.no_of_voltages, 2)
        angle_indices = np.arange(1, self.no_of_voltages, 2)
        self.transformation_params = {}

        for f in unique_frequencies:
            # get all datasets of the same frequency
            indices_f = self._get_indices_at_freq(all_data_sets, f)

            # for prescaling
            min_vals = data[indices_f].min(axis=axis, keepdims=True)[..., self.no_max_connections::2]
            max_vals = data[indices_f].max(axis=axis, keepdims=True)[..., self.no_max_connections::2]

            # Normalize only the magnitudes of the voltages
            data_buffer[indices_f, :, self.no_max_connections::2] = (
                    0.1 + 0.8 * ((data_buffer[indices_f, :, self.no_max_connections::2] - min_vals) /
                                                        (max_vals - min_vals + self.eps)))

            # Convert to splitted complex
            buffer_complex = polar_to_cartesian(data_buffer[indices_f, :, self.no_max_connections:],
                                                mag_index=magnitude_indices,
                                                phase_index=angle_indices, angle_type='degrees')

            data_n[indices_f, :, self.no_max_connections:] = buffer_complex[..., magnitude_indices] + 1j*buffer_complex[..., angle_indices]

            self.transformation_params[f] = {'min_vals': min_vals, 'max_vals': max_vals}
        # Convert complex datatype to array of components
        # data_n = np.nan_to_num(data_n, nan=0.1)
        data_n = converter.convert(data_n)
        data_buffer = converter.convert(data_buffer)
        if self.is_input_scaler:
            # Min Max scaling for admittances
            data_n[..., :self.no_max_connections, :] = \
                ((data_buffer[..., :self.no_max_connections, :] - self.transformation_params_admittances["min_vals"]) /
                (self.transformation_params_admittances["max_vals"] -
                 self.transformation_params_admittances["min_vals"] + self.eps))
        print("Data fitted")
        return data_n

    @staticmethod
    def _get_indices_at_freq(all_data_sets, freq, tolerance=1e-9):
        # Wenn es eine Liste ist, kannst du es in ein Array umwandeln: np.array(all_data_sets)
        all_data_np = np.array(all_data_sets)
        # Verwende np.isclose auf das gesamte Array
        close_elements = np.isclose(all_data_np, freq, atol=tolerance)
        # Finde die Indices der Elemente, die True sind
        return np.nonzero(close_elements)[0]

    def transform(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Call fit_transform first!
        transform data using the transformation parameters and axis used in fit_transform
        :param np.ndarray data: The input array to be normalized.
        :param meta_data: If needed, this is the data description.
        :return np.ndarray: The normalized array.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params, dict) or not len(self.transformation_params) > 0:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        if meta_data is None:
            meta_data = self.transformation_params

        # prepare voltages per frequencies
        all_data_sets = [f[1] for f in meta_data]
        unique_frequencies = np.unique(all_data_sets)

        # pre init data_n
        new_shape = data.shape[:-1] + (data.shape[-1] - self.no_of_complex_voltages,)
        data_n = np.zeros(new_shape, dtype=complex)
        data_buffer = copy.deepcopy(data)
        magnitude_indices = np.arange(0, self.no_of_voltages, 2)
        angle_indices = np.arange(1, self.no_of_voltages, 2)

        for f in unique_frequencies:
            # get all datasets of the same frequency
            indices_f = self._get_indices_at_freq(all_data_sets, f)
            f_closest_params = self._get_closest_freq(f)
            # Use given min and max values
            min_vals = f_closest_params['min_vals']
            max_vals = f_closest_params['max_vals']

            # Normalize only the magnitudes of the voltages
            data_buffer[indices_f, :, self.no_max_connections::2] = (
                    0.1 + 0.8 * ((data_buffer[indices_f, :, self.no_max_connections::2] - min_vals) /
                                 (max_vals - min_vals + self.eps)))

            # Convert to splitted complex
            buffer_complex = polar_to_cartesian(data_buffer[indices_f, :, self.no_max_connections:],
                                                mag_index=magnitude_indices,
                                                phase_index=angle_indices, angle_type='degrees')

            data_n[indices_f, :, self.no_max_connections:] = buffer_complex[..., magnitude_indices] + 1j * \
                                                             buffer_complex[..., angle_indices]

        # Convert complex datatype to array of components
        data_n = self.converter.convert(data_n)
        data_buffer = self.converter.convert(data_buffer)
        if self.is_input_scaler:
            # Min Max scaling for admittances
            data_n[..., :self.no_max_connections, :] = \
                ((data_buffer[..., :self.no_max_connections, :] - self.transformation_params_admittances["min_vals"]) /
                 (self.transformation_params_admittances["max_vals"] -
                  self.transformation_params_admittances["min_vals"] + self.eps))
        print("Data scaled")
        return data_n

    def revert(self, data: np.ndarray, meta_data=None) -> np.ndarray:
        """
        Reverts the standardization of a numpy array using the transformation parameters and axis from fit_transform.
        :param np.ndarray data: The standardized array to be reverted.
        :param meta_data: Data description - if needed .
        :return np.ndarray: The original array after reverting standardization.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params, dict) or not len(self.transformation_params) > 0:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        if meta_data is None:
            meta_data = self.transformation_params

        # prepare voltages per frequencies
        all_data_sets = [f[1] for f in meta_data]
        unique_frequencies = np.unique(all_data_sets)
        # pre init data_r
        new_shape = (data.shape[:-2]) + (data.shape[-2] * data.shape[-1],)
        data_r = np.zeros(new_shape)

        for f in unique_frequencies:
            # get all datasets of the same frequency
            indices_f = self._get_indices_at_freq(all_data_sets, f)
            f_closest = self._get_closest_freq(f)
            # Use given min and max values
            min_vals = f_closest['min_vals']
            max_vals = f_closest['max_vals']
            buffer = cartesian_to_polar(data[indices_f], angle_type='degrees', axis=3)

            buffer = buffer.reshape(buffer.shape[:-2] + (buffer.shape[-2] * buffer.shape[-1],))
            # Revert the voltage normalization by frequency
            data_r[indices_f, :, 0::2] = (((buffer[..., 0::2] - 0.1)/0.8) * (max_vals - min_vals + self.eps) + min_vals)
            data_r[indices_f, :, 1::2] = buffer[..., 1::2]
        print("Reverted Data")
        return data_r

    """
    def _scale_y_mats(self, y_mats_dict):
        df_y_mats = pd.DataFrame.from_dict(y_mats_dict, orient='index')

        column_names = [i for i in range(87)]
        df_y_mats.columns = column_names

        df_y_mats.index.name = 'Frequency'

        magnitude_df = df_y_mats.apply(np.abs)
        angle_df = df_y_mats.apply(np.angle)

        normalized_magnitude_df = magnitude_df.apply(lambda x: 0.1 + 0.8 * (x - x.min()) / (x.max() - x.min()))

        # normalized_magnitude_df = normalized_magnitude_df.replace(0, 1e-20)

        normalized_complex_df = pd.DataFrame(index=df_y_mats.index)

        for col in tqdm(df_y_mats.columns, desc="normalize y-mats"):
            normalized_complex_df[col] = self.polar_to_cartesian(normalized_magnitude_df[col], angle_df[col])

        normalized_complex_dict = normalized_complex_df.to_dict(orient="index")

        for key in tqdm(normalized_complex_dict, desc="df to dict"):
            normalized_complex_dict[key] = list(normalized_complex_dict[key].values())

        return normalized_complex_dict
    """
    def _get_closest_freq(self, f):
        # Check if the exact index exists
        if f in self.transformation_params:
            return self.transformation_params[f]
        else:
            # Find the key with the minimum difference to the given value
            nearest_key = min(self.transformation_params.keys(), key=lambda k: abs(k - f))
            return self.transformation_params[nearest_key]


def transform_matrix(mat, k=10):
    """
    Apply the logarithmic transformation to a matrix.
    """
    mask = (mat != 0)
    transformed = np.log1p(k * mat) / np.log1p(k)
    return np.where(mask, transformed, mat)


def redistribute_values(arr: np.ndarray, k: int = 10, axis=None) -> np.ndarray:
    """
    Redistribute values in an array using a logarithmic function for matrices
    specified by the target_axis.

    :param arr: numpy array of any shape
    :param k: scaling factor for redistribution strength
    :param axis: integer, tuple or list of integer along which to redistribute values
    :return: redistributed array
    """
    # copy array to avoid inplace operations
    arr = arr.copy()
    if axis is None:
        axis = tuple(range(arr.ndim))
    # If axis is a single integer, convert it into a tuple
    if isinstance(axis, int):
        axis = (axis,)

    # Check if axis is a tuple or list of integers
    if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
        raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")

    # Generate indices for all non-target axis
    non_target_axis = [i for i in range(arr.ndim) if i not in axis]
    non_target_dims = [arr.shape[i] for i in non_target_axis]
    from itertools import product
    all_indices = product(*[range(dim) for dim in non_target_dims])

    # Iterate over all non-target axis combinations and apply the transformation
    # on the slices defined by the target_axis
    for indices in all_indices:
        slicer = [slice(None)] * arr.ndim
        for i, idx in zip(non_target_axis, indices):
            slicer[i] = idx
        arr[tuple(slicer)] = transform_matrix(arr[tuple(slicer)], k=k)

    return arr
