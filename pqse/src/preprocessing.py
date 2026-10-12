import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn.model_selection as skms
from scipy import stats, interpolate

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.simulation import SimulationMode
from OpenDSSWrapper.simulation.analyze_results import read_lab_data
from ml_multi_dim_toolbox.split_complex import SplitComplexConverter, polar_to_cartesian, SplitComplexMode, \
    polar_to_complex, cartesian_to_polar, complex_to_polar
from tqdm import tqdm
from enum import Enum


class EstimationModes(Enum):
    TSE = 0
    HSE = 1


def read_data(data_path, network_data, grid, input_converter: SplitComplexConverter = None,
              target_converter: SplitComplexConverter = None, transpose_data_format=False):
    network_data['data_length_in_min'] = 527026
    network_data['step_width_in_min'] = 15
    network_data['input_converter'] = input_converter
    network_data['target_converter'] = target_converter
    network_data['grid'] = grid
    network_data['target_voltage'] = 20.

    network_data = read_network_data(data_path, network_data, transpose_data_format=transpose_data_format)

    if "Y_matrices_per_frequency" not in network_data:
        frequencies = list(np.arange(50, 1050, 50))
        network_data['Y_matrices_per_frequency'] = get_y_mats(grid, frequencies, voltage_level_kv=20.,
                                                              data_format=network_data['data_format'],
                                                              modes=input_converter.modes)

    scaling_axes = (0, 1) if network_data['data_format'] == 'channels_last' else (0, 3)
    scale_data(network_data, axes=scaling_axes, sep_output_scaler=True)
    return network_data

def mask_data(data, feature_indices, feature_axis):
    """
    Apply the mask to the input data and reshape the result.
    :param data: The input data array.
    :param feature_indices: Indices of the features to be masked.
    :param feature_axis: Axis of the features.
    :return: Reshaped masked data array.
    """
    # if indexing is negative, convert it to positive
    feature_axis = feature_axis if feature_axis >= 0 else feature_axis + data.ndim

    # Construct a tuple for advanced indexing
    index_tuple = tuple(feature_indices if axis == feature_axis else slice(None) for axis in range(data.ndim))

    # Extract the features along the specified axis using advanced indexing
    masked_data = data[index_tuple]
    return masked_data

def attach_admittances_and_reshape(x, network_data, store_path=None, use_buffered_files=True, range_freq=None):
    """
    This method calculates the grid's admittances according to the frequencies in the dataset
    :param x: The input voltage/current data array.
    :param network_data: network data holding default admittances and the frequency information
    :param store_path: If not None precalculated admittance-matrices are loaded and stored if a difference occured
    :param use_buffered_files: use the flag to by_pass algorithm and load temporary files.
    :return: new input data array with voltage vector and corresponding admittances@frequencies.
    """

    temp_buffer_file = 'temp_voltages_and_admittances.pic'
    temp_buffer_path_f = os.path.join(store_path, 'temp_voltages_and_admittances.pic')
    admittances_file = 'defaultAdmittances.pic'
    admittances_path_f = os.path.join(store_path, 'defaultAdmittances.pic')
    if use_buffered_files:
        if store_path is not None and os.path.exists(admittances_path_f):
            network_data['Y_matrices_per_frequency'] = load_pickle(store_path, admittances_file)
        if store_path is not None and os.path.exists(temp_buffer_path_f):
            print('loading stored admittances with voltage vectors.')
            return load_pickle(store_path, temp_buffer_file)

    frequency_array = [f[1] for f in network_data['frequencies']]

    # Either calculate custom selection of admittances or the required ones for the dataset
    needed_frequencies = frequency_array if range_freq is None else range_freq + frequency_array
    needed_frequencies = np.unique(needed_frequencies)
    needed_frequencies.sort()

    # load defaults if available
    if not 'Y_matrices_per_frequency' in network_data.keys():
        network_data['Y_matrices_per_frequency'] = {}

    default_freq_keys = network_data['Y_matrices_per_frequency'].keys()
    missing_frequencies = [f for f in needed_frequencies if f not in default_freq_keys]

    if len(missing_frequencies) > 0:
        new_y_matrices = get_y_mats_complex_np(network_data['grid'], missing_frequencies,
                                               voltage_level_kv=network_data['target_voltage'], compact=False)
        y_matr_dict = {key: new_y_matrices[i] for i, key in enumerate(missing_frequencies)}
        network_data['Y_matrices_per_frequency'].update(y_matr_dict)
        # update file
        if store_path is not None and os.path.exists(store_path):
            save_pickle(store_path, admittances_file, network_data['Y_matrices_per_frequency'])


    # analyze dimensions of admittances
    upper_triangular = np.triu(network_data['Y_matrices_per_frequency'][50.0], k=1)
    # get indices of upper triangular, which are not (0 +j0)
    non_zero_indices = np.nonzero(upper_triangular)

    phasors_length = int(x.shape[-1]/2)
    busses_length = x.shape[-2]
    new_dim = (int(phasors_length * busses_length))
    x_reshaped = x.reshape(x.shape[0], new_dim, 2)
    # Run over unique frequencies to attach the admittances by vector operations
    x_combined = np.zeros((x.shape[0], new_dim + len(non_zero_indices[0]), 2), np.float64)
    print('Attaching admittances to voltages...')
    for f in tqdm(needed_frequencies):
        indices_f = np.where(np.isclose(frequency_array, f, 1e-9))[0]
        x_combined[indices_f, :new_dim] = x_reshaped[indices_f, :]
        cart_admittances = network_data['Y_matrices_per_frequency'][f][non_zero_indices]
        # Add axis at the end of the array
        res = np.zeros(cart_admittances.shape + (2,))
        res[..., 0] = cart_admittances.real
        res[..., 1] = cart_admittances.imag
        x_combined[indices_f, new_dim:] = res

    if use_buffered_files and store_path is not None:
        save_pickle(store_path, temp_buffer_file, x_combined)
    print('Attached admittances to voltage vectors.')
    return x_combined


def read_network_data(data_path, network_data, transpose_data_format=True):
    """
    read network data from file path and split into train, test and validation data if necessary
    expects the following structure if train test split is performed later:
    data_path
        grid_name
            y_all.pic
    or the following if train test split is already done:
    data_path
        grid_name
            y_train.pic
            y_test.pic
            y_validation.pic

    :param data_path: path to the data
    :param network_data: dictionary with keys: grid_name, test_size, validation_size
    :return: dictionary with additional keys: x_train, y_train, x_test, y_test, x_val, y_val, input_dims, output_dims, ...
    """
    data_path = os.path.join(data_path, network_data['grid_name'])
    test_size = network_data['test_size']
    validation_size = network_data['validation_size']
    input_converter = network_data['input_converter']
    target_converter = network_data['target_converter']
    mask_axis = 1 if network_data['data_format'] == "channels_last" else -1
    harmonic_axis = 1 if network_data['data_format'] == "channels_first" else -1
    if check_files_exist(data_path, ['y_train.pic', 'y_test.pic', 'y_validation.pic']):
        # Dataset is split up in train test and validation set
        with open(os.path.join(data_path, 'y_train.pic'), 'rb') as pic:
            train = pickle.load(pic)
        # TODO: convert from dict to numpy array
        train_np, frequencies = convert_to_numpy(train) if network_data['estimation_mode'] == EstimationModes.HSE else (
                                                                                    convert_to_numpy_fft_data(train))
        if transpose_data_format:
            train_np = np.transpose(train_np, (0, 2, 1))
        network_data['frequencies'] = frequencies
        train_np = rotate_unmeasurable_angles(train_np, harmonic_axis=harmonic_axis,
                                              harmonics=[2, 5, 8, 11, 14, 17])
        x_train = mask_data(train_np, network_data['mask_x_test'], mask_axis)
        network_data['x_train'] = input_converter.convert(x_train)
        network_data['y_train'] = target_converter.convert(train_np)
        network_data['input_dims'] = network_data['x_train'].shape[1:]
        network_data['output_dims'] = network_data['y_train'].shape[1:]
        with open(os.path.join(data_path, 'y_test.pic'), 'rb') as pic:
            test = pickle.load(pic)
        test_np, _ = convert_to_numpy(test)
        if transpose_data_format:
            test_np = np.transpose(test_np, (0, 2, 1))
        test_np = rotate_unmeasurable_angles(test_np, harmonic_axis=harmonic_axis,
                                             harmonics=[2, 5, 8, 11, 14, 17])
        x_test = mask_data(test_np, network_data['mask_x_test'], mask_axis)
        network_data['x_test'] = input_converter.convert(x_test)
        network_data['y_test'] = target_converter.convert(test_np)
        with open(os.path.join(data_path, 'y_validation.pic'), 'rb') as pic:
            validation = pickle.load(pic)
        val_np, _ = convert_to_numpy(validation)
        if transpose_data_format:
            val_np = np.transpose(val_np, (0, 2, 1))
        val_np = rotate_unmeasurable_angles(val_np, harmonic_axis=harmonic_axis, harmonics=[2, 5, 8, 11, 14, 17])
        x_val = mask_data(val_np, network_data['mask_x_test'], mask_axis)
        network_data['x_val'] = input_converter.convert(x_val)
        network_data['y_val'] = target_converter.convert(val_np)
    else:
        all_data_path = os.path.join(data_path, 'data.pic')
        # merge multiple files if tse data is used
        if not os.path.exists(all_data_path) and network_data['estimation_mode'] == EstimationModes.TSE:

            all = merge_multiple_df_pic_files(data_path, network_data['grid_name'] + '_train_hse_conv', np.arange(1,4,1))
            #all = merge_multiple_df_pic_files(data_path, network_data['grid_name'] + '_train', np.arange(1,27,1))
        else:
            with open(all_data_path, 'rb') as pic:
                all = pickle.load(pic)
        all_np, frequencies = convert_to_numpy(all) if network_data['estimation_mode'] == EstimationModes.HSE else (
                                                                convert_to_numpy_fft_data(all, store_path=data_path))
        if transpose_data_format:
            all_np = np.transpose(all_np, (0, 2, 1))
        network_data['frequencies'] = frequencies
        if network_data['estimation_mode'] == EstimationModes.HSE:
            all_np = rotate_unmeasurable_angles(all_np, harmonic_axis=harmonic_axis, harmonics=[2, 5, 8, 11, 14, 17]) #\
                # if () else (rotate_unmeasurable_angles_fft_data(all_np, frequencies, store_path=data_path))

        x = mask_data(all_np, network_data['mask_x_test'], mask_axis)
        if network_data['estimation_mode'] == EstimationModes.TSE:
            x = attach_admittances_and_reshape(x, network_data, store_path=data_path,
                                               range_freq=list(np.arange(50, 1250 , 50)))
            phasors_length = int(all_np.shape[-1] / 2)
            y = all_np.reshape(all_np.shape[0], all_np.shape[1], phasors_length, 2)
        else:
            x = input_converter.convert(x)
            y = target_converter.convert(all_np)

        network_data['x_train'], network_data['x_test'], network_data['y_train'], network_data[
            'y_test'], network_data['labels_train'], network_data['labels_test'] = skms.train_test_split(
            x, y, network_data['frequencies'], test_size=test_size, random_state=1, shuffle=False)

        (network_data['x_train'], network_data['x_val'], network_data['y_train'], network_data['y_val'],
         network_data['labels_train'], network_data['labels_val']) = skms.train_test_split(
            network_data['x_train'], network_data['y_train'], network_data['labels_train'],
            test_size=validation_size, random_state=1, shuffle=False)

        network_data['input_dims'] = network_data['x_train'].shape[1:]
        network_data['output_dims'] = network_data['y_train'].shape[1:]

    return network_data

def convert_hse_to_tse(list_hse, idx_dataset) -> pd.DataFrame:
    columns = ['v1', 'vangle1']#,'v2', 'vangle2','v3', 'vangle3']
    index_names = ['step', 'bus', 'element', 'element_type', 'freq']
    state_prefix = 'train' if idx_dataset == 1 else ('val' if idx_dataset == 3 else 'test')

    indices = []
    states_v = []
    states_vangle = []
    for idx_st, state in enumerate(list_hse):
        for freq in state.keys():
            for bus in range(len(state[freq])):
                indices.append([state_prefix + '_' + str(idx_st),bus,0,0,freq])
                states_v.append(np.abs(state[freq][bus]))
                states_vangle.append(np.angle(state[freq][bus],deg=True))
    multi_index = pd.MultiIndex.from_tuples(indices, names=index_names)
    dummy_data = [0 for i in range(len(states_v))]
    df = pd.DataFrame({'v1': states_v, 'vangle1': states_vangle,
                            'i1':dummy_data, 'iangle1':dummy_data,
                            'i2':dummy_data, 'iangle2':dummy_data,
                            'i3':dummy_data, 'iangle3':dummy_data}, index=multi_index)
    return df


def merge_multiple_df_pic_files(base_path, base_name, num_range, convert_hse=False):
    all_pics = []

    if len(num_range) == 0:
        return pd.read_pickle(os.path.join(base_path, f'{base_name}_{1}'))
    for i in num_range:
        f_name = os.path.join(base_path, f'{base_name}_{i}')
        df_data = pd.read_pickle(f_name)
        if convert_hse:
            df_data = convert_hse_to_tse(df_data, i)
            pd.to_pickle(df_data, os.path.join(base_path, f'{base_name}_conv_{i}'))
        all_pics.append(df_data)
    return pd.concat(all_pics)


def add_measurement_noise(x, noise_level, feature_axis=2, scale_on_max=False):
    """
    Add scaled measurement noise to the voltage measurements.
    Either ensure that for each feature the values reside in the same range or scale the noise individually for each
    feature.
    The noise is scaled by the maximum value of the measurements to keep the noise proportional to the data magnitude.


    :param x: voltage measurements
    :param noise_level: standard deviation of the noise, relative to the maximum value of x
    :param feature_axis: axis along which the maximum value is calculated
    :param scale_on_max: if True scales the noise on the maximum value of the input data individually for each harmonic
    :return: x with added noise
    """
    if scale_on_max:
        feature_axis = feature_axis if isinstance(feature_axis, tuple) else (feature_axis, )
        selected_axes = tuple(set(range(x.ndim)) - set(feature_axis))
        max_values = np.max(x, axis=selected_axes)
        target_shape = [1] * x.ndim
        for axis in feature_axis:
            target_shape[axis] = x.shape[axis]
        scaled_noise = max_values.reshape(target_shape) * np.random.normal(0, noise_level, x.shape)
    else:
        scaled_noise = np.random.normal(0, noise_level, x.shape)
    return x + scaled_noise


def convert_to_numpy_fft_data(df_fft_states, drop_columns=None, store_path=None):
    """
    Convert a list of dictionaries to a numpy array.
    :param df_fft_states: a dataframe containing all grid states by "Name of Event", "Power of Event", "Bus", "Freq"
    :param drop_columns: If necessary drop specific columns
    :param store_path:  If not None precalculated objects are loaded and algorithms are bypassed.
    :return: multidimensional numpy array in dimension: Grid-States (Event, Power) x freq x bus
    """
    temp_array_f_name = 'numpy_array.pic'
    temp_states_f_name = 'dict_states.pic'
    if store_path is not None and os.path.exists(os.path.join(store_path, temp_array_f_name)):
        arr = load_pickle(store_path, temp_array_f_name)
        print('Loading stored numpy voltages and state descriptions.')
        return arr, load_pickle(store_path, temp_states_f_name)

    # Assuming the list is non-empty and can have different keys (Transient State Estimation)
    if df_fft_states is None:
        raise ValueError("The dataframe is empty.")
    if len(df_fft_states) == 0:
        raise ValueError("The dataframe is empty.")

    # Extract and save the keys (frequencies for admittances and system_states for reconstruction)
    # frequencies = df_fft_states.index.get_level_values('freq').unique()
    system_states = df_fft_states.index.get_level_values('step').unique()
    # Pre-allocate a list to hold all arrays
    all_arrays = []
    all_frequencies_and_system_states = []

    # drop the not wanted information
    if drop_columns is None:
        drop_columns = ['i1', 'iangle1', 'i2', 'iangle2', 'i3', 'iangle3']

    print('Converting dataframe to numpy matrix...')
    # Iterate over each measurement
    for state in tqdm(system_states):
        # get one state over its frequencies (e.g., Cigre LV x-freq x 44 buses x 3 Phase Voltages
        rows_state = df_fft_states.xs(state, level='step')
        frequencies_state = rows_state.index.get_level_values('freq').unique()

        states_per_freq = []
        for freq in frequencies_state:
            all_frequencies_and_system_states.append((state, freq))
            states_per_freq.append(rows_state.xs(freq, level='freq')
                                   .sort_index(level='bus').drop(columns=drop_columns))

        all_arrays.extend(states_per_freq)

    # Convert the list of arrays to a numpy array
    numpy_array = np.array(all_arrays)

    if store_path is not None:
        save_pickle(store_path, temp_array_f_name, numpy_array)
        save_pickle(store_path, temp_states_f_name, all_frequencies_and_system_states)
    print('Loaded and converted voltages to np array.')
    return numpy_array, all_frequencies_and_system_states

def rotate_fft_by_fundamental_angle(df_measurements):
    """
    angles of harmonic data from the openDSS simulation are relative to the fundamental. This method creates
    absolute angles by considering the fundamental's absolute phasor
    """
    # get angle columns
    angle_columns = [col for col in df_measurements.columns if 'angle' in col]

    def process_group(group):
        # Hole den ersten Eintrag jeder "angle"-Spalte
        first_row = group.iloc[0]
        fundamental_f = float(first_row.name[1])
        for col in angle_columns:
            first_value = first_row[col]
            # add fundamental offset to other frequencies
            group[col] = group[col] + first_value * group.index.get_level_values('freq')/fundamental_f
        return group

    # Führe die Operation für jede "bus"-Gruppe durch
    return df_measurements.groupby('bus').apply(process_group)


def convert_to_numpy(list_of_dicts):
    """
    Convert a list of dictionaries to a numpy array.
    :param list_of_dicts:
    :return: multidimensional numpy array
    """
    # Assuming the list is non-empty and can have different keys (Transient State Estimation)
    if list_of_dicts is None:
        raise ValueError("The list of dictionaries is empty.")

    # Check which type is read from pickle-file
    if isinstance(list_of_dicts, pd.DataFrame):
        df_reset = list_of_dicts.reset_index()
        list_of_dicts = df_reset.to_dict(orient='records')

    # Extract and save the keys (frequencies)
    frequencies = list(list_of_dicts[0].keys())

    # Pre-allocate a list to hold all arrays
    all_arrays = []

    # Iterate over each dictionary in the list
    for dct in list_of_dicts:
        # Verify that the keys are consistent
        if list(dct.keys()) != frequencies:
            raise ValueError("Inconsistent keys in dictionaries.")

        # Stack the arrays from this dictionary
        current_array = np.stack([dct[freq] for freq in frequencies], axis=0)
        all_arrays.append(current_array)

    # Convert the list of arrays to a numpy array
    numpy_array = np.array(all_arrays)

    return numpy_array, frequencies


def scale_data(network_data, axes=(0, 3), sep_output_scaler=True):
    """
    Scale the data using the scaler defined in network_data.
    Different scalers are used only if the parameter is set or if the shape mismatches on axes other than the specified ones.
    This happens only if the features of input and target have a different shape (e.g. mixed cartesian / polar input and cartesian output)
    :param network_data: dictionary
    :param axes: tuple, axes along which the scaling is to be consistent
    :param sep_output_scaler: if True uses a separate scaler for the output
    (required if output shape on the non-specified axes does not match input shape,
    e.g. if input consists of polar and cartesian coordinates while input consists only of cartesian)
    :return: network_data with scaled data and scalers for input and output
    """
    scaler = network_data['scaler']
    # Prepare scaling
    if network_data['estimation_mode'] == EstimationModes.TSE:
        meta_data = [network_data['labels_test'], network_data['labels_val']]
        network_data['x_train_n'] = scaler.fit_transform(network_data['x_train'], axes, network_data['labels_train'],
                                                         network_data['grid_name'], network_data['input_converter'])
    else:
        meta_data = [None, None]
        network_data['x_train_n'] = scaler.fit_transform(network_data['x_train'], axes)
    # scale validation and test on training parameters
    network_data['x_test_n'] = scaler.transform(network_data['x_test'], meta_data=meta_data[0])
    network_data['x_val_n'] = scaler.transform(network_data['x_val'], meta_data=meta_data[1])

    # Determine if shapes mismatch on non-specified axes
    shape_mismatch = False
    input_shape = list(network_data['x_train'].shape)
    output_shape = list(network_data['y_train'].shape)
    for axis in set(range(max(len(input_shape), len(output_shape)))) - set(axes):
        # Adjust index out of range for shorter shape
        if axis >= len(input_shape) or axis >= len(output_shape) or input_shape[axis] != output_shape[axis]:
            shape_mismatch = True
            break

    # Use different scaler for output if shape mismatches on non-specified axes
    if shape_mismatch or sep_output_scaler:
        # save the previous scaler along with internal parameters as input scaler
        network_data['input_scaler'] = deepcopy(scaler)
        network_data['y_train_n'] = scaler.fit_transform(network_data['y_train'], axes)
    else:
        network_data['y_train_n'] = scaler.transform(network_data['y_train'])
    network_data['y_test_n'] = scaler.transform(network_data['y_test'], meta_data[0])
    network_data['y_val_n'] = scaler.transform(network_data['y_val'], meta_data[1])
    return network_data


def check_files_exist(path, file_names):
    """
    For each file name in file_names check if it exists in path
    :param path: path
    :param file_names: list of file names
    :return: False if any file does not exist, else True if all files exist
    """
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def wrap_angle(angle, degree=True):
    """
    Vectorized wrap angle to [-180, 180] range.
    :param angle: Angle in degrees, numpy array
    :return: Wrapped angle, numpy array
    """
    if degree:
        angle = angle % 360
        angle[angle > 180] -= 360
        angle[angle <= -180] += 360
    else:
        angle = angle % (2*np.pi)
        angle[angle > np.pi] -= 2 * np.pi
        angle[angle <= -np.pi] += 2 * np.pi
    return angle


def rotate_angles_to_reference_node(df: pd.DataFrame, ref: int, columns, ref_col='vangle1', ref_freq=50.0):
    """
    Rotate the phase angles of the nodes to the reference node, considering specific columns and wrapping angles.
    :param df: Multi-indexed dataframe with index levels ['step', 'bus', 'element', 'element_type', 'freq']
    :param ref: Reference node
    :param columns: Columns to rotate
    :param ref_freq: Reference frequency
    :return: Rotated DataFrame
    """
    # Copy the DataFrame
    df = df.copy()
    ref_df = df.xs(key=(ref, ref_freq), level=('bus', 'freq'), drop_level=False)

    # Calculate mean reference angles per step
    ref_angle_per_step = ref_df.groupby(level='step')[columns].mean()
    # Broadcast reference angles to match df's shape
    broadcasted_ref_angles = df.index.get_level_values('step').map(ref_angle_per_step[ref_col])

    # Loop through columns to rotate
    for col in tqdm(columns,
                    desc=f"Rotating all angles of columns {columns} to reference node {ref} at frequency {ref_freq}"):
        # Subtract reference angles from df's columns and wrap angles
        df[col] = wrap_angle(df[col] - broadcasted_ref_angles.values)

    return df


def rotate_unmeasurable_angles_fft_data(arr: np.ndarray, freq_indices, angle_threshold=1e-9, magnitude_threshold=1e-9,
                                        convert_cartesian=True, store_path=None, use_buffered_files=True):
    """
    This method works similarly like the harmonic method, but considers all frequencies as especially higher frequent
    portions are damped along the electrical networks.
    :param arr: numpy array holding all datasets in format systems-state at frequency x bus x voltages L1,L2,L3
    :param freq_indices: list of frequencies, which describes the arr param data
    :param angle_threshold: threshold for the angle manipulations
    :param magnitude_threshold: threshold for the magnitude manipulations
    :param convert_cartesian: If True, the polar representation is onverted to cartesian
    :param store_path: for later predictions, the most common values need to be available for unseen data
    :param use_buffered_files: use the flag to by_pass algorithm and load temporary files.
    :return: the corrected voltage vectors in identical shape if convert_cartesian==False, else last dimension is dim/2
    """
    temp_buffer_f_name = 'temp_rotated_data.pic'
    if use_buffered_files and store_path is not None and os.path.exists(os.path.join(store_path, temp_buffer_f_name)):
        print('Loading corrected voltage vectors.')
        return load_pickle(store_path, temp_buffer_f_name)

    dim_voltages = arr.shape[2]
    # get unique frequencies from training data
    unique_freqs = np.unique([f[1] for f in freq_indices])
    angle_at_freq = np.zeros((unique_freqs.shape[0], 1 + int(dim_voltages/2)))
    print('Correcting voltages without magnitude...')
    for idx_f in tqdm(range(len(unique_freqs))):
        # all datasets corresponding to this frequency
        # relevant_indices = [freq_indices.index(f) for f in freq_indices if unique_freqs[idx_f] == f[1]]
        relevant_indices = [i for i,f in enumerate(freq_indices) if unique_freqs[idx_f] == f[1]]
        # get the most common values for every angle per phase (L1..L3) for correct readings
        most_common_values = np.zeros(int(dim_voltages/2))
        for idx in range(0, dim_voltages, 2):
            # get valid magnitudes
            indices_without_issues = np.where(arr[relevant_indices, :, idx] >= magnitude_threshold)
            # get related angles and their average
            most_common_values[int(idx/2)] = np.average(arr[indices_without_issues[0], indices_without_issues[1], idx+1].ravel())
            # fix the issue for all magnitudes, which are not measureable
            indices_with_issues = np.where(arr[relevant_indices, :, idx] < magnitude_threshold)
            # Apply correction on these values
            arr[indices_with_issues[0], indices_with_issues[1], idx+1] = most_common_values[int(idx/2)]
        angle_at_freq[idx_f, 0] = unique_freqs[idx_f]
        angle_at_freq[idx_f, 1:] = most_common_values
    if store_path:
        with open(os.path.join(store_path, 'fixed_angles.pic'), 'wb') as pic:
            pickle.dump(angle_at_freq, pic)
    if convert_cartesian:
        # convert all to cartesian complex representation
        cartesian_array = np.empty_like(arr[..., :int(dim_voltages/2)], complex)
        for idx in range(0, int(dim_voltages/2)):
            angles_rad = np.deg2rad(arr[..., idx * 2 + 1])
            real = arr[..., idx * 2] * np.cos(angles_rad)
            imag = arr[..., idx * 2] * np.sin(angles_rad)
            cartesian_array[..., idx] = real + 1j * imag
        if use_buffered_files and store_path is not None:
            save_pickle(store_path, temp_buffer_f_name, cartesian_array)
        print('Corrected unmeasureable angels and created .')
        return cartesian_array
    else:
        if use_buffered_files and store_path is not None:
            save_pickle(store_path, temp_buffer_f_name, arr)
        print('Loaded and converted voltages to np array.')
        return arr


def load_pickle(store_path, name):
    with open(os.path.join(store_path, name), 'rb') as pic:
        obj = pickle.load(pic)
    return obj


def save_pickle(store_path, name, obj):
    with open(os.path.join(store_path, name), 'wb') as pic:
        pickle.dump(obj, pic)



def rotate_unmeasurable_angles(arr: np.ndarray, harmonic_axis=2, harmonics=None, magnitude_threshold=1e-9,
                               angle_threshold=1e-9, angle_cutoff=0.4, mode="interpolate"):
    """
    Phase angles cannot be measured if the magnitude is zero or close to zero.
    This results in outliers in the complex plane which the neural network has difficulties estimating.
    This function fixes the problem by rotating the phase angle to the most common phase angle per frequency.
    Situations where this occurs include
     - triplen harmonics if delta-wye transformers are present
     - measurements with close-to-zero magnitude  (some measurement devices then automatically set the angle to zero)
    :param arr: numpy array of complex numbers
    :param harmonic_axis: axis along which the harmonics are stored
    :param harmonics: list of harmonic frequencies to be processed, 1-indexed (3 means 3rd harmonic)
    if None, all harmonics are processed
    :param magnitude_threshold: threshold below which magnitudes are considered as zero
    :param angle_threshold: threshold below which angles are considered as zero (in radians)
    :param angle_cutoff: cutoff value for angles in radians. Any phase value for magnitudes equal to zero outside
    the cutoff range will be set to the most common angle of non-zero phase / magnitude values.
    :param mode: "interpolate" or "common". If "interpolate" the phase angle is interpolated between nearby values.
    If "common" the most common angle is used for all unmeasurable angles.
    :return: Modified array
    """
    # move harmonic axis to the end
    arr = np.moveaxis(arr, harmonic_axis, -1)
    if harmonics is None:
        harmonics = range(arr.shape[-1])  # already 0-indexed
    else:
        # 0-index harmonics
        harmonics = [h - 1 for h in harmonics]
    if mode not in ["interpolate", "common"]:
        raise ValueError("mode must be either 'interpolate' or 'common'")

    # Loop through the specific frequencies
    for f in harmonics:
        slice_harm = arr[..., f]
        magnitude = np.abs(slice_harm)
        angle = np.angle(slice_harm)

        # Filter indices based on magnitude and phase conditions
        valid_indices = np.logical_and(magnitude >= magnitude_threshold, np.abs(angle) >= angle_threshold)
        if not np.any(valid_indices):  # at first consider only those where magnitude and angle are above threshold
            valid_indices = np.abs(angle) >= angle_threshold
        for phase in range(slice_harm.shape[2]):
            for node in range(slice_harm.shape[1]):  # iterate over each node
                if mode == "interpolate":
                    # 1d array of valid indices (those where values exist)
                    valid_indices_positions = np.argwhere(valid_indices[:, node, phase])
                    if len(valid_indices_positions) == 0:  # can't interpolate if no valid indices
                        continue
                    valid_angles = angle[valid_indices_positions, node, phase]
                    first_value = valid_angles[0]
                    last_value = valid_angles[-1]
                    # attach last element of array as -1 to handle wrap around
                    valid_indices_positions = np.insert(valid_indices_positions, 0, -1)
                    valid_angles = np.insert(valid_angles, 0, last_value)
                    # and first element at the end
                    valid_indices_positions = np.append(valid_indices_positions, slice_harm.shape[0])
                    valid_angles = np.append(valid_angles, first_value)

                    # Interpolate angles over the entire array
                    interp_func = interpolate.interp1d(valid_indices_positions, valid_angles, kind='linear')
                    interpolated_angles = interp_func(np.arange(angle[:, node, phase].size))
                    update_indices = np.logical_and(magnitude < magnitude_threshold, np.abs(angle) <= angle_threshold)[:,
                                     node, phase]
                    update_positions = np.argwhere(update_indices)
                    for pos in update_positions:
                        row = pos[0]  # Original indices
                        interpolated_value = interpolated_angles[row]
                        phase_value = angle[row, node]
                        if np.abs(phase_value - interpolated_value) > angle_cutoff:
                            angle[row, node] = interpolated_value
        # mode "common"
        # we don't need a mode if check here since all angles will have values if mode is interpolate unless
        # there we no valid indices at all. In that case this is the fallback
        if np.any(valid_indices):
            mode_angle = stats.mode(angle[valid_indices].ravel())[0]
        else:  # Fallback to zero
            mode_angle = 0
        update_indices = np.logical_and(magnitude < magnitude_threshold, np.abs(angle) <= angle_threshold)
        update_positions = np.argwhere(update_indices)
        for pos in update_positions:
            row, col = pos[0], pos[1]  # Original indices
            phase_value = angle[row, col]
            if np.abs(phase_value - mode_angle) > angle_cutoff:
                angle[row, col] = mode_angle

        # convert magnitude and phase back to complex number
        arr[..., f] = magnitude * np.exp(1j * angle)
    # move harmonic axis back to original position
    arr = np.moveaxis(arr, -1, harmonic_axis)
    return arr


def get_y_mats(grid: BaseGrid, frequencies, voltage_level_kv=0.4, data_format="channels_last",
               modes=SplitComplexMode.CARTESIAN, shunt_position=-1, compact=True, num_phases=1):
    """
    Get the admittance matrices for the given frequencies as numpy array.
    :param grid: Grid to get the admittance matrices from
    :param frequencies: frequencies to get the admittance matrices for
    :param voltage_level_kv: target voltage level in kV
    :param data_format: "channels_last" or "channels_first"
    :param modes: SplitComplexMode
    :param shunt_position: position of shunt admittance in the matrix. If None, shunt admittance is not saved
    :param compact: if True, shunt admittance is included in diagonal which is the sum of all off-diagonal elements.
    :param num_phases: The amount of phases. If other than 1, the dim with complex values is repeated for all phases.
    :return: numpy array of admittance matrices converted to the desired format
    """
    y_mats_np = get_y_mats_complex_np(grid, frequencies, voltage_level_kv, shunt_position, compact)

    # convert to split complex
    sc_conv = SplitComplexConverter(modes=modes, target_axis=3)
    y_mats_cart = sc_conv.convert(y_mats_np)
    if data_format == 'channels_first':
        y_mats_cart = np.swapaxes(y_mats_cart, 1, 3)
    else:
        y_mats_cart = np.swapaxes(y_mats_cart, 0, 2)
    return np.repeat(y_mats_cart[:, :, :, np.newaxis, :], num_phases, axis=num_phases)



def get_y_mats_complex_np(grid: BaseGrid, frequencies, voltage_level_kv=0.4, shunt_position=-1, compact=True):
    """
    Get the admittance matrices for the given frequencies as numpy array as complex numbers.
    :param grid: Grid to get the admittance matrices from
    :param frequencies: frequencies to get the admittance matrices for
    :param voltage_level_kv: target voltage level in kV
    :param shunt_position: position of shunt admittance in the matrix. If None, shunt admittance is not saved
    :param compact: if True, shunt admittance is included in diagonal which is the sum of all off-diagonal elements.
    :return: numpy array of admittance matrices converted to the desired format
    """
    from OpenDSSWrapper.converter.admittance import get_y_matrices_per_frequency
    base_kvs_dict = grid.get_base_kvs()
    base_kvs = list(base_kvs_dict.values())
    y_mats = get_y_matrices_per_frequency(grid.pp_net, voltage_level_kv, base_kvs, frequencies,
                                          shunt_position=shunt_position, compact=compact)
    # convert to numpy
    y_mats_np = np.array([np.array(value) for key, value in y_mats.items()])
    return y_mats_np


def apply_transformations(data: pd.DataFrame, grid: BaseGrid, network_data: dict, reference_rotate: int|None,
                          unify_voltage_level: bool, voltage_level_kv: float, variance_select: float = None,
                          fix_phase_angle_triplen=False, degrees=True):
    """
    Apply data transformations
    - rotate voltage angle to reference node (if node is provided)
    - unify voltage level to voltage_level_kv (if set to true)
    - to polar coordinates
    - mask data (to mask provided in network_data)
    - to n-dimensional numpy array
    :param data: as multi-indexed pandas dataframe
    :param grid: instance of BaseGrid
    :param network_data:
    :param reference_rotate:
    :param unify_voltage_level:
    :param voltage_level_kv:
    :param variance_select: ratio of highest variance data to be selected, default None
    :param fix_phase_angle_triplen: fix the issue of triplen harmonics phase angle being random since
    magnitude is likely zero if delta-wye transformers are present
    :param degrees: Shall the angle be interpretated as degrees (True) or radiant (False)
    :return:
    """
    from OpenDSSWrapper.simulation.store import DataStore
    from OpenDSSWrapper.simulation.post_processing import PostProcessor

    is_3ph_data = len(data.columns) == 6

    if reference_rotate is not None:
        angle_columns = ['vangle1', 'vangle2', 'vangle3'] if is_3ph_data else ['vangle1']
        data = rotate_angles_to_reference_node(data, reference_rotate, columns=angle_columns)
    if unify_voltage_level:
        post_processor = PostProcessor(grid.pp_net, grid.get_base_kvs(), voltage_level_kv)
        data = post_processor.unify_voltage_level(data)
    if variance_select:
        data = get_highest_variance_data(data, variance_select)
    data_format = network_data['data_format']
    data_np, desc_tuple = DataStore.df_to_numpy(data, data_format=data_format)
    if data_format == "channels_last":
        mask_axis = 1
        harmonic_axis = 2
        complex_component_axis = 3
        phase_data_axis = 4 if is_3ph_data else 3
    elif data_format == "channels_first":
        harmonic_axis = 1
        complex_component_axis = 2
        mask_axis = 4 if is_3ph_data else 3
        phase_data_axis = 3 if is_3ph_data else 2
    else:
        raise ValueError("data_format must be either channels_last or channels_first")

    modes = network_data['split_complex_modes']
    input_converter = SplitComplexConverter(modes, target_axis=phase_data_axis)
    target_converter = SplitComplexConverter(modes[0], target_axis=phase_data_axis)
    # convert numpy array to complex numbers
    mag_indices = [0, 2, 4] if is_3ph_data else 0
    phase_indices = [1, 3, 5] if is_3ph_data else 1
    complex_arr = polar_to_complex(data_np, mag_index=mag_indices, phase_index=phase_indices, target_axis=complex_component_axis,
                                   degrees=degrees)
    if fix_phase_angle_triplen:
        complex_arr = rotate_unmeasurable_angles(complex_arr, harmonic_axis)
    # data_np = polar_to_cartesian(data_np, axis=complex_component_axis, mag_index=(0,), phase_index=(1,),
    #                              angle_type='degrees')
    x = input_converter.convert(complex_arr)
    x = mask_data(x, network_data['mask_x_test'], mask_axis)
    if 'mask_y_test_meas' in network_data.keys():
        y = target_converter.convert(complex_arr)
        y = mask_data(y, network_data['mask_y_test_meas'], mask_axis)
    else:
        y = target_converter.convert(complex_arr)
    # add singelton dimension if 1phase data is used
    if not is_3ph_data:
        x = np.expand_dims(x, axis=-2)
        y = np.expand_dims(y, axis=-2)
    return x, y, desc_tuple


def get_highest_variance_data(df: pd.DataFrame, variance_select=1 / 15):
    """
    Select the data with the highest variance
    This should be done after scaling (or at least transforming data to the same range / voltage level)
    :param df: multi-indexed dataframe
    :param variance_select: percentage to select
    :return: dataframe with the highest variance data
    """
    # Calculate variance for each step by aggregating across other indices
    step_variance = df.groupby('step').var().mean(axis=1)

    # Select the top 10% steps based on variance
    top_x_percent_threshold = step_variance.quantile(1 - variance_select)
    top_x_percent_steps = step_variance[step_variance > top_x_percent_threshold]
    top_variance_data = df.loc[df.index.get_level_values('step').isin(top_x_percent_steps.index)]
    return top_variance_data


def remove_steps_with_outliers(arr:np.ndarray, factor=1.5) -> np.ndarray:
    """
    Remove steps with outliers based on the IQR method
    :param arr: numpy array of complex datatype with shape (steps, nodes, harmonics)
    :param factor: factor to multiply the IQR with to determine the bounds for outliers
    :return: filtered numpy array
    """
    # Calculate magnitudes
    magnitudes = np.abs(arr)

    # Determine the bounds for outliers based on IQR
    Q1 = np.percentile(magnitudes, 25, axis=(1, 2), keepdims=True)
    Q3 = np.percentile(magnitudes, 75, axis=(1, 2), keepdims=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Detect outliers
    outlier_mask = (magnitudes < lower_bound) | (magnitudes > upper_bound)

    # Identify steps to remove: any step with at least one outlier
    steps_to_remove = np.any(outlier_mask, axis=(1, 2))

    # Filter the array to remove these steps
    filtered_array = arr[~steps_to_remove]

    return filtered_array

def read_postgresql_db(db_config: dict, network_data: dict, grid: BaseGrid, separate_datasets: bool = False,
                       reference_rotate: int = None, columns: list = None, unify_voltage_level: bool = False,
                       voltage_level_kv: float = 0.4, frequencies: list = None, fix_unmeasurable_angles=False,
                       use_lab_data_test=False, lab_data_path=None):
    """
    Read data from a postgres database and preprocess it
    :param db_config: database configuration
    :param network_data: dictionary with information about train test splits, ...
    :param grid: instance of BaseGrid
    :param separate_datasets: if True, the postgres database contains separate tables for train, test and validation
    datasets, else the split is performed from the same set
    :param reference_rotate: Reference node to rotate relative to, else None -> no rotation
    :param columns: list of columns, default ['v1', 'vangle1']
    :param unify_voltage_level: True if voltage level shall be unified to voltage_level_kv
    :param voltage_level_kv: voltage level in kV to unify the data to. Must be provided even if unify_voltage_level is false for the admittance matrix
    :param frequencies: list of frequencies to query for
    :param fix_unmeasurable_angles: if True adjusts the angle of unmeasurable angles in triplen harmonics to the most common angle for a more beneficial data distribution
    :use_lab_data_test: if True, use lab test data instead of the sql-db test dataset
    :lab_data_path: path to lab test data .csv
    :return:
    """
    from OpenDSSWrapper.simulation.store import PostgreSQLStore
    if not columns:
        columns = ['v1', 'vangle1']
    if separate_datasets:
        train_store = PostgreSQLStore('data', columns, db_config,
                                      table_name=f'harmonics_{network_data["base_table_name"].lower()}_train')
        data_train = train_store.read_data(steps=list(np.arange(0, 100000, 1)), columns=columns, element_types=[0],
                                           frequencies=frequencies)
        x_train, y_train, desc_tuple = apply_transformations(data_train, grid,
                                                               network_data,
                                                               reference_rotate,
                                                               unify_voltage_level,
                                                               voltage_level_kv,
                                                               variance_select=None,
                                                               fix_phase_angle_triplen=fix_unmeasurable_angles)
        network_data['x_train'] = x_train
        network_data['y_train'] = y_train
        validation_store = PostgreSQLStore('data', columns, db_config,
                                           table_name=f'harmonics_{network_data["base_table_name"].lower()}_validation')
        data_val = validation_store.read_data(steps=list(np.arange(0, 10000, 1)), columns=columns,
                                              element_types=[0], frequencies=frequencies)
        x_val, y_val, _ = apply_transformations(data_val, grid, network_data,
                                                           reference_rotate,
                                                           unify_voltage_level,
                                                           voltage_level_kv,
                                                           fix_phase_angle_triplen=fix_unmeasurable_angles)
        network_data['x_val'] = x_val
        network_data['y_val'] = y_val
        # x_train0, x_val0, y_train0, y_val0 = skms.train_test_split(x_random, y_random,
        #                                                            test_size=0.3, random_state=1, shuffle=False)
        # x_train1, x_val1, y_train1, y_val1 = skms.train_test_split(x_realistic_spectrum, y_realistic_spectrum,
        #                                                            test_size=0.3, random_state=1, shuffle=False)
        # network_data['x_train'] = np.concatenate((x_train0, x_train1), axis=0)
        # network_data['x_val'] = np.concatenate((x_val0, x_val1), axis=0)
        # network_data['y_train'] = np.concatenate((y_train0, y_train1), axis=0)
        # network_data['y_val'] = np.concatenate((y_val0, y_val1), axis=0)
        if use_lab_data_test and lab_data_path:
            # ## Test data from actual grid - fit measurement device indices for other topologies:
            buffer_mask_x_test = network_data['mask_x_test']
            network_data['mask_x_test'] = [0, 5]  # equals [0, 7] at 12 measured nodes
            network_data['mask_y_test_meas'] = [0, 3, 4, 5]  # equals [0, 3, 6, 7] at 12 measured nodes
            x_test_cart, y_test_cart, data_desc = read_microgrid_data(lab_data_path, grid, network_data, frequencies)
            network_data['mask_x_test'] = buffer_mask_x_test
            # write to data dictionary and apply necessary transformations
            network_data['x_test'] = x_test_cart  # x_test_meas
            network_data['y_test'] = y_test_cart  # y_test_meas
        else:
            test_store = PostgreSQLStore('data', columns, db_config,
                                         table_name=f'harmonics_{network_data["base_table_name"].lower()}_test')
            data_test = test_store.read_data(steps=list(np.arange(0, 3504, 1)), columns=columns, element_types=[0],
                                             frequencies=frequencies)
            network_data['x_test'], network_data['y_test'], _ = apply_transformations(data_test, grid, network_data,
                                                                                      reference_rotate,
                                                                                      unify_voltage_level,
                                                                                      voltage_level_kv,
                                                                                      fix_phase_angle_triplen=fix_unmeasurable_angles)

    else:
        store = PostgreSQLStore('data', columns, db_config, table_name=f'harmonics_{network_data["base_table_name"].lower()}')
        data = store.read_data(steps=list(np.arange(0, 525600, 15)), columns=columns, element_types=[0])
        x, y, desc_tuple = apply_transformations(data, grid, network_data, reference_rotate, unify_voltage_level,
                                                 voltage_level_kv, fix_phase_angle_triplen=fix_unmeasurable_angles)
        network_data['x_train'], network_data['x_test'], network_data['y_train'], network_data[
            'y_test'] = skms.train_test_split(x, y,
                                              test_size=network_data['test_size'], random_state=1, shuffle=False)
        network_data['x_train'], network_data['x_val'], network_data['y_train'], network_data[
            'y_val'] = skms.train_test_split(network_data['x_train'], network_data['y_train'],
                                             test_size=network_data['validation_size'], random_state=1, shuffle=False)
    # data_bus = DataStore.get_bus_data(data, phases=['1'])
    # convert to numpy
    time_steps, elements, frequencies, columns = desc_tuple
    network_data['Y_matrices_per_frequency'] = get_y_mats(grid, frequencies, voltage_level_kv=voltage_level_kv,
                                                          data_format=network_data['data_format'],
                                                          num_phases=int(len(columns)/2))

    network_data['input_dims'] = network_data['x_train'].shape[1:]
    network_data['output_dims'] = network_data['y_train'].shape[1:]
    network_data['frequencies'] = frequencies
    scaling_axes = (0, 1) if network_data['data_format'] == 'channels_last' else (0, 3)
    network_data_scaled = scale_data(network_data, axes=scaling_axes, sep_output_scaler=True)
    return network_data_scaled


def read_microgrid_data(data_path, grid, network_data, frequencies=None, swap_sequence_buses=None):
    """
    Read "live" data of microgrid from csv
    :param data_path: path to data directory
    :return: x_test_cart, y_test_cart
    """

    df_filtered = read_lab_data(os.path.join(data_path, 'thkoelnnetwork'),'SE_Results_9-9-2025.csv',
                                frequencies, swap_sequence_buses)

    x_test, y_test, data_desc = apply_transformations(df_filtered, grid, network_data, reference_rotate=None,
                                                      unify_voltage_level=False,
                                                      voltage_level_kv=0.4,
                                                      fix_phase_angle_triplen=False,
                                                      degrees=False)
    return x_test, y_test, data_desc


if __name__ == '__main__':
    # save the admittance matrix as pickle file
    from OpenDSSWrapper.converter.grids import get_y_matrices_per_frequency
    from OpenDSSWrapper.converter.grids import CigreLVDistPV

    grid = CigreLVDistPV()
    frequencies = list(np.arange(50, 1050, 50))
    base_kvs_dict = grid.get_base_kvs()
    base_kvs = list(base_kvs_dict.values())
    y_mats = get_y_matrices_per_frequency(grid.pp_net, 20., base_kvs, frequencies, shunt_position=-1,
                                          compact=True)
    with open('../../pqse_concept_pann/data/y_mats_per_frequency.pic', 'wb') as pic:
        pickle.dump(y_mats, pic)
