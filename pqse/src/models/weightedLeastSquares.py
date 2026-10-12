import math
from random import Random
import numpy as np
import pandas as pd
import pickle as pi
import os
from OpenDSSWrapper.converter.grids import CigreLVDistPV,IEEE33TestFeeder
from OpenDSSWrapper.converter import components, admittance
from evaluation import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime, timedelta


def calculate_nrmse_per_frequency(data_true, data_hse, orders_to_evaluate, min_max_ref=False, exclude_mv_buses=None,
                                  exclude_measure_nodes=None):
    list_return_nrmse_per_freq = []
    if exclude_mv_buses is None:
        exclude_mv_buses = [-1]
    if exclude_measure_nodes is None:
        exclude_measure_nodes = [-1]
    for order in orders_to_evaluate:
        data_per_freq_true = data_true[:, :, order-1, :]
        data_per_freq_est = data_hse[:, :, order-1, :]
        # select bus by bus
        list_return_nrmse_per_bus = []
        for bus in range(0, len(data_per_freq_true[0])):
            if (bus in exclude_mv_buses) and order % 3 == 0:
                # 3rd order harmonics do not propagate into medium voltage level due to DYN-Transformers
                list_return_nrmse_per_bus.append(0)
            elif bus not in exclude_measure_nodes: # do not evaluate measured nodes
                all_bus_values_true = data_per_freq_true[:, bus, :]
                all_bus_values_est = data_per_freq_est[:, bus, :]
                all_bus_values_true = [math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_true]
                all_bus_values_est = [math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_est]
                ref_on = 0
                if min_max_ref:
                    ref_on = np.max(all_bus_values_true) - np.min(all_bus_values_true)
                else:
                    ref_on = np.mean(all_bus_values_true)
                deltas = [all_bus_values_true[i] - all_bus_values_est[i] for i in range(len(all_bus_values_true))]
                rmse = np.sqrt(np.mean(np.square(deltas)))
                list_return_nrmse_per_bus.append(rmse/ref_on)
        list_return_nrmse_per_freq.append(list_return_nrmse_per_bus)
    return list_return_nrmse_per_freq


def calculate_mse_rmse_per_frequency(data_true, data_hse, orders_to_evaluate, voltage_base,
                                     use_mse=True, use_pu=False, use_rel_error=False, other_ref_v: float = 1,
                                     exclude_buses=None, mv_buses=None):
    list_return_nrmse_per_freq = []

    scaling_per_bus = [other_ref_v*1/(1000*voltage_base[v]/math.sqrt(3)) if use_pu else 1 for v in voltage_base]

    if exclude_buses is None:
        exclude_buses = [0, 12, 28]
    if mv_buses is None:
        mv_buses = [-1]
    for order in orders_to_evaluate:
        data_per_freq_true = data_true[:, :, order-1, :]
        data_per_freq_est = data_hse[:, :, order-1, :]
        # select bus by bus
        list_return_nrmse_per_bus = []
        deltas_per_order = []
        for bus in range(0, len(data_per_freq_true[0])):
            if bus not in exclude_buses:
                scaler = scaling_per_bus[bus]

                all_bus_values_true = data_per_freq_true[:, bus, :]
                all_bus_values_est = data_per_freq_est[:, bus, :]

                if use_rel_error:
                    all_bus_values_true = [scaler * math.sqrt(d[0] ** 2 + d[1] ** 2) for d in all_bus_values_true]
                    all_bus_values_est = [scaler * math.sqrt(d[0] ** 2 + d[1] ** 2) for d in all_bus_values_est]
                    delta_per_bus = []
                    for x in range(0, len(all_bus_values_true)):
                        if abs(all_bus_values_true[x]) > 0.0000001:
                            delta_per_bus.append((all_bus_values_true[x] - all_bus_values_est[x])
                                                 / all_bus_values_true[x])

                    deltas_per_order.extend(delta_per_bus)
                else:
                    all_bus_values_true = [scaler * math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_true]
                    all_bus_values_est = [scaler * math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_est]
                    delta_per_bus = [(all_bus_values_true[i] - all_bus_values_est[i])**2 for i in range(len(all_bus_values_true))]
                    deltas_per_order.extend(delta_per_bus)
                # print('Bus ' + str(bus) + ': ' + str(np.mean(np.square(delta_per_bus))))
        squared_errors = np.zeros_like(deltas_per_order)
        if use_rel_error:
            squared_errors = np.abs(deltas_per_order)
        else:
            squared_errors = np.square(deltas_per_order)
        mse = np.mean(squared_errors)
        add_to_list = mse if use_mse else np.sqrt(mse)
        list_return_nrmse_per_freq.append(add_to_list)
    return list_return_nrmse_per_freq


def calculate_thd(data_true, data_hse, orders_to_evaluate):
    thd_per_bus_true = np.zeros((len(data_true), len(data_true[0])))
    thd_per_bus_est = np.zeros((len(data_true), len(data_true[0])))
    # calculte thd for every state and and bus
    for state in range(len(data_true)):

        for bus in range(0, 44):
            thd_denom_u_sqared_true = 0
            thd_denom_u_sqared_est = 0
            fundamental_squared_true = math.sqrt(
                data_true[state, bus, 0, 0] ** 2 + data_true[state, bus, 0, 1] ** 2)
            fundamental_squared_est = math.sqrt(data_hse[state, bus, 0, 0] ** 2 + data_hse[state, bus, 0, 1] ** 2)
            for o in orders_to_evaluate:
                val_true = math.sqrt(data_true[state, bus, o - 1, 0] ** 2 + data_true[state, bus, o - 1, 1] ** 2)
                thd_denom_u_sqared_true += (val_true ** 2)
                val_est = math.sqrt(data_hse[state, bus, o - 1, 0] ** 2 + data_hse[state, bus, o - 1, 1] ** 2)
                thd_denom_u_sqared_est += (val_est ** 2)
            thd_per_bus_true[state, bus] = 100 * math.sqrt(thd_denom_u_sqared_true) / fundamental_squared_true
            thd_per_bus_est[state, bus] = 100 * math.sqrt(thd_denom_u_sqared_est) / fundamental_squared_est
    return thd_per_bus_true, thd_per_bus_est


def plot_thd_bus(data_true, data_hse, orders_to_evaluate, show_busses):
    thd_per_bus_true, thd_per_bus_est = calculate_thd(data_true, data_hse, orders_to_evaluate)
    start_day = 4
    end_day = 5
    # configure plot
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Create subplots
    fig, axs = plt.subplots(len(show_busses), 1, figsize=(10, 8), sharex=True)
    time_of_day_minutes = range(4*24*(start_day-1), 4*24*(end_day-1))  # time of day from 0 to 24 hours
    start_time = datetime(2024, 1, 1)
    time_of_day = [start_time + timedelta(minutes=tm*15) for tm in time_of_day_minutes]# time of day from 0 to 24 hours
    for i, bus in enumerate(show_busses):

        thd_data_true = thd_per_bus_true[time_of_day_minutes, bus]
        thd_data_est = thd_per_bus_est[time_of_day_minutes, bus]

        # Plotting the data for each subplot
        axs[i].plot(time_of_day, thd_data_true, label='True')
        axs[i].plot(time_of_day, thd_data_est, label='Est.')
        axs[i].set_ylabel('$THD_{V} [\\%]$')
        axs[i].legend(loc='upper left')
        axs[i].set_title('Busbar ' + str(bus))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator())
        axs[i].set_ylim(ymin=0.0)

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # Set x-axis limits
    min_time = start_time + timedelta(minutes=(start_day-1) * 24 * 60)  # start time
    max_time = start_time + timedelta(minutes=(end_day-1) * 24 * 60 - 1)  # end time

    for ax in axs:
        ax.set_xlim(min_time, max_time)

    plt.xticks(rotation=45)
    # Common x-axis label
    plt.xlabel('time of day [HH:MM]')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('THD_busses.png')
    # Show plot
    plt.show()
    pass


def calculate_nrmse_vthd(data_true, data_hse, orders_to_evaluate, min_max_ref=False):
    thd_per_bus_true, thd_per_bus_est = calculate_thd(data_true, data_hse, orders_to_evaluate)
    # select bus by bus
    list_return_nrmse_per_bus = []
    for bus in range(0, 44):
        all_bus_values_true = thd_per_bus_true[:, bus]
        all_bus_values_est = thd_per_bus_est[:, bus]
        ref_on = 0
        if min_max_ref:
            ref_on = np.max(all_bus_values_true) - np.min(all_bus_values_true)
        else:
            ref_on = np.mean(all_bus_values_true)

        deltas = [all_bus_values_true[i] - all_bus_values_est[i] for i in range(len(all_bus_values_true))]
        rmse = np.sqrt(np.mean(np.square(deltas)))
        list_return_nrmse_per_bus.append(rmse / ref_on)

    return list_return_nrmse_per_bus


def import_hse_pann_data(path_test_data, only_import_sim_data=False):

    p_data_x = os.path.join(path_test_data, 'PANN_Test_x.pic')
    p_data_y = os.path.join(path_test_data, 'PANN_Test_y.pic')
    data_x = pd.read_pickle(p_data_x)
    data_y = pd.read_pickle(p_data_y)
    if only_import_sim_data:
        return data_x
    orders_to_evaluate_thd = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # plot_thd_bus(data_x, data_y, orders_to_evaluate_thd, [16, 25, 37])
    # nmrse_thd = calculate_nrmse_vthd(data_x, data_y, orders_to_evaluate_thd)
    orders_to_evaluate = [1, 3, 5, 7]
    grid_scalers = IEEE33TestFeeder().get_base_kvs()
    mse_results = calculate_mse_rmse_per_frequency(data_x, data_y, orders_to_evaluate, grid_scalers,
                                                   True, True, True)
    for order in range(len(orders_to_evaluate)):
         print(f'maximum RMSE/MSE on order {orders_to_evaluate[order]}: {mse_results[order]}')

    nrmse_results = calculate_nrmse_per_frequency(data_x, data_y, orders_to_evaluate, True)
    for order in range(len(orders_to_evaluate)):
        print(f'maximum nRMSE on order {orders_to_evaluate[order]}: {max(nrmse_results[order])}')

def import_and_compare(base_csv_filepath, start_idx_l, end_idx_l, step_width_l,
                        scale_voltages, known_busses, pred_busses):

    for i in range(start_idx_l, end_idx_l, step_width_l):

        df_orig = pd.read_csv(os.path.join(base_csv_filepath, str(i), f"CIGRELVDist_train_step_{i}.csv"), index_col=0)
        df_est = pd.read_csv(os.path.join(base_csv_filepath, str(i), f"CIGRELVDist_WLS_step_{i}.csv"), index_col=0)
        for f_l in range(50, 1050, 50):
            x_est = []
            y_orig = []
            y_est = []
            harmonic_order = f_l/50
            for b in range(0, 44):
                criterion1 = df_orig['element_type'] == 0
                criterion2 = df_orig['element'] == 'bus' + str(b)
                criterion3 = df_orig['freq'] == f_l

                filtered_df_orig = df_orig[criterion1 & criterion2 & criterion3]
                mag_orig = filtered_df_orig['v1'].to_numpy()[0] * np.absolute(scale_voltages[b])
                ang_orig = filtered_df_orig['vangle1'].to_numpy()[0]
                y_orig.append(mag_orig)

                criterion1 = df_est['element_type'] == 0
                criterion2 = df_est['element'] == 'bus' + str(b)
                criterion3 = df_est['freq'] == f_l
                filtered_df_est = df_est[criterion1 & criterion2 & criterion3]
                mag_est = filtered_df_est['v1'].to_numpy()[0] * np.absolute(scale_voltages[b])
                ang_est = filtered_df_est['vangle1'].to_numpy()[0]

                if b in known_busses:
                    x_est.append(mag_est)
                else:
                    y_est.append(mag_est)

            plot.plot_estimations(y_orig, x_est, {'WLS': y_est},
                              f'WLS estimations Order {int(harmonic_order)}', bus_numbers_pred=pred_busses)


def scale_and_reshape_data(data, start_idx: int, end_idx: int, step_width: int, rotate_voltages: [complex]):
    data_pos = 0
    result = data[start_idx:end_idx:step_width, :, :, :]  # dim (t_step, harm, real/imag, bus)
    real_part = result[:, :, 0, :]
    imag_part = result[:, :, 1, :]
    complex_array = real_part + 1j * imag_part

    harmonic_rotations = []
    for h in range(1, complex_array.shape[1] + 1):
        harmonic_rotations.append(([1/(i**h) for i in rotate_voltages]))
    rotaded_results = complex_array[:] * np.array(harmonic_rotations)
    return rotaded_results


def load_hse_results_from_csv(base_csv_filepath, start_idx: int, end_idx: int, step_width: int, scale_voltages: [float]):
    # dimension of data: [time step] [no of busses] [harmonics per reading]
    data_length = int((end_idx-start_idx)/step_width)
    data = np.empty([data_length, 44, 20], dtype=np.complex128)
    data_pos = 0
    for i in range(start_idx, end_idx, step_width):

        df = pd.read_csv(os.path.join(base_csv_filepath, str(i), f"CIGRELVDist_train_step_{i}.csv"), index_col=0)
        criterion1 = df['element_type'] == 0
        for b in range(0, 44):
            criterion2 = df['element'] == 'bus' + str(b)
            filtered_df = df[criterion1 & criterion2]
            mag = filtered_df['v1'].to_numpy()
            ang = filtered_df['vangle1'].to_numpy()
            # Calculate the real and imaginary parts
            real_parts = mag * np.cos(ang * np.pi / 180)
            imaginary_parts = mag * np.sin(ang * np.pi / 180)
            # Combine the real and imaginary parts to form complex numbers
            complex_numbers = real_parts + 1j * imaginary_parts
            data[data_pos, b, :] = complex_numbers * scale_voltages[b]
        data_pos += 1

    return data


def get_connections_path(start_node, end_node, adjacency):
    found_connections = []
    for con in range(len(adjacency)):
        if abs(adjacency[start_node, con]) > 0 and con != end_node:
            if start_node < con:
                found_connections.append((start_node, con))
            else:
                found_connections.append((con, start_node))

    return len(found_connections), found_connections


def save_hse_results_to_csv(base_csv_filepath, df_step, idx_step):
    df_step.to_csv(os.path.join(base_csv_filepath, str(idx_step), f"CIGRELVDist_WLS_step_{idx_step}.csv"))


def get_network_currents(v_data, y_bus_all_freq, frequencies_all):
    dim_matrix = len(y_bus_all_freq[frequencies_all[0]])
    currents = np.empty((v_data.shape[0], len(frequencies_all), dim_matrix, dim_matrix), dtype=np.complex128)
    for f in range(0, len(frequencies_all)):
        y_bus_at_freq_l = y_bus_all_freq[frequencies_all[f]]
        for d in range(0, v_data.shape[0]):
            for from_bus_l in range(dim_matrix):
                for to_bus_l in range(dim_matrix):
                    # This is a bus-connection admittance
                    if y_bus_at_freq_l[from_bus_l][to_bus_l] != complex(0, 0):
                        currents[d, f, from_bus_l, to_bus_l] = (y_bus_at_freq_l[from_bus_l][to_bus_l] *
                                                                (v_data[d, f, from_bus_l] - v_data[d, f, to_bus_l]))

    return currents


def get_power_inj(currents, mask_curr_inj_l, frequencies_all, y_bus_all_freq):

    current_injections = np.empty((currents.shape[0], len(frequencies_all), len(mask_curr_inj_l)), dtype=np.complex128)
    for f in range(0, len(frequencies_all)):
        y_bus_at_freq_l = y_bus_all_freq[frequencies_all[f]]
        for d in range(0, currents.shape[0]):
            for idx_inj in range(len(mask_curr_inj_l)):
                bus_inj = mask_curr_inj_l[idx_inj]

                for i in range(len(y_bus_at_freq_l)):
                    if abs(y_bus_at_freq_l[bus_inj][i]) > 0:
                        print(f"Current from {bus_inj} to {i} is {currents[d, f, bus_inj, i]}")
                        current_injections[d, f, idx_inj] += currents[d, f, bus_inj, i]

                print(f'Injection is: {current_injections[d, f, idx_inj]}')
    return current_injections


def get_current_state_idx(from_bus_l, to_bus_l):
    if idx_matr[from_bus_l, to_bus_l] == 0:
        return idx_matr[to_bus_l, from_bus_l]
    return idx_matr[from_bus_l, to_bus_l]


def count_entries(row):
    count = 0
    pos = []
    for i in range(len(row)):
        if abs(row[i]) > 1E-5:
            count += 1
            pos.append(i)
    return count, pos


# import_hse_pann_data()

base_path =  os.path.join('C:\\', 'Users', 'Patrick Mack', 'sciebo', '99_Transfer', 'Grafiken_Paper', 'Daten', 'CIGRE_LV_DIST')
n_busses = 44
idx_matr = np.matrix(np.zeros((n_busses, n_busses), dtype=np.int32))
idx_matr[0, 1] = 0; idx_matr[0, 20] = 1; idx_matr[0, 23] = 2; idx_matr[1, 2] = 3; idx_matr[2, 3] = 4; idx_matr[3, 4] = 5
idx_matr[4, 5] = 6; idx_matr[5, 6] = 7; idx_matr[6, 7] = 8; idx_matr[7, 8] = 9; idx_matr[8, 9] = 10; idx_matr[9, 10] = 11
idx_matr[10, 11] = 12; idx_matr[11, 19] = 13; idx_matr[4, 12] = 14; idx_matr[5, 13] = 15; idx_matr[13, 14] = 16; idx_matr[14, 15] = 17
idx_matr[15, 16] = 18; idx_matr[7, 17] = 19; idx_matr[10, 18] = 20; idx_matr[20, 21] = 21; idx_matr[21, 22] = 22; idx_matr[23, 24] = 23
idx_matr[24, 25] = 24; idx_matr[25, 26] = 25; idx_matr[26, 27] = 26; idx_matr[27, 28] = 27; idx_matr[28, 29] = 28; idx_matr[29, 30] = 29
idx_matr[30, 31] = 30; idx_matr[31, 32] = 31; idx_matr[32, 43] = 32; idx_matr[26, 33] = 33; idx_matr[33, 34] = 34; idx_matr[33, 37] = 35
idx_matr[34, 36] = 36; idx_matr[34, 35] = 37; idx_matr[28, 38] = 38; idx_matr[38, 41] = 39; idx_matr[38, 39] = 40; idx_matr[39, 40] = 41
idx_matr[31, 42] = 42

ref_kv = 20
init_val = 1000 * 20/math.sqrt(3)
rd = Random()
phase_shift_transformer = -30 * (np.pi/180)
lv_kv_complex = complex(0.4 * np.cos(phase_shift_transformer), 0.4 * np.sin(phase_shift_transformer))
transformer_shift_complex = complex(np.cos(phase_shift_transformer), np.sin(phase_shift_transformer))
mv_kv_complex = complex(ref_kv, 0)
kv_levels_complex = [lv_kv_complex for b in range(0, 44)]
angles_complex = [transformer_shift_complex for b in range(0, 44)]
kv_levels_complex[0] = mv_kv_complex
kv_levels_complex[1] = mv_kv_complex
kv_levels_complex[20] = mv_kv_complex
kv_levels_complex[23] = mv_kv_complex
angles_complex[0] = complex(1,0)
angles_complex[1] = complex(1,0)
angles_complex[20] = complex(1,0)
angles_complex[23] = complex(1,0)
kv_levels = [abs(i) for i in kv_levels_complex]


kv_scaler_complex = [mv_kv_complex / i for i in kv_levels_complex]
# load data
start_idx = 0
end_idx = 60
step_width = 15
len_data = int((end_idx - start_idx) / step_width)
mask_lv_busses = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
# mask_busses_z_v = [0, 2, 4, 6, 7, 8, 10, 11, 13, 15, 21, 24, 26, 28, 29, 30, 31, 32, 34, 37, 38, 39]
# mask_busses_z_c = [(0, 1),(2, 3), (0, 20), (0, 23),(4, 5), (4, 12), (6, 7), (8, 9), (10, 18), (13,14), (15, 16), (7, 17), (11, 19),
#                    (20, 21), (21, 22), (23, 24), (24, 25), (26, 33), (27, 28), (30, 31), (32, 43), (34, 35), (33, 37),
#                    (39, 40), (38, 41), (31, 42)]
#
# mask_busses_x_v = [1, 3, 5, 9, 12, 14, 16, 17, 18, 19, 20, 22, 23, 25, 27, 33, 35, 36, 40, 41, 42, 43]
# mask_busses_x_c = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (10, 11), (5, 13), (14, 15),
#                    (25, 26), (26, 27), (28, 29), (28, 38), (29, 30), (31, 32), (38, 39), (33, 34), (34, 36)]

mask_busses_z_v = [1, 3, 6, 9, 12, 13, 16, 17, 18, 19, 20, 22, 23, 25, 27, 28, 30, 32, 35, 36, 37, 40, 41, 42]
mask_busses_z_c = [(1, 2), (3, 4), (4, 12), (5, 6), (7, 17), (9, 10), (10, 18), (11, 19), (13, 14), (14, 15), (15, 16), (20, 21),
                   (21, 22), (23, 24), (24, 25), (26, 27), (29, 30), (31, 42), (32, 43), (33, 37), (34, 35), (34, 36),
                   (38, 40), (39, 41)]

mask_busses_x_v = [0, 2, 4, 5, 7, 8, 10, 11, 14, 15, 21, 24, 26, 29, 31, 33, 38, 34, 39, 43]
mask_busses_x_c = [(0, 1), (0, 20), (0, 23), (2, 3), (4, 5), (5, 13), (6, 7), (7, 8), (8, 9), (10, 11),
                   (25, 26), (26, 33), (27, 28), (28, 29), (28, 38), (30, 31), (31, 32), (33, 34), (38, 39)]
mask_harmonics = [0]

all_curr_inj = [0, 2, 9, 12, 16, 17, 18, 19, 22, 24, 32, 35, 36, 37, 40, 41, 42, 43]
mask_curr_inj = [0, 2, 9, 32, 37, 40, 41]

# import_and_compare(base_path, start_idx, end_idx, step_width, kv_scaler_complex, mask_busses_z_v, mask_busses_x_v)

# data = load_hse_results_from_csv(base_path, start_idx=start_idx, end_idx=end_idx, step_width=step_width, scale_voltages=kv_scaler_complex)
data = import_hse_pann_data(base_path, True)
data = scale_and_reshape_data(data, start_idx=start_idx, end_idx=end_idx, step_width=step_width,rotate_voltages=angles_complex)
# Sample data
z_v = data[:, :, mask_busses_z_v]  # m x 1 measurement vector
x_v = data[:, :, mask_busses_x_v]  # n x 1 state vector

frequencies = [50*o for o in range(1, 21)]
y_bus = admittance.get_y_matrices_per_frequency(CigreLVDistPV().pp_net,
                                                20, kv_levels, frequencies, None)  # Get YBus..

# Calculate currents:
data_curr = get_network_currents(data, y_bus, frequencies)
data_inj = get_power_inj(data_curr, mask_curr_inj, frequencies, y_bus)
z_c = data_curr[:, :, [n[0] for n in mask_busses_z_c], [n[1] for n in mask_busses_z_c]]  # m x 1 measurement vector
x_c = data_curr[:, :, [n[0] for n in mask_busses_x_c], [n[1] for n in mask_busses_x_c]]  # n x 1 state vector

# Get Measurement data..
len_measure_vector_v = len(mask_busses_z_v)
len_state_vector_v = len(mask_busses_x_v)
len_measure_vector_c = len(mask_busses_z_c)
len_state_vector_c = len(mask_busses_x_c)
n_bus = len(y_bus[50])  # Get number of buses..
Ri = np.diagflat([1 for i in range(len_measure_vector_v+len_measure_vector_c)])  # Measurement Error..
inits = [init_val, 0.02 * init_val, 0.02 * init_val, 0.05 * init_val, 0.03 * init_val, 0.04 * init_val, 0.02 * init_val,
         0.07 * init_val, 0.02 * init_val, 0.05 * init_val, 0.02 * init_val, 0.04 * init_val, 0.02 * init_val,
         0.05 * init_val, 0.02 * init_val, 0.05 * init_val, 0.02 * init_val, 0.05 * init_val, 0.01 * init_val,
         0.01 * init_val]

evaluate_H = False
evaluate_V_I = True

for d in range(0, len_data):
    df = pd.DataFrame(columns=['bus', 'element', 'element_type', 'freq', 'v1', 'vangle1'])
    list_r = []
    for f, v_f in enumerate(frequencies):
        converge_list = [i for i in range(0, 10)]
        y_bus_at_freq = y_bus[v_f]
        V = np.full((len_state_vector_v, 1), inits[f], dtype=np.complex128)  # Initialize the bus voltages..
        I = np.zeros((len_state_vector_c, 1), dtype=np.complex128)  # Initialize the line currents..
        I[:, 0] = [rd.randint(0, 100)/10000 + rd.randint(0, 10)*1j/100000 for i in range(len_state_vector_c)]
        E = np.matrix(np.concatenate((V, I)))
        E_hist = np.zeros((len(E), 1), dtype=np.complex128)
        # The measurement vector Z...
        curr_z_v = np.transpose(np.asmatrix(z_v[d, f, :], dtype=np.complex128))
        curr_z_c = np.transpose(np.asmatrix(z_c[d, f, :], dtype=np.complex128))
        curr_z_inj = np.transpose(np.asmatrix(data_inj[d, f, :], dtype=np.complex128))

        Z = np.concatenate((curr_z_v, curr_z_c))

        tol = 5
        while np.sum(converge_list) > tol:
            # Measurement Function, h
            # Voltages
            h1 = np.zeros((len_measure_vector_v, 1), dtype=np.complex128)
            h2 = np.zeros((len_measure_vector_c, 1), dtype=np.complex128)
            H_v_c = np.mat(np.zeros((len_measure_vector_v, len_state_vector_c), dtype=np.complex128))
            H_v_v = np.mat(np.zeros((len_measure_vector_v, len_state_vector_v), dtype=np.complex128))
            H_c_c = np.mat(np.zeros((len_measure_vector_c, len_state_vector_c), dtype=np.complex128))
            H_c_v = np.mat(np.zeros((len_measure_vector_c, len_state_vector_v), dtype=np.complex128))

            # match states with measurements
            # Voltages
            for idx_z in range(0, len(h1)):

                # identify voltage measurement according to idx in Z
                bus_z = mask_busses_z_v[idx_z]
                # find dependecies via adjacency:
                for con_bus_z in range(len(y_bus_at_freq[bus_z])):
                    if abs(y_bus_at_freq[bus_z][con_bus_z]) > 1E-10:
                        connection_count, connections = get_connections_path(bus_z, -1, y_bus_at_freq)
                        # Strom bekannt
                        if (bus_z, con_bus_z) in mask_busses_z_c or (con_bus_z, bus_z) in mask_busses_z_c:
                            idx_c = mask_busses_z_c.index((bus_z, con_bus_z)) if bus_z < con_bus_z else (
                                mask_busses_z_c.index((con_bus_z, bus_z)))
                            # Logik für Stromrichtung
                            sign = 1 if bus_z < con_bus_z else -1

                            # Spannung am anderen Ende der Leitung auch bekannt
                            if con_bus_z in mask_busses_z_v:
                                idx_v_meas = mask_busses_z_v.index(con_bus_z)
                                h1[idx_z] = curr_z_v[idx_v_meas] + sign * curr_z_c[idx_c] / y_bus_at_freq[bus_z][con_bus_z]
                                # dV/dv
                                # H_v_v[idx_z] = 0 + 0j
                                # dV/di
                                H_v_c[idx_z, idx_c] = sign / y_bus_at_freq[bus_z][con_bus_z]

                            # Spannung am Ende der Leitung unbekannt, Strom bekannt
                            elif con_bus_z in mask_busses_x_v:
                                idx_v_z = mask_busses_x_v.index(con_bus_z)
                                h1[idx_z] = V[idx_v_z] + sign * curr_z_c[idx_c] / y_bus_at_freq[bus_z][con_bus_z]
                                # dV/dv
                                H_v_v[idx_z, idx_v_z] = 1 + 1j
                                # dV/di
                                # H_v_c[idx_z, idx_c] = sign / y_bus_at_freq[bus_z][con_bus_z]
                        # Strom unbekannt
                        elif (bus_z, con_bus_z) in mask_busses_x_c or (con_bus_z, bus_z) in mask_busses_x_c:
                            idx_c = mask_busses_x_c.index((bus_z, con_bus_z)) if bus_z < con_bus_z else (
                                mask_busses_x_c.index((con_bus_z, bus_z)))
                            # Logik für Stromrichtung
                            sign = 1 if bus_z < con_bus_z else -1

                            # Spannung am anderen Ende der Leitung auch bekannt
                            if con_bus_z in mask_busses_z_v:
                                idx_v_meas = mask_busses_z_v.index(con_bus_z)
                                h1[idx_z] = curr_z_v[idx_v_meas] + sign * I[idx_c] / y_bus_at_freq[bus_z][con_bus_z]
                                # dV/dv
                                # H_v_v[idx_z] = 0 + 0j
                                # dV/di
                                H_v_c[idx_z, idx_c] = sign / y_bus_at_freq[bus_z][con_bus_z]

                            # Spannung und Strom am Ende/ über die Leitung unbekannt
                            elif con_bus_z in mask_busses_x_v:
                                idx_v_z = mask_busses_x_v.index(con_bus_z)
                                h1[idx_z] = V[idx_v_z] + sign * I[idx_c] / y_bus_at_freq[bus_z][con_bus_z]
                                # dV/dv
                                H_v_v[idx_z, idx_v_z] = 1 + 1j
                                # dV/di
                                H_v_c[idx_z, idx_c] = sign / y_bus_at_freq[bus_z][con_bus_z]

                        # Letzter Bus der Leitung und bekannte Curr Injection?
                        elif bus_z in mask_curr_inj and connection_count == 1:
                            sign = 1 if connections[0][0] < connections[0][1] else -1
                            idx_inj = mask_curr_inj.index(bus_z)
                            idx_v_z = mask_busses_x_v.index(con_bus_z)
                            h1[idx_z] = V[idx_v_z] + sign * curr_z_inj[idx_inj] / y_bus_at_freq[connections[0][0]][connections[0][1]]
                            # dV/dv
                            H_v_v[idx_z, idx_v_z] = 1 + 1j


            # Currents
            for idx_z in range(0, len(h2)):
                bus_pair_z = mask_busses_z_c[idx_z]
                # Anzahl der Verbindungen an Knoten ermitteln
                con_count_from, con_from = get_connections_path(bus_pair_z[0], bus_pair_z[1], y_bus_at_freq)
                con_count_to, con_to = get_connections_path(bus_pair_z[1], bus_pair_z[0], y_bus_at_freq)
                # - Je nur eine Verbindung, keine Power Injection (Wichtigste Bedingung)
                if con_count_from == 1 and bus_pair_z[0] not in all_curr_inj :
                    # Strom durch Strommessung einer anderen Leitung bekannt
                    if con_from[0] in mask_busses_z_c:
                        idx_c = mask_busses_z_c.index(con_from[0])
                        h2[idx_z] = -curr_z_c[idx_c]
                        H_c_c[idx_z, idx_c] = 0 + 0j
                        continue
                    # Strom abhängig von Strom einer anderen Leitung
                    elif con_from[0] in mask_busses_x_c:
                        idx_c = mask_busses_x_c.index(con_from[0])
                        h2[idx_z] = I[idx_c]
                        H_c_c[idx_z, idx_c] = 1 + 1j
                        continue

                if con_count_to == 1 and bus_pair_z[1] not in all_curr_inj:
                    # Strom durch Strommessung einer anderen Leitung bekannt

                    if con_to[0] in mask_busses_z_c:
                        idx_c = mask_busses_z_c.index(con_to[0])
                        h2[idx_z] = curr_z_c[idx_c]
                        H_c_c[idx_z, idx_c] = 0 + 0j
                        continue
                    # Strom abhängig von Strom einer anderen Leitung
                    elif con_to[0] in mask_busses_x_c:
                        idx_c = mask_busses_x_c.index(con_to[0])
                        h2[idx_z] = I[idx_c]
                        H_c_c[idx_z, idx_c] = 1 + 1j
                        continue
                    bus_admittance = y_bus_at_freq[bus_pair_z[0]][bus_pair_z[1]]
                    # Spannungsabhängigkeiten
                    if bus_pair_z[0] in mask_busses_z_v and bus_pair_z[1] in mask_busses_z_v :
                        # Beide Spannungen bekannt
                        idx_v_1 = mask_busses_z_v.index(bus_pair_z[0])
                        idx_v_2 = mask_busses_z_v.index(bus_pair_z[1])
                        h2[idx_z] = (curr_z_v[idx_v_1] - curr_z_v[idx_v_2]) * bus_admittance
                        continue
                    elif bus_pair_z[0] in mask_busses_x_v and bus_pair_z[1] in mask_busses_z_v :
                        # Eine Spannungen bekannt
                        idx_v_1 = mask_busses_x_v.index(bus_pair_z[0])
                        idx_v_2 = mask_busses_z_v.index(bus_pair_z[1])
                        h2[idx_z] = (curr_z_v[idx_v_1] - curr_z_v[idx_v_2]) * bus_admittance
                        H_c_v[idx_z, idx_v_1] = bus_admittance
                        continue
                    elif bus_pair_z[0] in mask_busses_z_v and bus_pair_z[1] in mask_busses_x_v :
                        # Eine Spannungen bekannt
                        idx_v_1 = mask_busses_z_v.index(bus_pair_z[0])
                        idx_v_2 = mask_busses_x_v.index(bus_pair_z[1])
                        h2[idx_z] = (curr_z_v[idx_v_1] - curr_z_v[idx_v_2]) * bus_admittance
                        H_c_v[idx_z, idx_v_2] = -1 * bus_admittance
                        continue

                if con_count_from > 1 and bus_pair_z[0] not in all_curr_inj:
                    node_currents = np.matrix([0 + 0j])
                    for c in con_from:
                        sign = 1 if c[1] == bus_pair_z[0] else -1
                        if c in mask_busses_z_c:
                            idx_c_x = mask_busses_z_c.index(c)
                            node_currents += sign * curr_z_c[idx_c_x]
                        else:
                            idx_c_x = mask_busses_x_c.index(c)
                            node_currents += sign * I[idx_c_x]
                            H_c_c[idx_z, idx_c_x] = sign * (1 + 1j)
                    h2[idx_z] = node_currents
                elif con_count_to > 1 and bus_pair_z[1] not in all_curr_inj:
                    node_currents = np.matrix([0 + 0j])
                    for c in con_to:
                        sign = -1 if c[0] == bus_pair_z[1] else 1
                        if c in mask_busses_z_c:
                            idx_c_x = mask_busses_z_c.index(c)
                            node_currents += sign * curr_z_c[idx_c_x]
                        else:
                            idx_c_x = mask_busses_x_c.index(c)
                            node_currents += sign * I[idx_c_x]
                            H_c_c[idx_z, idx_c_x] = sign * (1 + 1j)
                    h2[idx_z] = node_currents


                # - Mehrere Ströme an Knoten 1 (auch Power Injection)
                # - Mehrere Ströme an Knoten 2 (auch Power Injection)

            h = np.concatenate((h1, h2))
            r = Z - h
            list_r.append(abs(r))
            # Jacobian
            H = np.zeros((len_measure_vector_v + len_measure_vector_c,
                                 len_state_vector_v + len_state_vector_c), dtype=np.complex128)

            H[:len_measure_vector_v, :len_state_vector_v] = H_v_v  # oben links
            H[:len_measure_vector_v, len_state_vector_v:] = H_v_c  # oben rechts
            H[len_measure_vector_v:, :len_state_vector_v] = H_c_v  # unten links
            H[len_measure_vector_v:, len_state_vector_v:] = H_c_c  # unten rechts

            if evaluate_H:
                for row in range(0, len(H)):
                    if row < len_measure_vector_c:
                        z_pos = "z_v_" + str(row + 1) + " bus " + str(mask_busses_z_v[row])
                    else:
                        z_pos = "z_c_" + str(row + 1 - len_measure_vector_v) + " con " + str(mask_busses_z_c[row - len_measure_vector_v])

                    entries, col_pos = count_entries(H[row])
                    strPos = ''
                    for p in col_pos:
                        if p < len_state_vector_v:
                            strPos += 'x_v_' + str(p + 1) + ' bus ' + str(mask_busses_x_v[p]) + '; '
                        else:
                            strPos += 'x_c_' + str(p + 1 - len_state_vector_v) + ' con ' + str(mask_busses_x_c[p - len_state_vector_v]) + '; '
                    print(f'Row {row} ({z_pos}) has {entries} entries at positions: {strPos}')

            # Gain matrix
            # Gm = np.transpose(H) * np.linalg.inv(Ri) * H
            H_T = H.T
            R_inv = np.linalg.inv(Ri)
            temp = np.dot(H_T, R_inv)

            Gm = np.dot(temp, H)

            dE = np.linalg.inv(Gm) * (np.transpose(H).dot(np.linalg.inv(Ri)) * r)
            E = E - dE


            I = E[len_state_vector_v:]
            V = E[0:len_state_vector_v]

            if evaluate_V_I:
                r_o_c = E - E_hist
                for row in range(0, len(r_o_c)):
                    if row < len_state_vector_v:
                        z_pos = "x_v_" + str(row + 1) + " bus " + str(mask_busses_z_v[row])
                    else:
                        z_pos = "z_c_" + str(row + 1 - len_state_vector_v) + " con " + str(mask_busses_z_c[row - len_state_vector_v])
                    print(f'Row {row} ({z_pos}) Rate of Change: {r_o_c[row]}')
            E_hist = E
            tol = np.sum(np.abs(dE))
            converge_list.append(tol)
            converge_list.pop(0)
            print(tol)

        # add to df
        for i in range(n_bus):
            if i in mask_busses_x_v:
                v = V[mask_busses_x_v.index(i), 0]
            else:
                v = z_v[d, mask_busses_z_v.index(i), f]
            v /= kv_scaler_complex[i]
            v1 = np.absolute(v)
            vangle1 = np.angle(v, deg=True)

            new_row = [str(i), 'bus' + str(i), '0', frequencies[f], v1, vangle1]

            # Append the new row to the DataFrame
            df.loc[f*n_bus + i] = new_row
    df.sort_values(by=['bus', 'freq'], inplace=True)
    df.set_index('bus', inplace=True)
    save_hse_results_to_csv(base_path, df, int(d*step_width))


