from OpenDSSWrapper.converter.grids import BaseGrid, CigreLVDistPV, MicrogridTHKoeln
import numpy as np
import os
import pandas as pd
import copy
from tqdm import tqdm
from clustering import DbscanCluster
from distance_calculations import compute_hausdorff_distance, compute_euclidean_distance, compute_mse_distance
from distance_calculations import compute_mae_distance, complex_to_2d_array
from evaluation.evaluate_validation import eval_results_per_node, eval_results, get_worst_validations
from mesh_current_analysis import mesh_analysis, calc_grid_voltages
from grid_inspection import (get_all_possible_load_states, get_all_possible_pv_states, get_all_possible_slack_states,
                             get_nodes_with_loads, get_nodes_with_sources, get_slack_node, combine_all_states,
                             get_all_meshes_in_grid, get_complete_y_mats_per_freq, get_adjacency_matrix,
                             get_complex_transformer_ratios, get_pv_powers)
"""
This algorithm evaluates all potential system states for state estimations for ANNs.
It provides an overview over physically calculated system states and ranks the prediction within possible states.
The mesh current analysis is used to compute the nodel voltage phasors.
"""
DEBUG = False
EVAL = False
VARY_LOAD_VOLTAGES = True
measure_comparison = {}
A = np.cos(2/3 * np.pi) + 1j*np.sin(2/3 * np.pi)
FORTESCUE_MATRIX = (1/3)*np.array([[1, A, A**2], [1, A**2, A],[1, 1, 1]], dtype=np.complex128)
INV_FORTESCUE_MATRIX = np.array([[1, 1, 1], [1, A**2, A], [1, A, A**2]], dtype=np.complex128)

# region I/O
def get_system_state(se_results_path, file_name):
    p_data_se = os.path.join(se_results_path, file_name)
    all_complex_phasors = pd.read_pickle(p_data_se)
    return all_complex_phasors


def save_validation_results(se_results_path, data, file_name='Validation_results.pic'):
    p_validation_se = os.path.join(se_results_path, file_name)
    import pickle
    with open(p_validation_se, 'wb') as pic:
        pickle.dump(data, pic)


def load_validation_results(se_results_path, file_name='Validation_results.pic'):
    p_validation_se = os.path.join(se_results_path, file_name)
    if os.path.exists(p_validation_se):
        import pickle
        with open(p_validation_se, 'rb') as pic:
            return pickle.load(pic)
    else:
        return None


def get_complex_voltages(current_state, complex_transformer_ratios):
    if np.iscomplexobj(current_state):
        result = np.squeeze(np.asarray(current_state))
    else:
        result = current_state[0, :] + 1j*current_state[1, :]
    angles = complex_transformer_ratios[:,1]
    magnitudes = np.ones_like(angles)
    complex_rotation = magnitudes*(np.cos(np.pi*angles/180) +1j*np.sin(np.pi*angles/180))
    result = result * complex_rotation
    return result
# endregion


def debug_results(edges_and_currents, coords):
    import networkx as nx
    import matplotlib.pyplot as plt

    # Initialisiere den Graphen
    G = nx.Graph()

    # Füge Knoten und Kanten hinzu
    edges = [e for e in edges_and_currents.keys()]
    G.add_edges_from(edges)

    # Definiere Farben für die Kanten
    edge_colors = []
    # Füge Labels für die Kanten hinzu
    edge_labels = {}
    edgelist = []
    for e in edges_and_currents.keys():
        raw_string = (str(round(abs(edges_and_currents[e]), 3)) +
                      u"\u2220" + str(round(np.angle(edges_and_currents[e], deg=True), 2)) + '°')
        edge_labels[e] = r'{}'.format(raw_string)
        edge_colors.append('blue' if edges_and_currents[e].real >= 0 else 'red')
        edgelist.append(e)

    # Zeichne den Graphen
    #  pos = nx.spring_layout(G, k=5/math.sqrt(len(edge_colors)))  # Positionen der Knoten im Layout
    pos = coords.to_dict(orient='index')
    pos = {key: (value['x'], value['y']) for key, value in pos.items()}
    nx.draw(G, pos, node_color='lightblue', node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=edge_colors)
    # Zeichne die Kanten-Labels
    nx.draw_networkx_edge_labels(G, pos, font_size=7, font_weight='bold', horizontalalignment='left', edge_labels=edge_labels, rotate=False)
    nx.draw_networkx_labels(G, pos)
    plt.show()


def calc_combination(state, keys, admittances, available_pv_indices, available_load_indices, available_slack_indices,
                     all_slack_impedances, grid_meshes, geo_data):
    # reset impedances
    current_admittance_mat = admittances.copy()
    current_sources = []  # [(0, 1, 1)]  # Beispielwerte: eine Stromquelle von Knoten 0 zu Knoten 1 mit 1A

    # check if PV is in load state (no current is injected)
    for idx in available_pv_indices:
        pv_state = state[idx]
        pv_key = keys[idx]
        if pv_state.real < 1: #TODO find better logic
            # multiply by -1 to create current source orientation
            current_sources.append((-1, pv_key, -pv_state))
        else:
            current_admittance_mat[-1, pv_key] += 1 / pv_state
            current_admittance_mat[pv_key, -1] = current_admittance_mat[-1, pv_key]

    # add load impedances
    load_keys = keys[available_load_indices]
    current_admittance_mat[-1, load_keys] += 1 / state[available_load_indices]
    current_admittance_mat[load_keys, -1] = current_admittance_mat[-1, load_keys]

    # update slacks as voltage source
    current_v_sources = []
    for i, idx in enumerate(available_slack_indices):
        current_v_sources.append([state[idx], keys[idx], all_slack_impedances[i]])

    # convert admittances to impedances
    non_zero_entries = abs(current_admittance_mat) > 0
    grid_impedances = np.zeros_like(current_admittance_mat, dtype=complex)
    grid_impedances[non_zero_entries] = 1 / current_admittance_mat[non_zero_entries]

    solved_grid_currents, solved_slack_currents = mesh_analysis(grid_impedances, current_v_sources, current_sources,
                                                                copy.deepcopy(grid_meshes))

    # calculate nodal voltages
    grid_voltages = calc_grid_voltages(grid_impedances, solved_grid_currents,
                                       current_v_sources[0], solved_slack_currents[0])

    if DEBUG:
        debug_results(solved_grid_currents, geo_data)

    return grid_voltages


def compute_all_states(idx_act_state, all_combinations, all_keys, admittances, pv_indices,
                       l_indices, s_indices, s_impedances, g_meshes, g_data, c_state, all_results,
                       results_orig, results_masked, mask_maes_nodes, cur_sequence=''):

    for comb in tqdm(enumerate(all_combinations), total=len(all_combinations)):
        idx_comb, state = comb

        g_voltages = calc_combination(state, all_keys, admittances, pv_indices, l_indices,
                                      s_indices, s_impedances, g_meshes, g_data)
        all_results.append(g_voltages)

        # results evaluation
        vect_voltages = complex_to_2d_array(g_voltages)
        vect_state = complex_to_2d_array(c_state)
        deltas = vect_state - vect_voltages
        results_orig['euclid'].append(compute_euclidean_distance(vect_state, vect_voltages, deltas))
        results_orig['hausdorff'].append(compute_hausdorff_distance(vect_state, vect_voltages))
        results_orig['mse'].append(compute_mse_distance(vect_state, vect_voltages, deltas))
        results_orig['mae'].append(compute_mae_distance(vect_state, vect_voltages, deltas))

        # results evaluation
        # vect_voltages = complex_to_2d_array(g_voltages)[mask_maes_nodes]
        # vect_state = complex_to_2d_array(c_state)[mask_maes_nodes]
        # deltas = vect_state - vect_voltages
        # results_masked['euclid'].append(compute_euclidean_distance(vect_state, vect_voltages, deltas))
        # results_masked['hausdorff'].append(compute_hausdorff_distance(vect_state, vect_voltages))
        # results_masked['mse'].append(compute_mse_distance(vect_state, vect_voltages, deltas))
        # results_masked['mae'].append(compute_mae_distance(vect_state, vect_voltages, deltas))

    idx_best, eval_data = generate_summary('Fit all', idx_act_state, all_results, results_orig['euclid'],
                                           results_orig['hausdorff'], results_orig['mse'], results_orig['mae'],
                                           c_state, cur_seq=cur_sequence)
    # idx_best_masked, eval_data_masked = generate_summary('Masked fit', idx_act_state, all_results, results_masked['euclid'],
    #                                                     results_masked['hausdorff'], results_masked['mse'],
    #                                                     results_masked['mae'], c_state)

    return idx_best, eval_data # , idx_best_masked, eval_data_masked


def run_validation_for_state(prediction, ref_kv, ref_hz, geo_data, grid, idx_act_state, admittances, adjacency_mat,
                             pv_1ph_max_power_w, mask_measurements=None, reduce_dim=False, zero_seq=False,
                             cur_seq=''):
    load_clustering = DbscanCluster(0.15, 1)
    pv_clustering = DbscanCluster(0.15, 1)
    slack_clustering = DbscanCluster(0.15, 1)
    len_comb_all = np.zeros(2)
    complex_state = get_complex_voltages(prediction, get_complex_transformer_ratios(grid.get_base_kvs(), ref_kv))
    # complex_ground_truth = get_complex_voltages(sim_truth,
    #                                             get_complex_transformer_ratios(grid.get_base_kvs(), ref_kv))

    # get grid node attachments
    indices_loads = get_nodes_with_loads(grid)
    indices_pv = get_nodes_with_sources(grid)
    indices_loads = [l for l in indices_loads if l not in indices_pv]
    indices_sources = get_slack_node(grid)

    # calculate theoretic loads
    all_load_states_orig = get_all_possible_load_states(adjacency_mat, complex_state, indices_loads, indices_pv,
                                                        admittances, VARY_LOAD_VOLTAGES)

    # calculate theoretic pv systems
    all_pv_states_orig = get_all_possible_pv_states(adjacency_mat, complex_state, indices_pv, indices_loads, admittances,
                                                    pv_1ph_max_power_w, VARY_LOAD_VOLTAGES)

    # calculate theoretic slacks
    all_other_components = np.concatenate((indices_loads, indices_pv))
    all_slack_states_orig, all_slack_impedances = get_all_possible_slack_states(adjacency_mat, complex_state,
                                                                                indices_sources, all_other_components,
                                                                                admittances, grid, ref_kv, ref_hz,
                                                                                zero_seq, VARY_LOAD_VOLTAGES)

    if reduce_dim:
        all_load_states = load_clustering.cluster_data(all_load_states_orig)
        all_pv_states = pv_clustering.cluster_data(all_pv_states_orig)
        all_slack_states = slack_clustering.cluster_data(all_slack_states_orig)
    else:
        all_load_states, all_pv_states, all_slack_states = (all_load_states_orig, all_pv_states_orig,
                                                            all_slack_states_orig)

    keys, np_all_possible_combinations = combine_all_states(all_load_states, all_pv_states, all_slack_states)
    len_comb_all[0] = len(np_all_possible_combinations)

    available_load_indices = [keys.index(idx) for idx in keys if idx in indices_loads]
    available_pv_indices = [keys.index(idx) for idx in keys if idx in indices_pv]
    available_slack_indices = [keys.index(idx) for idx in keys if idx in indices_sources]
    keys = np.array(keys)

    non_zero_entries = abs(admittances) > 0
    grid_impedances = np.zeros_like(admittances, dtype=complex)
    grid_impedances[non_zero_entries] = 1 / admittances[non_zero_entries]

    grid_meshes = get_all_meshes_in_grid(grid_impedances)

    # analyze clustered data
    physical_results = []
    all_deviations_euclid = []; all_deviations_hau = []; all_deviations_mse = []; all_deviations_mae = []
    all_deviations_euclid_mask = []; all_deviations_hau_mask = []; all_deviations_mse_mask = []; all_deviations_mae_mask = []
    results_original = {'euclid': all_deviations_euclid, 'hausdorff': all_deviations_hau,
                        'mse': all_deviations_mse, 'mae': all_deviations_mae}
    results_masked = {'euclid': all_deviations_euclid_mask, 'hausdorff': all_deviations_hau_mask,
                      'mse': all_deviations_mse_mask, 'mae': all_deviations_mae_mask}
    # .., idx_best_fit_masked, eval_data_masked
    idx_best_fit, eval_data, = compute_all_states(idx_act_state,
                                          np_all_possible_combinations, keys, admittances, available_pv_indices,
                                          available_load_indices, available_slack_indices, all_slack_impedances,
                                          grid_meshes, geo_data, complex_state, physical_results,
                                          results_original, results_masked, mask_measurements,
                                          cur_sequence=cur_seq)

    # get specific data for best clustered data
    if reduce_dim:
        new_load_states = load_clustering.get_cluster_by_combination(keys, indices_loads,
                                                                     np_all_possible_combinations[idx_best_fit],
                                                                     all_load_states_orig)
        new_pv_states = pv_clustering.get_cluster_by_combination(keys, indices_pv,
                                                                 np_all_possible_combinations[idx_best_fit],
                                                                 all_pv_states_orig)
        new_slack_states = slack_clustering.get_cluster_by_combination(keys, indices_sources,
                                                                       np_all_possible_combinations[idx_best_fit],
                                                                       all_slack_states_orig)
        _, np_all_possible_combinations = combine_all_states(new_load_states, new_pv_states, new_slack_states)
        len_comb_all[1] = len(np_all_possible_combinations)
        # analyze all data from best clusters
        # , idx_best_fit_masked, eval_data_masked
        idx_best_fit, eval_data = compute_all_states(idx_act_state,
                                                       np_all_possible_combinations, keys, admittances,
                                                       available_pv_indices, available_load_indices,
                                                       available_slack_indices, all_slack_impedances, grid_meshes,
                                                       geo_data, complex_state, physical_results,
                                                       results_original, results_masked, mask_measurements, mask_merged_buses)

    eval_data['comb_count_1'] = len_comb_all[0]
    eval_data['comb_count_2'] = len_comb_all[1]
    # eval_data_masked['comb_count_1'] = len_comb_all[0]
    # eval_data_masked['comb_count_2'] = len_comb_all[1]

    # eval_data['raw_data'] = physical_results
    return eval_data #, eval_data_masked


def generate_summary(title, idx_act_state, res_list, all_deviations_euc, all_deviations_hau, all_deviations_mse,
                     all_deviations_mae, c_state, c_ground_truth=0+0j, cur_seq=''):
    print(
        f'-----------------------------{title} results state {idx_act_state} Seq:{cur_seq}---------------------------')
    min_euc = min(all_deviations_euc)
    min_hau = min(all_deviations_hau)
    min_mse = min(all_deviations_mse)
    min_mae = min(all_deviations_mae)
    max_euc = max(all_deviations_euc)
    max_hau = max(all_deviations_hau)
    max_mse = max(all_deviations_mse)
    max_mae = max(all_deviations_mae)
    idx_best_euc = all_deviations_euc.index(min_euc)
    idx_best_hau = all_deviations_hau.index(min_hau)
    idx_best_mae = all_deviations_mae.index(min_mae)
    idx_best_mse = all_deviations_mse.index(min_mse)
    best_indices = [idx_best_euc, idx_best_hau, idx_best_mae, idx_best_mse]
    mae_best_mag = [np.mean(np.abs(abs(res_list[i]) - abs(c_ground_truth))) for i in best_indices]
    mae_pred_mag = np.mean(np.abs(abs(c_state) - abs(c_ground_truth)))
    mae_euc_mag = np.mean(np.abs(abs(c_state) - abs(res_list[idx_best_euc])))
    mae_euc_ang = np.mean(np.abs(np.angle(c_state, deg=True) - np.angle(res_list[idx_best_euc], deg=True)))
    mae_best_ang = [np.mean(np.abs(np.angle(res_list[i], deg=True) - np.angle(c_ground_truth, deg=True))) for i in best_indices]
    mae_pred_ang = np.mean(np.abs(np.angle(c_state, deg=True) - np.angle(c_ground_truth, deg=True)))
    mae_results_magnitude = {'Prediction': mae_pred_mag, 'Euclidean': mae_best_mag[0], 'Hausdorff': mae_best_mag[1],
                             'MAE': mae_best_mag[2], 'MSE': mae_best_mag[3]}
    mae_results_phase = {'Prediction': mae_pred_ang, 'Euclidean': mae_best_ang[0], 'Hausdorff': mae_best_ang[1],
                             'MAE': mae_best_ang[2], 'MSE': mae_best_ang[3]}
    state_data = {'ground_truth': c_ground_truth, 'predicted_state': c_state, 'best_fit_Euclidean': res_list[idx_best_euc],
                  'best_fit_Hausdorff': res_list[idx_best_hau], 'best_fit_mae': res_list[idx_best_mae],
                  'best_fit_mse': res_list[idx_best_mse]}
    decision_data = {'Euclidean_idx': idx_best_euc, 'Euclidean_min': min_euc, 'Euclidean_max': max_euc,
                     'Hausdorff_idx': idx_best_hau, 'Hausdorff_min': min_hau, 'Hausdorff_max':max_hau,
                     'MAE_idx': idx_best_mae, 'MAE_min': min_mae, 'MAE_max': max_mae,
                     'MSE_idx': idx_best_mse, 'MSE_min': min_mse, 'MSE_max': max_mse}
    store_data = {'state_data': state_data, 'mae_results_magnitude': mae_results_magnitude,
                  'mae_results_phase': mae_results_phase, 'decision_data': decision_data}

    # print(f'Euclidean min: {min_euc}, max: {max_euc} Hausdorff min: {min_hau}, max: {max_hau}')
    # print(f'MSE min: {min_mse}, max: {max_mse} MAE min: {min_mae}, max: {max_mae}')
    # print(f'Indices: Euc: {idx_best_euc}, Hausdorff: {idx_best_hau}, MAE: {idx_best_mae}, MSE: {idx_best_mse}')
    # print(f'MAEs Mag.: Euc: {mae_best_mag[0]}, Hausdorff: {mae_best_mag[1]}, MAE: {mae_best_mag[2]}, MSE: {mae_best_mag[3]}')
    # print(f'MAEs Ang.: Euc: {mae_best_ang[0]}, Hausdorff: {mae_best_ang[1]}, MAE: {mae_best_ang[2]}, MSE: {mae_best_ang[3]}')
    # print(f'MAE pred Mag.: {mae_pred_mag}')
    # print(f'MAE pred Ang.: {mae_pred_ang}')
    print(f'MAE Val Mag.: {mae_euc_mag}V')
    print(f'MAE Val Ang.: {mae_euc_ang}°')
    return idx_best_mse, store_data


def compute_sequences(unbalanced_voltages):
    complex_voltages = np.matrix(unbalanced_voltages[:, :, 0] + 1j * unbalanced_voltages[:, :, 1])
    return np.dot(complex_voltages, FORTESCUE_MATRIX.T)


def compute_unbalanced_voltages(sequence_voltages):
    return np.dot(sequence_voltages, INV_FORTESCUE_MATRIX)


def remove_switches_merge_buses(pp_net, admit_mat, shunt_pos):
    from OpenDSSWrapper.converter.components import Switch
    merge_list = []
    for idx, s in pp_net.switch.iterrows():
        switch = Switch(s, 1, 1)
        if switch.is_closed():
            upstream_bus = switch.from_bus if switch.from_bus < switch.to_bus else switch.to_bus
            downstream_bus = switch.from_bus if switch.from_bus > switch.to_bus else switch.to_bus
            merge_list.append((upstream_bus, downstream_bus))  # the first bus will be kept. the second removed

    for merge_b in merge_list:
        for k in admit_mat['pos_seq'].keys():
            admit_mat['pos_seq'][k][:, merge_b[0]] += admit_mat['pos_seq'][k][:, merge_b[1]]
            admit_mat['pos_seq'][k][merge_b[0], :] += admit_mat['pos_seq'][k][merge_b[1], :]
            admit_mat['zero_seq'][k][:, merge_b[0]] += admit_mat['zero_seq'][k][:, merge_b[1]]
            admit_mat['zero_seq'][k][merge_b[0], :] += admit_mat['zero_seq'][k][merge_b[1], :]

            admit_mat['pos_seq'][k][:, merge_b[1]] = 0 + 0j
            admit_mat['pos_seq'][k][merge_b[1], :] = 0 + 0j
            admit_mat['zero_seq'][k][:, merge_b[1]] = 0 + 0j
            admit_mat['zero_seq'][k][merge_b[1], :] = 0 + 0j

    return merge_list


def run_validation_for_grid(source_path, results_path, figure_path, ground_truth_file_name, pred_file_name, m_name,
                            grid: BaseGrid = CigreLVDistPV(), ref_voltage_kv=20.0, mask_meas=None, mask_true=None,
                            reduce_dims=True):
    frequencies = [o*50 for o in range(1, 26)]
    f_base_name = 'Validation_results_dict'
    f_name_results = f'{f_base_name}.pic'
    f_name_results_masked = f'{f_base_name}_masked.pic'
    if mask_meas is None:
        mask_meas = [6, 22, 28]
    if mask_true is None:
        mask_true = [i for i in grid.descriptive_nodes]
    if EVAL:
        results = load_validation_results(results_path, f_name_results)
        results_masked = load_validation_results(results_path, f_name_results_masked)
        # eval_results_per_node(results, figure_path)
        eval_results(results, results_masked, figure_path)
    else:
        # load grid and results
        geo_d = grid.pp_net.bus_geodata.drop(columns=['coords'])

        if isinstance(grid, CigreLVDistPV):
            # add geo data for recently added MV cable
            geo_d.loc[1, 'x'] = geo_d.loc[2, 'x']
            geo_d.loc[0, 'y'] = geo_d.loc[0, 'y'] - 1
            geo_d.loc[23, 'x'] = geo_d.loc[24, 'x']

        pv_1_ph_powers_w = get_pv_powers(grid.pp_net)

        all_predictions = get_system_state(source_path, pred_file_name)[m_name]['y_pred']
        all_ground_truth = get_system_state(source_path, ground_truth_file_name)[0]

        # get admittances and adjacency
        shunt_pos = -1
        admit_mat = {}
        admit_mat['pos_seq'] = get_complete_y_mats_per_freq(grid, ref_voltage_kv, frequencies, shunt_pos)
        admit_mat['zero_seq'] = get_complete_y_mats_per_freq(grid, ref_voltage_kv, frequencies, shunt_pos, zero_seq=True)

        # if switches are in the grid, merge these buses, as switches will disturb
        # merged_buses = remove_switches_merge_buses(grid.pp_net, admit_mat, shunt_pos)
        adj_mat = get_adjacency_matrix(admit_mat['pos_seq'][50])[:shunt_pos, :shunt_pos]

        validation_results = load_validation_results(results_path, f_name_results)
        validation_results_masked = load_validation_results(results_path, f_name_results_masked)

        if validation_results is None:
            validation_results = {}
            validation_results_masked = {}

        states_indices = np.arange(0, all_predictions.shape[0])
        select_state = -1
        if select_state >= 0:
            if select_state > 0:
                states_indices = [625]
            elif select_state == 0:
                all_comb_mag = [validation_results[r]['mae_results_magnitude'] for r in validation_results.keys()]
                states_indices = get_worst_validations(all_comb_mag, 'MAE', 10)
            else:
                states_indices = [s for s in states_indices if s not in validation_results.keys()]

        all_validations = np.zeros(all_predictions.shape[:4], dtype=np.complex128)
        for idx_f, f in enumerate(frequencies):
            for idx_state in states_indices:
                pred_seq = compute_sequences(all_predictions[idx_state, :, idx_f])

                for idx_seq in range(0, 3):
                    is_zero_seq = idx_seq == 2
                    key_seq = 'zero_seq' if is_zero_seq else 'pos_seq'
                    #, validation_results_masked[idx_state]
                    res_validation_at_f = run_validation_for_state(
                                pred_seq[:, idx_seq],
                                ref_voltage_kv, f, geo_d, grid, idx_state,
                                admit_mat[key_seq][f], adj_mat, pv_1_ph_powers_w,
                                mask_measurements=mask_meas, reduce_dim=reduce_dims, zero_seq=is_zero_seq,
                                cur_seq='neg_seq' if idx_seq == 1 else key_seq)

                    all_validations[idx_state, :, idx_f, idx_seq] = res_validation_at_f['state_data']['best_fit_mse']


            seq_residuals = compute_sequences(all_ground_truth[:, :, 0, :]) - all_validations[:, :, 0, :]
            pred_seq = compute_unbalanced_voltages(all_validations)
            save_validation_results(results_path, validation_results, file_name=f_name_results)
            save_validation_results(results_path, validation_results_masked, file_name=f_name_results_masked)



if __name__ == '__main__':
    """
    r_path = os.path.join('C:\\', 'Users', 'Patrick Mack', 'sciebo3',
                                '00_Ablage', 'Validation', 'Vergleich_all_meas_fit_1')
    f_path = os.path.join('C:\\', 'Users', 'Patrick Mack', 'sciebo3',
                               '01_Organisatorisches', '10_Promotionsdokumente', '99_Grafiken', 'lab_validation')
    res_f_name = 'PANN_Test_x.pic'
    """
    b_path = os.path.join('C:\\', 'Users', 'Patrick Mack', 'SynologyDrive', '02_Uni_Ausbildung', '06_Promotion')
    r_path = os.path.join(b_path, '98_Daten', '7000Epochs_exact_DNN_PANN')
    f_path = os.path.join(b_path, '99_Grafiken', 'lab_validation')
    res_f_name = 'true_y.pic'
    pred_f_name = 'pred_y.pic'
    model_name = 'DNN Gauss 0.01'
    run_validation_for_grid(r_path, f_path, res_f_name, res_f_name, pred_f_name, model_name,
                            MicrogridTHKoeln(None,None, False, False),
                            ref_voltage_kv=0.4, mask_meas=[0, 7], mask_true=[0, 3, 6, 7], reduce_dims=False)
