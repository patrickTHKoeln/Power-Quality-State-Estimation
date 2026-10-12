import numpy as np
import itertools
import cmath
from OpenDSSWrapper.converter.grids import CigreLVDistPV, BaseGrid
from OpenDSSWrapper.converter.components import ExternalGrid


# region general inspection
def get_slack_node(grid: BaseGrid):
    return grid.pp_net.ext_grid['bus'].values


def get_nodes_with_sources(grid: BaseGrid):
    return grid.pp_net.sgen['bus'].values


def get_nodes_with_loads(grid: BaseGrid):
    return grid.pp_net.load['bus'].values


def get_complete_y_mats_per_freq(grid: BaseGrid, ref_voltage_level_kv, frequencies,
                                 shunt_pos=-1, nodal_analysis_form=False,
                                 zero_seq=False):
    from OpenDSSWrapper.converter.admittance import get_y_matrices_per_frequency
    base_kvs_dict = grid.get_base_kvs()
    base_kvs = list(base_kvs_dict.values())
    y_mats = get_y_matrices_per_frequency(grid.pp_net, ref_voltage_level_kv, base_kvs, frequencies,
                                          shunt_position=shunt_pos, compact=nodal_analysis_form, include_slack=False,
                                          zero_seq=zero_seq)
    return y_mats


def get_pv_powers(pp_net):
    pv_indices = pp_net.sgen['bus']
    pv_1_ph_powers = 1E6 * pp_net.sgen['p_mw'] / 3
    return dict(zip(pv_indices, pv_1_ph_powers))


def get_adjacency_matrix(y_mat: np.matrix):
    return abs(y_mat) > 0


def get_complex_transformer_ratios(voltages_per_bus_kv: dict, ref_kv: float, lv_mv_angle=30):
    conversion_pairs = np.zeros(shape=(len(voltages_per_bus_kv), 2))
    for idx, v in enumerate(voltages_per_bus_kv.values()):
        conversion_pairs[idx, 0] = ref_kv / v
        conversion_pairs[idx, 1] = lv_mv_angle if v<ref_kv else 0
    return conversion_pairs
# endregion


# region combinations
def combine_all_states(load_states, pv_states, slack_states):
    # prepare nodal analysis
    all_keys = []
    all_values = []
    for d in [load_states, pv_states, slack_states]:
        for k, v in zip(d.keys(), d.values()):
            if len(v) > 0:
                all_keys.append(k)
                all_values.append(v)

    # Get all possible states
    np_possible_combinations = np.array(list(itertools.product(*all_values)))

    return all_keys, np_possible_combinations


def check_complex_curr_dir(curr: complex):
    """
    Checks if powerflow is positive or negative
    :param curr: complex current
    :return: + if current is load, - if source
    """
    return + 1 if curr.real > 0 else -1


def get_all_currents(voltages, nodes, admittances):
    """
    Get all current values from voltages across bus admittances along a path
    :param voltages: bus voltages along path's buses
    :param nodes: buses along the path
    :param admittances: admittances matrix with bus admittances
    :return: all possible current values along paths
    """
    all_currents = []
    local_voltages = voltages[nodes]
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            path_impedance = get_impedance_along_path(i, j, nodes, admittances)
            if nodes[i] < nodes[j]:  # this means a node at idx i is upstream
                all_currents.append((local_voltages[i] - local_voltages[j]) / path_impedance)
            else:
                all_currents.append((local_voltages[j] - local_voltages[i]) / path_impedance)
    return all_currents


def get_impedance_along_path(from_idx, to_idx, nodes, admittances):
    """Summarizes bus impedances along a path
    :param from_idx: start bus index
    :param to_idx: stop bus index
    :param nodes: nodes along path
    :param admittances: admittances matrix with bus admittances
    :return: bus impedances along path
    """
    impedance = 0 + 1j*0
    for p in range(from_idx, to_idx):
        impedance += 1/admittances[nodes[p], nodes[p + 1]]
    return impedance


def get_sign_of_current_in_node(path_at_node, node):
    """
    If the path is e.g., 2, 3, 4 it means, the current is flowing FROM the node, if it's 2, 1, 0 it's flowing TO
    the node. orderd Path is expected
    """
    if node == path_at_node[0]:
        return -1
    else:
        return +1


def calc_load_combinations(load_bus, state, all_paths, admittance_mat, vary_load_voltages):
    """
        Calculates all possible currents for a load.
        :param load_bus: voltage phasors
        :param state: current state
        :param all_paths: all paths
        :param admittance_mat: admittance matrix
        :param vary_load_voltages: more than 1 predicted voltages shall be computed
        :return: all possible virtual load impedances for the load_bus
        """
    # in the first version assume higher node means closer to slack
    possible_load_impedances = []
    load_voltages = [state[load_bus]]
    if vary_load_voltages:
        load_voltages.extend(_get_more_combinations_for_voltage(load_bus, state, admittance_mat))

    for voltage_at_load in load_voltages:

        # end of line
        if len(all_paths) == 1:
            all_currents = get_all_currents(state, all_paths[0], admittance_mat)
            possible_load_impedances.extend([voltage_at_load/c for c in all_currents])
        # attached to node with min two lines connected
        else:
            currents_at_node = []
            for path_at_node in all_paths:
                sign = get_sign_of_current_in_node(path_at_node, load_bus)
                currents = [c*sign for c in get_all_currents(state, path_at_node, admittance_mat)]
                currents_at_node.append(currents)

                # All possible combinations
            combinations = list(itertools.product(*currents_at_node))
            # Add all sums
            possible_load_currents = [sum(c) for c in combinations]
            possible_load_impedances.extend([voltage_at_load / c for c in possible_load_currents])

    # this can be manipulated, if unknown PV sources should be possible
    return [p for p in possible_load_impedances if p.real > 0]


def _get_more_combinations_for_voltage(bus, voltages, admittances):
    """
    Calculates other possible voltages for a load or pv by using possible line currents and voltage drops around
    :param bus: current bus
    :param voltages: all voltages in the state
    :param admittances: admittance matrix
    :return: all possible voltages for at the load or the pv bus
    """
    neighbour_buses = np.nonzero(admittances[bus])[0]
    if len(admittances)-1 in neighbour_buses:
        index = np.argwhere(neighbour_buses == len(admittances)-1)[0]
        neighbour_buses = np.delete(neighbour_buses, index)
    other_possible_voltages = []
    neighbour_voltages = []
    for nb_bus in neighbour_buses:
        neighbour_voltages.append(voltages[nb_bus])

        connected_buses = np.nonzero(admittances[nb_bus])[0]
        # remove the actual bus and the ground connection
        index = np.argwhere(connected_buses == len(admittances) - 1)[0]
        connected_buses = np.delete(connected_buses, index)
        index = np.argwhere(connected_buses == bus)[0]
        connected_buses = np.delete(connected_buses, index)
        currents_at_node = []

        for con_bus in connected_buses:
            # check if the connecting bus is a switch -> this algorithms creates large currents in this case
            # also use the predicted voltage at the other terminal of the switch
            if abs(admittances[nb_bus, con_bus]) > 1e3:
                neighbour_voltages.append(voltages[con_bus])
            if con_bus > nb_bus:  # from the node
                currents_at_node.append(-1*(voltages[nb_bus] - voltages[con_bus]) * admittances[nb_bus, con_bus])
            else:  # to the node
                currents_at_node.append((voltages[con_bus] - voltages[nb_bus]) * admittances[nb_bus, con_bus])
        for v_nb in neighbour_voltages:
            other_possible_voltages.append(v_nb - np.sum(currents_at_node) / admittances[bus, nb_bus])
    return other_possible_voltages


def calc_pv_combinations(pv_bus, state, all_paths, admittance_mat, rated_1ph_power_w, vary_pv_voltages=False):
    """
    Calculates all possible currents for a PV system. if Current is positive, assume a load.
    :param pv_bus: voltage phasors
    :param state: current state
    :param all_paths: all paths
    :param admittance_mat: admittance matrix
    :param rated_1ph_power_w: limit for 1phase power
    :param vary_pv_voltages: Consider other possibilities for the actual pv voltage
    :return: all possible voltage phasors
    """
    possible_pv_voltages = [state[pv_bus]]
    if vary_pv_voltages:
        possible_pv_voltages.extend(_get_more_combinations_for_voltage(pv_bus, state, admittance_mat))
    possible_load_impedances = []
    all_currents_at_node = []

    for voltage_at_pv in possible_pv_voltages:
        # in the first version assume higher node means closer to slack
        # end of line
        if len(all_paths) == 1:
            currents_at_node = get_all_currents(state, all_paths[0], admittance_mat)
            sign_current = -1 if all_paths[0][0] > all_paths[0][-1] else 1
            currents_at_node = [sign_current*c for c in currents_at_node]
            possible_load_impedances.extend([voltage_at_pv/c for c in currents_at_node if (voltage_at_pv/c).real > 0])
            currents_at_node = [c for c in currents_at_node if (voltage_at_pv/c).real < 0]
        # attached to node with min two lines connected
        else:
            all_currents = []
            for path_at_node in all_paths:
                sign = get_sign_of_current_in_node(path_at_node, pv_bus)
                currents = [c*sign for c in get_all_currents(state, path_at_node, admittance_mat)]
                all_currents.append(currents)

                # All possible combinations
            combinations = list(itertools.product(*all_currents))
            # Add all sums
            currents_at_node = [sum(c) for c in combinations]
            possible_load_impedances.extend([voltage_at_pv / c for c in currents_at_node if (voltage_at_pv / c).real > 0])
            currents_at_node = [c for c in currents_at_node if (voltage_at_pv/c).real < 0]
        # check if the current is greater as the potential max current
        check_max_current_constraint(currents_at_node, voltage_at_pv, rated_1ph_power_w)
        all_currents_at_node.extend(currents_at_node)
    return possible_load_impedances + all_currents_at_node


def check_max_current_constraint(currents_a, voltage_v, max_1ph_power_w):
    for idx_c in range(0, len(currents_a)):
        if abs(currents_a[idx_c]) * abs(voltage_v) > max_1ph_power_w:
            abs_current = max_1ph_power_w / abs(voltage_v)
            limited_current = complex(abs_current * np.cos(cmath.phase(voltage_v)),
                                        abs_current * np.sin(cmath.phase(voltage_v)))
            if limited_current not in currents_a:
                currents_a[idx_c] = limited_current


def calc_slack_combinations(slack_bus, state, all_paths, admittance_mat, slack_sc_impedance, vary_slack_voltages=False):
    """
    Calculates all possible voltages for a slack system.
    :param slack_bus: voltage phasors
    :param state: current state
    :param all_paths: all paths
    :param admittance_mat: admittance matrix
    :param slack_sc_impedance: Single phase shortcircuit impedance from S_k''
    :param vary_slack_voltages: Search for more possible Slack voltages
    :return: all possible voltage phasors
    """

    slack_voltages = [state[slack_bus]]
    possible_slack_voltages = []
    if vary_slack_voltages:
        slack_voltages.extend(_get_more_combinations_for_voltage(slack_bus, state, admittance_mat))
    for v_bus in slack_voltages:
        # in the first version assume higher node means closer to slack
        # end of line
        if len(all_paths) == 1:
            currents_at_node = get_all_currents(state, all_paths[0], admittance_mat)
            possible_slack_voltages.extend([v_bus + slack_sc_impedance*c for c in currents_at_node])
        # attached to node with min two lines connected
        else:
            all_currents = []
            for path_at_node in all_paths:
                sign = get_sign_of_current_in_node(path_at_node, slack_bus)
                currents = [c*sign for c in get_all_currents(state, path_at_node, admittance_mat)]
                all_currents.append(currents)

            # All possible combinations
            combinations = list(itertools.product(*all_currents))
            # Add all sums
            currents_at_node = [sum(c) for c in combinations]
            possible_slack_voltages.extend([v_bus + slack_sc_impedance*c for c in currents_at_node])

    return possible_slack_voltages


def reorder_paths(p_list):
    ordered_paths = []
    for p in p_list:
        ordered_paths.append(sorted(p))
    return ordered_paths


def get_all_possible_load_states(adjacency_mat, state, indices_loads, indices_pv_buses, admittances, vary_load_voltages=False):
    possible_load_impedances_dict = {}
    for load in indices_loads:
        all_paths_per_load = []
        get_paths_from_node(load, adjacency_mat, all_paths_per_load, indices_pv_buses)
        all_paths_per_load = reorder_paths(all_paths_per_load)
        possible_load_impedances_dict[load] = calc_load_combinations(load, state, all_paths_per_load, admittances, vary_load_voltages)
    return possible_load_impedances_dict


def get_all_possible_pv_states(adjacency_mat, state, indices_pv, indices_load_buses, admittances, pv_1ph_p_w, vary_pv_voltages=False):
    possible_pv_currents_or_impedances_dict = {}
    for pv in indices_pv:
        all_paths_per_pv = []
        get_paths_from_node(pv, adjacency_mat, all_paths_per_pv, indices_load_buses)
        all_paths_per_pv = reorder_paths(all_paths_per_pv)
        possible_pv_currents_or_impedances_dict[pv] = calc_pv_combinations(pv, state, all_paths_per_pv, admittances, pv_1ph_p_w[pv], vary_pv_voltages)

    return possible_pv_currents_or_impedances_dict


def get_all_possible_slack_states(adjacency_mat, state, indices_slacks, indices_load_buses_and_pv_buses, admittances,
                                  grid_slack:BaseGrid, target_voltage_kv, freq_hz, zero_seq, vary_slack_voltages=False):
    possible_slack_voltages_dict = {}
    internal_slack_impedances = []
    for sl in indices_slacks:
        this_slack = None
        for idx, ext in grid_slack.pp_net.ext_grid.iterrows():
            if ext['bus'] == sl:
                this_slack = ExternalGrid(ext, target_voltage_kv, freq_hz)
        if this_slack is None:
            return possible_slack_voltages_dict
        z_q_complex = 1/this_slack.get_pi_longitude_admittance(target_voltage_kv, freq_hz, zero_seq)
        internal_slack_impedances.append(z_q_complex)
        all_paths_per_slack = []
        get_paths_from_node(sl, adjacency_mat, all_paths_per_slack, indices_load_buses_and_pv_buses)
        all_paths_per_slack = reorder_paths(all_paths_per_slack)
        possible_slack_voltages_dict[sl] = calc_slack_combinations(sl, state, all_paths_per_slack, admittances,
                                                                   z_q_complex, vary_slack_voltages)

    return possible_slack_voltages_dict, internal_slack_impedances
# endregion


# region meshes
def get_paths_from_node(start, adjacency_matrix, all_paths, indices_other, curr_path=None, level=0):
    """
    This method searches possible paths without
    :param start: start node
    :param adjacency_matrix: adjacency matrix
    :param all_paths: all paths
    :param indices_other: indices of other paths
    :param curr_path: current path
    :param level: current level
    :return: all paths
    """
    connections = np.where(adjacency_matrix[start])[0]
    # the path ends when it's 1 connection or more than 2
    if len(connections) == 2 or (len(connections) > 2 and level == 0):
        for con_bus in connections:
            if level == 0:  # always add the start at the beginning
                curr_path = []
                curr_path.append(start)
            if con_bus in indices_other:  # this can be everything drawing or injecting current
                curr_path.append(con_bus)
                all_paths.append(curr_path)
            elif con_bus not in curr_path:
                curr_path.append(con_bus)
                get_paths_from_node(con_bus, adjacency_matrix, all_paths, indices_other, curr_path, level + 1)
    else:
        if level == 0 and len(connections) == 1:
            curr_path = [start, connections[0]]
            all_paths.append(curr_path)
            if connections[0] not in indices_other:
                get_paths_from_node(connections[0], adjacency_matrix, all_paths, indices_other, curr_path, level + 1)
        elif len(connections) > 2 and level > 0:
            # break as a new node was reached
            if curr_path not in all_paths:
                all_paths.append(curr_path)


def get_all_meshes_in_grid(impedances_or_admittances):
    # identify meshes
    all_meshs_and_values = []
    idx_ground = len(impedances_or_admittances) - 1
    for index, row in enumerate(impedances_or_admittances):
        positive_count = len(np.where(np.abs(row) > 0)[0])  # Zähle die Anzahl der Einträge > 0 in der Zeile
        if positive_count == 2:
            get_mesh_values_iterative(impedances_or_admittances, index, all_meshs_and_values, idx_ground)
            break
    # reorder all meshes, so the upper node should be start
    ordered_mesh_list = []
    for m in all_meshs_and_values:

        if m[1][0] > m[1][1]:
            ordered_mesh = [m[2], (m[1][1], m[1][0]), m[0]]
        else:
            ordered_mesh = m
        if ordered_mesh[0][0] > ordered_mesh[2][0]:
            ordered_mesh = [ordered_mesh[2], ordered_mesh[1], ordered_mesh[0]]
        ordered_mesh_list.append(ordered_mesh)
    return ordered_mesh_list


def get_mesh_values_iterative(impedances, idx_start, all_meshes_and_values, idx_ground, all_connections_used=None):
    """
    This method finds the next mesh, which is attached to a starting branch (idx_start vs ground).
    """

    if all_connections_used is None:
        all_connections_used = []
    act_mesh = []

    first_branch_connection = (idx_start, idx_ground)  # dir is node to ground
    connections = np.where(np.abs(impedances[idx_start, :idx_ground]) > 0)
    act_mesh.append(first_branch_connection)
    mesh_closed = False
    for branch in connections[0]:
        if (branch, idx_start) in all_connections_used or (idx_start, branch) in all_connections_used:
            continue
        if mesh_closed:
            # get next mesh, if more connections are available end the current mesh is closed
            get_mesh_values_iterative(impedances, branch, all_meshes_and_values, idx_ground, all_connections_used)
        else:
            # close mesh
            act_mesh.append((branch, idx_start))
            all_connections_used.extend([(branch, idx_start), (idx_start, branch)])
            act_mesh.append((branch, idx_ground))
            all_meshes_and_values.append(act_mesh)
            mesh_closed = True
            # are there any connections form the closing node?
            get_mesh_values_iterative(impedances, branch, all_meshes_and_values, idx_ground, all_connections_used)

# endregion
