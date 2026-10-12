import numpy as np

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix


def get_sign_of_current_in_merged_mesh(merged_mesh, i_source_mesh, idx_ground):
    # do the meshes have the same orientation?
    merged_mesh_bus_conn = [c for c in merged_mesh if c[1] != idx_ground and c not in i_source_mesh][0]
    i_source_conn = [c for c in i_source_mesh if c[1] != idx_ground][0]
    mesh_1_downstream = i_source_conn[1] > i_source_conn[0]
    mesh_2_downstream = merged_mesh_bus_conn[1] > merged_mesh_bus_conn[0]
    # if the meshes have the same orientation, the coefficients are positive
    return 1 if mesh_1_downstream == mesh_2_downstream else -1


def superpose_merged_mesh_currents(currents_and_directions, common_path):
    """
    Calculate superposed currents depending on orientations
    :param currents_and_directions: format (form_bus, to_bus), complex float/double
    :param common_path: the matching path for all currents (form_bus, to_bus)
    """
    sum_current = 0 + 1j*0
    orientation_unknown_path = 1 if common_path[0] < common_path[1] else -1
    for c in currents_and_directions:
        orientation_mesh_path = 1 if c[0][0] < c[0][1] else -1
        if orientation_unknown_path == orientation_mesh_path:
            sum_current += c[1]
        else:
            sum_current -= c[1]
    return sum_current


def get_sign_of_current_coupled_mesh(curr_mesh, i_source_mesh, idx_ground):

    i_source_conn = [c for c in i_source_mesh if c[1] != idx_ground][0]
    curr_mesh_conn = [c for c in curr_mesh if c[1] != idx_ground and c != i_source_conn][0]
    # is the mesh with the i source located at the end of the current mesh? -> sign is actually "-"
    return 1 if curr_mesh_conn[1] == i_source_conn[0] else -1


def get_sign_in_coupled_mesh(curr_mesh, connected_mesh, idx_ground):
    # if the other node in the connected mesh is smaller (upstream), then the coefficient is
    # remove shunt connections
    bus_actual = [c for c in curr_mesh if c[1] != idx_ground][0]
    bus_next = [c for c in connected_mesh if c[1] != idx_ground][0]
    both_same_dir = (bus_next[1] < bus_next[0] and bus_actual[1] < bus_actual[0] or
                     bus_next[0] < bus_next[1] and bus_actual[0] < bus_actual[1])

    if both_same_dir:
        if bus_next[0] == bus_actual[0] or bus_next[1] == bus_actual[1]:
            return 1
        else:
            return -1
    else:
        if bus_next[0] == bus_actual[0] or bus_next[1] == bus_actual[1]:
            return 1
        else:
            return -1


def mesh_analysis(impedances, voltage_sources, all_current_sources, mesh_list):
    """
    Do mesh current analysis on base of impedances, voltage sources and ideal current sources.

    :param impedances: squared numpy matrix, which contains all branch and bus admittances.
    :param voltage_sources: List of Tuples [(cplx phasor, conn bus, cplx slack impedance), ...].
    :param all_current_sources: List of Tuples [(ground bus, conn bus, cplx current), ...],
    :param mesh_list: all directly couple meshes in the grid as List of List of Tuples [[(0,1), (1,2), 2,0)],...]
    :return: Dict for the grid currents, Numpy array for slack currents.
    """
    # Anzahl der Knoten und damit die Anzahl der Maschen
    num_mesh_currents = len(mesh_list)
    idx_ground = len(impedances) - 1
    orig_meshes = mesh_list.copy()
    i_sources_mesh_currents_dict = {} # connection part of [mesh] and current
    grid_currents = {}
    # create Resistance matrix
    num_mesh_currents = num_mesh_currents + len(voltage_sources)

    V = np.zeros(num_mesh_currents, dtype=complex)

    # update meshes with ideal current sources
    remove_i_source_meshes_idx = []
    for source in all_current_sources:
        tuple_i_source_branch = (source[1], idx_ground)
        coupled_meshes = [m for m in mesh_list if tuple_i_source_branch in m]

        if len(coupled_meshes) >= 1:
            idx_mesh = mesh_list.index(coupled_meshes[0])

            if len(coupled_meshes) == 2:
                # manipulate the first mesh to cover both meshes (current source is the branch between) -> merge
                mesh_list[mesh_list.index(coupled_meshes[0])].extend(mesh_list[mesh_list.index(coupled_meshes[1])])
                mesh_list[mesh_list.index(coupled_meshes[0])].remove(tuple_i_source_branch)  # remove duplicate
                mesh_list[mesh_list.index(coupled_meshes[0])].remove(tuple_i_source_branch)  # remove duplicate
                # the current source is at the end of the line, find attached meshes
                i_source_mesh_without_i = [m for m in coupled_meshes[0] if m in coupled_meshes[1] and m != tuple_i_source_branch]
                i_source_mesh_without_ground = [m for m in i_source_mesh_without_i if m[1] != idx_ground][0]

                rows, cols = zip(*i_source_mesh_without_i)
                V[idx_mesh] = (-get_sign_of_current_in_merged_mesh(coupled_meshes[0], coupled_meshes[1], idx_ground) *
                               source[2] * np.sum(impedances[rows, cols]))
                remove_i_source_meshes_idx.append(mesh_list.index(coupled_meshes[1]))
                i_sources_mesh_currents_dict[i_source_mesh_without_ground] = source[2]
                # are other meshes attached after the merged one?
                further_downstream_meshes = [m for m in mesh_list
                                             if i_source_mesh_without_i[1] in m and m[-1][0] > i_source_mesh_without_i[-1][0]]
                for m in further_downstream_meshes:
                    idx_next_mesh = mesh_list.index(m)
                    V[idx_next_mesh] = source[2] * impedances[i_source_mesh_without_i[1]]
            else:
                # the current source is at the end of the line, find attached meshes
                i_source_mesh_without_i = [m for m in coupled_meshes[0] if m != tuple_i_source_branch]
                i_source_mesh_without_ground = [m for m in i_source_mesh_without_i if m[1] != idx_ground][0]
                coupling_conn = [c for c in coupled_meshes[0] if c[1] == idx_ground and c != tuple_i_source_branch][0]
                influenced_meshes = [m for m in mesh_list if coupling_conn in m and m != coupled_meshes[0]]
                for im in influenced_meshes:
                    idx_mesh = mesh_list.index(im)
                    V[idx_mesh] = (-get_sign_of_current_coupled_mesh(im, i_source_mesh_without_i, idx_ground) *
                               source[2] * impedances[coupling_conn])
                # add the known source current to the grid currents
                grid_currents[i_source_mesh_without_ground] = -1 * source[2]
                remove_i_source_meshes_idx.append(mesh_list.index(coupled_meshes[0]))
                i_sources_mesh_currents_dict[i_source_mesh_without_ground] = source[2]

    # remove the i_source meshes, their couplings are evaluated above
    mesh_list = [x for i, x in enumerate(mesh_list) if i not in remove_i_source_meshes_idx]
    mask = np.ones(V.shape[0], dtype=bool)
    mask[remove_i_source_meshes_idx] = False
    V = V[mask]
    A = np.zeros((V.shape[0], V.shape[0]), dtype=complex)
    num_mesh_currents -= len(remove_i_source_meshes_idx)

    # assign impedances to Resistance-Matrix
    for i, mesh in enumerate(mesh_list):

        # set summed values on diagonal, but remove i source values
        row_indices, col_indices = zip(*mesh)
        A[i, i] = np.sum(impedances[row_indices, col_indices])

        # get coupled mesh currents
        mesh_only_branches = [m for m in mesh if m[1] == idx_ground]
        for connection in mesh_only_branches:
            # these are the coupled meshes, which are connected to the current mesh
            meshes_with_connections = [lst for lst in orig_meshes if connection in lst and lst != mesh]
            # check each coupled mesh
            for connected_mesh in meshes_with_connections:
                if connected_mesh in mesh_list:
                    index_mesh_current = mesh_list.index(connected_mesh)
                    # add coupling impedance
                    A[i, index_mesh_current] = get_sign_in_coupled_mesh(mesh, connected_mesh, idx_ground) * \
                                               impedances[connection]

    # add meshes which hold the v_sources
    v_source_offset = 0
    for vs in voltage_sources:
        # get connected branch and meshes
        branch_v_source = (vs[1], idx_ground)
        connected_meshes = [m for m in mesh_list if branch_v_source in m]
        v_source_idx = num_mesh_currents + v_source_offset - 1
        for connected_mesh in connected_meshes:
            idx_mesh_current = mesh_list.index(connected_mesh)
            A[idx_mesh_current, v_source_idx] = - impedances[branch_v_source]  # all meshes are downstream from v_source
            A[v_source_idx, idx_mesh_current] = - impedances[branch_v_source]
        A[v_source_idx, v_source_idx] = impedances[branch_v_source] + vs[2]
        V[v_source_idx] = vs[0]
        v_source_offset += 1

    # Solve A * I = V
    sparse_matrix = csc_matrix(A)
    mesh_currents = spsolve(sparse_matrix, V)  # mesh_currents = np.linalg.solve(A, V) not efficient for sparse mat.

    slack_currents = np.zeros((len(voltage_sources)), dtype=complex)
    for idx in range(len(voltage_sources)):
        v_source_idx = num_mesh_currents + idx - 1
        slack_currents[idx] = mesh_currents[v_source_idx]

    for idx, mesh in enumerate(mesh_list):

        # remove branch connections
        connections = [m for m in mesh if m[1] != idx_ground]
        if len(connections) == 1:
            grid_currents[connections[0]] = mesh_currents[idx]
        else:
            # this is the case for merged meshes, which contain 1 ideal current source (more than 1 tdb)
            conn_without_superposed_currents = [c for c in connections if c not in i_sources_mesh_currents_dict.keys()][0]
            # here a current source was introduced
            currents_along_common_path = []
            # identify current source
            mesh_i_source_keys = [k for k in i_sources_mesh_currents_dict.keys() if k in mesh][0]
            common_connection = [k for k in i_sources_mesh_currents_dict.keys() if k in connections][0]

            # add the known i source currents to the equation
            currents_along_common_path.append((conn_without_superposed_currents, mesh_currents[idx]))
            currents_along_common_path.append((common_connection, i_sources_mesh_currents_dict[mesh_i_source_keys]))

            grid_currents[common_connection] = superpose_merged_mesh_currents(currents_along_common_path, common_connection)
            grid_currents[conn_without_superposed_currents] = mesh_currents[idx]

    return grid_currents, slack_currents


# region post-processing
def calc_voltages_along_path(node, voltage_at_node, v_deltas, target_v_mat):
    indices_conn_nodes = np.where((np.abs(v_deltas[node, :]) > 0) & (np.arange(len(v_deltas[node, :])) > node))[0]
    for n in indices_conn_nodes:
        target_v_mat[n] = voltage_at_node - v_deltas[node, n]
        calc_voltages_along_path(n, target_v_mat[n], v_deltas, target_v_mat)


def calc_grid_voltages(bus_impedances, act_grid_currents, slack_0_voltage, act_slack_current, copy_voltages=None):
    current_mat = np.zeros_like(bus_impedances)
    for k in act_grid_currents.keys():
        current_mat[k[0], k[1]] = act_grid_currents[k]
        current_mat[k[1], k[0]] = -act_grid_currents[k]

    all_v_deltas = bus_impedances * current_mat
    V = np.zeros((bus_impedances.shape[0] - 1), dtype=complex)
    V[slack_0_voltage[1]] = slack_0_voltage[0] - slack_0_voltage[2] * act_slack_current
    calc_voltages_along_path(slack_0_voltage[1], V[slack_0_voltage[1]], all_v_deltas, V)
    if copy_voltages:
        for c_v in copy_voltages:
            V[c_v[1]] = V[c_v[0]]
    return V

# endregion
