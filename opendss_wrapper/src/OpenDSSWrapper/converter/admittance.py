import numpy as np
import pandapower as pp

from OpenDSSWrapper.converter.components import get_all_grid_components, ExternalGrid, Transformer, Switch
from OpenDSSWrapper.converter.grids import BaseGrid
from tqdm import tqdm

def get_y_matrices_per_frequency(net: pp.pandapowerNet, vt_kv: float | int,
                                 vn_kvs: list[float],
                                 frequencies_hz: list[int],
                                 shunt_position: int | None = -1, compact=False,
                                 include_slack=True, zero_seq=False) -> dict[int, np.ndarray]:
    """
    For each frequency, get Y Matrices and return as dictionary
    :param net: pandapower net
    :param vt_kv: target voltage level in kV
    :param vn_kvs: list of base voltages per node in kV
    :param frequencies_hz: list of frequencies in Hz
    :param shunt_position: position of shunt admittance in the matrix. If None, shunt admittance is not saved
    :param compact: if True, shunt admittance is included in diagonal which is the sum of all off-diagonal elements.
    Off-diagonal elements are negative, while diagonal elements are positive.
    :param include_slack: The shunt admittance of the slack can be neglected if needed
    :param zero_seq: Calculate the Zero sequence admittances
    :return: dictionary, key = frequency, value = admittance matrix
    """
    res = {}
    for f in tqdm(frequencies_hz):
        res[f] = get_y_matrix_extended(net, vt_kv, vn_kvs, f, shunt_position=shunt_position,
                                       compact=compact, include_slack=include_slack, zero_seq=zero_seq)
    return res


def get_y_matrix_extended(net: pp.pandapowerNet, vt_kv, vn_kvs, f_hz,
                          shunt_position=None,
                          compact=False,
                          include_slack=True,
                          zero_seq=False) -> np.ndarray:
    """
    Get the extended admittance matrix for a given frequency, extended by the shunt admittances in respect to one
    voltage-level
    :param net: pandapower net
    :param vt_kv: target voltage level in kV to reference admittances to
    :param vn_kvs: nominal voltage levels in kV of all busses
    :param f_hz: frequency in Hz
    :param shunt_position: position of shunt admittance in the matrix. If None, shunt admittance is not saved.
    :param compact: if True, diagonal is the sum of all off-diagonal elements. Shunt admittance is added to the diagonal
    :param include_slack: if False, the internal admittance of the source is not added as shunt to the bus.
    :param zero_seq: if True, the zero sequence admittances are computed.
    if the parameter shunt_position is set to anything but None.
    :return: extended admittance matrix with shunt admittance at last row / column
    """
    n_bus = len(net.bus) + 1
    matrix = np.zeros((n_bus, n_bus), dtype=np.complex_)
    grid_components = get_all_grid_components(net, vn_kvs)

    for component in grid_components:
        if not isinstance(component, ExternalGrid):
            pi_longitudinal = component.get_pi_longitude_admittance(vt_kv, f_hz, zero_seq)
            matrix[component.from_bus][component.to_bus] = pi_longitudinal
            matrix[component.to_bus][component.from_bus] = pi_longitudinal
            # calculate shunt admittance
            pi_branch_vs_ground = component.get_pi_branch_admittance(vt_kv, f_hz, zero_seq)
            matrix[component.from_bus][-1] += pi_branch_vs_ground
            matrix[-1][component.from_bus] += pi_branch_vs_ground
            # on both sides of the line/transformer/...
            if not (zero_seq and isinstance(component, Transformer)):
                matrix[component.to_bus][-1] += pi_branch_vs_ground
                matrix[-1][component.to_bus] += pi_branch_vs_ground
        elif include_slack:
            matrix[component.from_bus][component.to_bus] += component.get_pi_longitude_admittance(vt_kv,
                                                                                                  f_hz, zero_seq)
            matrix[component.to_bus][component.from_bus] += component.get_pi_longitude_admittance(vt_kv,
                                                                                                  f_hz, zero_seq)

    matrix = filter_y_mat_shunt_and_convert_to_nap(matrix, n_bus, compact, shunt_position)

    return matrix


def filter_y_mat_shunt_and_convert_to_nap(matrix, n_bus, compact=True, shunt_position=None):
    """
    This method converts the weighted adjacency matrix to the matrix used for node potential analysis and removes or
    shifts the row/columns, which is holding the admittances against ground
    :param compact: if True, diagonal is the sum of all off-diagonal elements.
    :param n_bus: amount of busses in network
    :param matrix: weighted adjacency matrix
    :param shunt_position: position of shunt admittance in the matrix
    """
    if compact:
        for i in range(n_bus - 1):
            for j in range(n_bus - 1):
                if i != j:
                    matrix[i][i] += matrix[i][j]
                    matrix[i][j] = -matrix[i][j]
            if shunt_position is not None:
                matrix[i][i] += matrix[i][-1]
    # move shunt admittance from last row / column to desired position
    if shunt_position and not compact:
        # Swap the rows
        matrix[[shunt_position, -1]] = matrix[[-1, shunt_position]]
        # Swap the columns
        matrix[:, [shunt_position, -1]] = matrix[:, [-1, shunt_position]]
    else:
        matrix = np.delete(matrix, -1, 0)
        matrix = np.delete(matrix, -1, 1)
    return matrix


def get_y_pu(net: pp.pandapowerNet) -> np.ndarray:
    """
    gets Admittance Matrix Y in per-unit from pandapower-object
    documentation at https://github.com/e2nIEE/pandapower/blob/develop/tutorials/internal_datastructure.ipynb:
    "Note that the nodal point admittance matrix is given in per unit values."
    "If you just call runpp() the buses are fused internally and the Ybus matrix has a lower dimension.
    You can also use the parameter r_switch to set impedance values for switches.
    In this case, the buses are not fused and the Ybus matrix has a higher dimension.
    This also changes your power flow result."
    :param net: pandapower network
    :return: Admittance Matrix Y in per-unit as numpy array
    """

    # modify switches so that they don't get fused internally and we get the full ybus matrix
    net.switch['z_ohm'] = 0.1  # This significantly changes the outcome of the power flow calculation
    # check if the network has been simulated at least once
    if net._ppc is None:
        pp.runpp(net, numba=False)
    print(net._pd2ppc_lookups["bus"])
    return np.array(net._ppc['internal']['Ybus'].todense().tolist())


def convert_y_to_original(y_pu: np.ndarray, base_kvs: np.ndarray, sn_mva: float | int, mode="p2p") -> np.ndarray:
    """
    Convert an admittance matrix from per-unit to original values in Siemens.
    Behaves the same as the function convert_y_to_pu called with base_kvs = 1 / base_kvs
    :param y_pu: The admittance matrix in per-unit.
    :param base_kvs: An array of base voltages in kV for each bus (given as p2p)
    :param sn_mva: The system's nominal power in MVA.
    :param mode: If "p2p", admittance is returned as phase-to-phase values. If "p2n", admittance is returned as line-to-neutral values.
    :return: The admittance matrix in original values (Siemens).
    """
    return convert_y_to_pu(y_pu, 1 / base_kvs, sn_mva, mode)


def convert_y_to_pu(y_original: np.ndarray, base_kvs: np.ndarray, sn_mva: float | int, mode="p2p") -> np.ndarray:
    """
    Convert an admittance matrix from original values in Siemens to per-unit.
    :param y_original: The admittance matrix in original values (Siemens).
    :param base_kvs: An array of base voltages in kV for each bus. (given as p2p)
    :param sn_mva: The system's nominal power in MVA.
    :param mode: If "p2p", admittance is returned as phase-to-phase values. If "p2n", admittance is returned as line-to-neutral values.
    :return: The admittance matrix in per-unit.
    """
    if mode not in ["p2p", "p2n"]:
        raise ValueError("mode must be 'p2p' or 'p2n'")
    shape = y_original.shape
    y_pu = np.empty(shape, dtype=np.complex128)

    for i in range(shape[0]):
        for j in range(shape[1]):
            factor = 1
            if mode == "p2n":
                factor = 1 / np.sqrt(3)
            base_kv_f = base_kvs[i] * factor  # from
            base_kv_t = base_kvs[j] * factor  # to
            if base_kv_f > base_kv_t:
                base_kv_t = base_kv_f
            elif base_kv_t > base_kv_f:
                base_kv_f = base_kv_t
            z_base = (base_kv_t * base_kv_f) / sn_mva  # Reference impedance
            y_base = 1 / z_base  # Reference admittance
            y_pu[i, j] = y_original[i, j] / y_base

    return y_pu


def compare_matrices(grid: BaseGrid):
    base_kvs_dict = grid.get_base_kvs()
    base_kvs = np.array(list(base_kvs_dict.values()))

    f_nominal_hz = grid.pp_net.f_hz if hasattr(grid.pp_net, 'f_hz') else 50

    ## Own Calculation
    admittance_own_calc_npa = get_y_matrix_extended(grid.pp_net, 20, base_kvs, f_nominal_hz, compact=True)
    admittance_own_calc_npa2 = get_y_matrix_extended(grid.pp_net, base_kvs, base_kvs, f_nominal_hz, compact=True)

    ## PandaPower
    # bus numbers are ordered in the same way as in the admittance matrices, can simply convert to array:
    base_kvs_own_calc = np.ones(base_kvs.shape)
    base_kvs_own_calc[base_kvs == 0.4] = 0.4 / 20
    pp_net = grid.pp_net
    pp_pu = get_y_pu(pp_net)

    # reconstruct original admittance matrices from per-unit (in Siemens)
    pp_original = convert_y_to_original(pp_pu, base_kvs, pp_net.sn_mva)
    pass


if __name__ == '__main__':
    from OpenDSSWrapper.converter.grids import CigreLVDistPV, BaseGrid

    grid = CigreLVDistPV()
    components = get_all_grid_components(grid.pp_net, np.array(grid.get_base_kvs().values()), branch_only=True)
    vn_kvs = list(grid.get_base_kvs().values())
    frequencies_hz = list(np.arange(50, 1050, 50))
    y_mats = get_y_matrices_per_frequency(grid.pp_net, 20., vn_kvs, frequencies_hz, compact=True)

    pass
