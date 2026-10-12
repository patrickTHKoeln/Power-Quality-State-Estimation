import cmath
import logging
import math

import numpy as np
import pandapower
import pandas as pd
from tqdm import tqdm

from OpenDSSWrapper.converter import get_all_grid_components, Transformer, ExternalGrid
from OpenDSSWrapper.converter.admittance import get_y_matrices_per_frequency

logger = logging.getLogger(__name__)


def components_to_polar(complex_num):
    """
    Converts a complex number to polar coordinates.
    Angle is returned in degrees.
    :param complex_num: complex number
    :return: magnitude, angle (in degree)
    """
    return abs(complex_num), np.angle(complex_num, True)


def get_voltage_ln_complex(vln_kv: float, angle_ln: float, vn_kv: float, vt_kv: float) -> complex:
    """
    Compute and return a complex representation of a phasor value of Line-Neutral Voltage.

    :param vln_kv (float): Line-Neutral Voltage in kilovolts (actual voltage of the element)
    :param angle_ln (float): Angle of the phasor in degrees.
    :param vn_kv (float): Base voltage of the system in kilovolts.
    :param vt_kv (float): Target voltage level for which the calculation is being done in kilovolts.

    :return complex: Complex phasor representation of the Line-Neutral voltage.
    """

    angle_to_pi = np.deg2rad(angle_ln)
    voltage_ratio = vln_kv * (vt_kv / vn_kv)
    real = voltage_ratio * np.cos(angle_to_pi)
    imag = voltage_ratio * np.sin(angle_to_pi)

    return complex(real, imag)


def get_voltages_ln_complex(vln_kvs: np.array, angles_ln: np.array, vn_kvs: np.array,
                            vt_kv: float) -> np.array:
    """
    Array wise operation to speed up processing time
    :param vln_kvs (np.array): Line-Neutral Voltages in kilovolts.
    :param angles_ln (np.array): Angle of the phasor in degrees.
    :param vn_kvs (np.array): Base voltage of the system in kilovolts.
    :param vt_kv (float): Voltage level for which the calculation is being done in kilovolts.

    :return complex: Complex phasor representation of the Line-Neutral voltage.
    """
    angles_to_pi = np.deg2rad(angles_ln)
    voltage_ratios = vln_kvs * (vt_kv / vn_kvs)

    return voltage_ratios * np.exp(1j * angles_to_pi)


class PostProcessor:
    def __init__(self, net: pandapower.pandapowerNet, vn_kv_per_node: dict[str, float],
                 vt_kv: float = 20.):
        """
        Post Processor for OpenDSS simulation results
        :param net: pandapower network
        :param vn_kv_per_node: base voltage per node as dictionary with node id as key and base voltage in kV as value
        :param vt_kv: target voltage level in kV
        """
        self.net = net
        self.vt_kv = vt_kv
        self.grid_elements = []
        self.node_voltage_vectors = {}
        self.x_solved = []
        self.y_target = []
        self.vn_kv_per_node = vn_kv_per_node
        self.Y_matrices_per_frequency = {}
        self.vt_ln_v = 1000 * vt_kv / math.sqrt(3)  # target voltage level line-neutral in v
        self.grid_elements = get_all_grid_components(net, np.array(net['bus']['vn_kv']))

        # Get all voltage Levels in the grid
        # To eliminate the vector group, rotate all LV Voltages to the MV level
        # Low Level measurements are rotated by the vector group before computing the network
        for element in self.grid_elements:
            if isinstance(element, Transformer):
                self._add_voltage_vector_to_bus(element.from_bus, element.df['vn_hv_kv'])
                self._add_voltage_vector_to_bus(element.to_bus, element.df['vn_lv_kv'])
            elif isinstance(element, ExternalGrid):
                self._add_voltage_vector_to_bus(element.from_bus, element.vn_kv)
            else:
                self._add_voltage_vector_to_bus(element.from_bus, element.vn_kv)
                self._add_voltage_vector_to_bus(element.to_bus, element.vn_kv)

    def unify_voltage_level(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bring all nodes to the same voltage level
        :param data: Data as a multi-indexed pandas Dataframe
        :return: data in the same format with unified voltage bases
        """
        voltage_columns = ['v1', 'v2', 'v3']
        current_columns = ['i1', 'i2', 'i3']
        angle_columns = ['vangle1', 'vangle2', 'vangle3', 'iangle1', 'iangle2', 'iangle3']
        # Ensure columns are in lowercase
        data.columns = map(str.lower, data.columns)
        # Perform operations vectorized over groups by 'bus'
        logger.info(
            f"Unifying voltage level of all nodes to {self.vt_kv}, node base voltages: {self.vn_kv_per_node}, lead lag vectors: {self.node_voltage_vectors}")
        for bus, df_group in tqdm(data.groupby(level='bus'),
                                  desc=f"Unifying voltage level of all nodes to {self.vt_kv}"):
            base_kV = self.vn_kv_per_node[str(bus)]
            # Scale voltages
            for col in voltage_columns:
                if col in df_group:
                    data.loc[(slice(None), bus), col] *= (self.vt_kv / base_kV)
            # Scale currents
            for col in current_columns:
                if col in df_group:
                    data.loc[(slice(None), bus), col] *= (base_kV / self.vt_kv)
            # Adjust angles
            # self._convert_lead_lag_angle(data, bus, df_group)
        return data

    def to_pu(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data to per unit
        :param data: data as multi-indexed pandas DataFrame
        :return: data in per unit and same format
        """
        # if for some reason, step does not exist in the index yet, add it and initialize it to 0
        if 'step' not in data.index.names:
            tuples = [(0, *rest) for rest in data.index]
            data.index = pd.MultiIndex.from_tuples(tuples,
                                                   names=['step'] + data.index.names if data.index.names else ['step'])

        voltage_columns = ['v1', 'v2', 'v3']
        current_columns = ['i1', 'i2', 'i3']
        # Ensure columns are in lowercase
        data.columns = map(str.lower, data.columns)

        logger.info(
            f"Converting data to per unit, node base voltages: {self.vn_kv_per_node}, lead lag vectors: {self.node_voltage_vectors}")
        # Perform operations vectorized over groups by 'bus'
        for bus, df_group in data.groupby(level='bus'):
            base_kV = self.vn_kv_per_node[str(bus)]
            v_base = base_kV / np.sqrt(3) * 1000
            s_base = 1e6  # assume base power of 1 MW
            i_base = s_base / (base_kV * 1000)
            # Scale voltages
            for col in voltage_columns:
                if col in df_group:
                    data.loc[(slice(None), bus), col] /= v_base
            # Scale currents
            for col in current_columns:
                if col in df_group:
                    data.loc[(slice(None), bus), col] /= i_base
            # Adjust angles
            self._convert_lead_lag_angle(data, bus, df_group)
        return data

    def _convert_lead_lag_angle(self, data, bus, df_group):
        """
        Lead lag conversion based on voltage level
        TODO: at the moment static 30° between MV and LV, see issue 10
        :param data:
        :param bus:
        :param df_group:
        :return:
        """
        angle_columns = ['vangle1', 'vangle2', 'vangle3', 'iangle1', 'iangle2', 'iangle3']
        for col in angle_columns:
            if col in df_group:
                angles_rad = np.deg2rad(df_group[col])
                complex_numbers = np.exp(1j * angles_rad)
                r = complex_numbers / self.node_voltage_vectors[str(bus)]["vectorGroup"]
                data.loc[(slice(None), bus), col] = np.rad2deg(np.angle(r))

    def _add_voltage_vector_to_bus(self, bus: int, v_level_in_kV: float):
        """
        Rotate all LV voltages to the MV level
        :param bus: bus id
        :param v_level_in_kV: voltage level in kV
        :return:
        """

        if str(bus) not in self.node_voltage_vectors:
            if not np.isclose(v_level_in_kV, self.vt_kv):
                vector_group = cmath.rect(1, np.deg2rad(-30))
            else:
                vector_group = cmath.rect(1, np.deg2rad(0))

            new_item = {"vectorGroup": vector_group}

            self.node_voltage_vectors[f"{bus}"] = new_item

    def _pre_calculations_worker(self, data_at_step, known_elements):
        """
        Prepare data
        :param data_at_step: target data at time step i
        :param known_elements: known elements, i.e. measurement buses
        :return:
        """
        n_nodes = len(list(self.Y_matrices_per_frequency.values())[0][0]) - 1
        x_solved = {}
        # convert data_at_step to dict of frequencies
        data_complex = {}
        # pre-calculations for now only considers one phase
        data_p1 = {}
        for freq, _ in self.Y_matrices_per_frequency.items():
            data_complex[freq] = {}
            data_p1[freq] = {}
            for bus, df in data_at_step.items():
                data_complex[freq][bus] = {}
                # set voltage level to 1 to prevent any changes (should be done prior)
                freq_row = df[df['freq'] == freq]
                data_complex[freq][bus]['v1'] = get_voltage_ln_complex(freq_row['v1'], freq_row['vangle1'], 1, 1)
                data_complex[freq][bus]['v2'] = get_voltage_ln_complex(freq_row['v2'], freq_row['vangle2'], 1, 1)
                data_complex[freq][bus]['v3'] = get_voltage_ln_complex(freq_row['v3'], freq_row['vangle3'], 1, 1)
                data_p1[freq][bus] = data_complex[freq][bus]['v1']

        for freq, y_mat_ext in self.Y_matrices_per_frequency.items():
            x_solved[freq] = self._pre_calculations(freq, y_mat_ext, n_nodes, data_p1[freq], known_elements)

        return x_solved

    def _pre_calculations(self, freq: float, y_mat_ext, n_nodes: int, y_target_freq, known_elements):
        """
        Primitive solving algorithm for given frequency
        :param freq: frequency
        :param y_mat_ext: extended admittance matrix at specified frequency
        :param n_nodes: amount of nodes
        :param y_target_freq: target voltages at specified frequency
        :param known_elements: known elements, i.e. measurement buses
        :return:
        """
        # region Knotengleichungen allgemein aufstellen:
        admittance_matrix_to_solve = np.zeros((n_nodes, n_nodes), dtype=np.complex_)
        known_currents = np.zeros(n_nodes, dtype=np.complex_)

        for bus in range(0, n_nodes):
            node_equation = np.zeros(n_nodes, dtype=np.complex_)
            # Erde (0) ist der Bezugs-Knoten
            # Leitwerte am Knoten ermitteln
            sum_at_node = complex(0, 0)

            for n in y_mat_ext[bus]:
                sum_at_node += n

            node_equation[bus] = sum_at_node

            # Koppel-Leitwerte ermitteln
            for n in range(0, len(y_mat_ext[bus]) - 1):
                # Werte, die in der Zeile auftauchen, aber nicht mit nem Knotenpotential der
                # Spalte übereinstimmen finden
                if n != bus:
                    node_equation[n] = -1 * y_mat_ext[bus][n]

            admittance_matrix_to_solve[bus] = node_equation

        # update Angle of Slack and update currents-vector
        if freq == self.net['f_hz']:
            for src in [x for x in self.grid_elements if isinstance(x, ExternalGrid)]:
                known_currents[src.from_bus] += src.get_i0_at_voltage_level(
                    self.vt_kv / math.sqrt(3))
                # (x_solved[freq][src.from_bus])

        # region Gleichungssystem lösen

        x_solved = np.linalg.solve(admittance_matrix_to_solve, known_currents)
        # get critical res_freqs?

        # reset currents
        known_currents = [0 for _ in range(n_nodes)]
        measured_voltages = [0 for _ in range(n_nodes)]

        for nn in known_elements:
            # get voltage vector for voltage Level and rotate measurement to the correct voltage Level
            measured_voltages[nn] = y_target_freq[nn] / self.node_voltage_vectors[str(nn)][
                "vectorGroup"]

            # bekannte Spannungen in Strom umrechnen
            for row in range(0, len(admittance_matrix_to_solve)):
                known_currents[row] -= admittance_matrix_to_solve[row][nn] * measured_voltages[nn]
                # Leitwert aus der Matrix entfernen
                admittance_matrix_to_solve[row][nn] = complex(0, 0)

        # let the slack angle converge
        if np.isclose(freq, self.net['f_hz'], rtol=1e-3):
            x_solved, known_currents = self._slack_angle_converge(x_solved, known_currents,
                                                                  admittance_matrix_to_solve)
        # endregion

        # region Postprocessing
        # add known voltages to the solution

        for nn in known_elements:
            x_solved[nn] = measured_voltages[nn]

        # Rotate all voltages according to DYN11 back to the actual angle
        for bus_voltage in range(0, len(x_solved)):
            x_solved[bus_voltage] = x_solved[bus_voltage] * \
                                    self.node_voltage_vectors[str(bus_voltage)][
                                        "vectorGroup"]
        # endregion
        return x_solved

    def _slack_angle_converge(self, x_solved, known_currents, admittance_matrix_to_solve):
        save_current = complex(0, 0)
        current0_converged = False
        for iterations in range(0, 100):

            # update Angle of Slack and update currents-vector

            for src in [x for x in self.grid_elements if isinstance(x, ExternalGrid)]:
                voltage_n_minus_one = x_solved[src.from_bus] * 0.001
                # print(str(iterations) + "-" + str(voltage_n_minus_one))
                known_currents[src.from_bus] += src.get_i0_at_voltage_level(voltage_n_minus_one)
                delta_real = abs(save_current.real - known_currents[src.from_bus].real)
                delta_imag = abs(save_current.imag - known_currents[src.from_bus].imag)
                if delta_real < 0.01 and delta_imag < 0.01:
                    current0_converged = True
                    break
                else:
                    save_current = known_currents[src.from_bus]

            if current0_converged:
                break

            # solve the non-squared matrix
            x_solved = np.linalg.lstsq(admittance_matrix_to_solve, known_currents, rcond=None)[0]
            for src in [x for x in self.grid_elements if isinstance(x, ExternalGrid)]:
                known_currents[src.from_bus] -= save_current

        if not current0_converged:
            raise RuntimeError("Current did not converge")
        return x_solved, known_currents

    def pre_calculations_wrapper(self, data: dict, start_index: int, stop_index: int, step: int,
                                 known_elements: list[int], frequencies_hz: list[float] = None):
        """
        Wrap pre_calculations_worker in tqdm progress bar and allows for better compilation (native multiprocessing)
        :param data: target data
        :param start_index: start index
        :param stop_index: stop index
        :param step: step size
        :param known_elements: list of known elements, i.e. measurement buses
        :param frequencies_hz: list of frequencies in Hz; if None, fundamental frequency is used
        :return:
        """
        if frequencies_hz is None:
            frequencies_hz = [self.net['f_hz']]
        self.Y_matrices_per_frequency = get_y_matrices_per_frequency(self.net, self.vt_kv, list(self.vn_kv_per_node.values()),
                                                                     frequencies_hz=frequencies_hz)
        results = {}
        for i in tqdm(range(start_index, stop_index, step), total=int((stop_index - start_index) / step)):
            result = self._pre_calculations_worker(data[i], known_elements)
            if result is not None:
                results[i] = result
        return results
