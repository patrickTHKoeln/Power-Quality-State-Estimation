import logging
import math
import os
from enum import Enum
from typing import Sequence

import numpy as np
import opendssdirect as dss
import pandas as pd
from tqdm import tqdm

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.harmonic_spectra import Spectrum, DynamicSpectrum
from OpenDSSWrapper.simulation import DataStore
from OpenDSSWrapper.tools import config as cfg
from OpenDSSWrapper.tools import find_missing_files
from OpenDSSWrapper.tools.logging import DebugMode, SIM_DEBUG_MODE, SIM_DEBUG_PLOT

logger = logging.getLogger(__name__)
REQUIREDFILES = [
    'Master.dss',
    'LineCodes.dss',
    'Lines.dss',
    'Loads.dss',
    'Renewables.dss',
    'Monitors.dss',
    'BusCoords.dss'
]
TRANSIENTNAMESOURCE = "transient_source"
TRANSIENTNAMELOAD = "transient_load"
TRANSIENTLCURVE = "transient_l_curve"
TRANSIENTRCURVE = "transient_r_curve"


class SimulationMode(Enum):
    POWERFLOW = 1
    HARMONICS = 2
    TRANSIENTS = 3
    FLICKERS = 4


class SolutionMode(Enum):
    """
    Solution mode according to OpenDSS Direct Implementation
    """
    Snap = 0  # Solve a single snapshot power flow for the present conditions.
    Daily = 1
    Yearly = 2
    M1 = 3
    LD1 = 4
    PeakDay = 5
    DutyCycle = 6
    Direct = 7  # Solve a single snapshot solution using an admittance model of all loads. This is non-iterative;
    MF = 8
    FaultStudy = 9
    M2 = 10
    M3 = 11
    LD2 = 12
    AutoAdd = 13
    Dynamic = 14
    Harmonic = 15
    Time = 16
    HarmonicT = 17


class LoadModel(Enum):
    """
    See constants: https://dss-extensions.org/dss-extensions/pascal/DSSGlobals.html
    """
    POWERFLOW = 1
    ADMITTANCE = 2


class ElementType(Enum):
    LINE = 0
    LOAD = 1
    GENERATOR = 2
    VSOURCE = 3


def enable_transient_source(enable: bool):
    """
    Enable or disable the transient source
    :param enable: boolean to connect or disconnect transient error-source
    :return:
    """
    str_enable = "Enable"
    if not enable:
        str_enable = "Disable"
    # Add/Remove transient on bus for further Simulation steps
    if TRANSIENTNAMESOURCE in str([dss.Vsources.AllNames()]):
        if enable:
            dss.Circuit.Enable(TRANSIENTNAMESOURCE)
        else:
            dss.Circuit.Disable(TRANSIENTNAMESOURCE)
        send_text_command(str_enable + ' vsource.' + TRANSIENTNAMESOURCE + '')

    # Enable or disable the transient fundamental damping as reactor.
    if TRANSIENTNAMELOAD in str([dss.Reactors.AllNames()]):
        if enable:
            dss.Circuit.Enable(TRANSIENTNAMELOAD)
        else:
            dss.Circuit.Disable(TRANSIENTNAMELOAD)
        send_text_command(str_enable + ' Reactor.' + TRANSIENTNAMELOAD + '')


def read_frequencies_from_opendss():
    """
    Reads the frequencies from a monitor object, which were calculated by OpenDSS internally.
    A harmonics solution with spectra must be performed beforehand.
    :return: frequencies list
    """
    if dss.Monitors.Count()>0:
        dss.Monitors.Idx(1)

        header = dss.Monitors.Header()
        return dss.Monitors.AsMatrix()[:,0].tolist()
    else:
        return []


def send_text_command(message: str):
    """
    Send a text command to the OpenDSS interface
    :param message: Message to send
    :return:
    """
    dss.Text.Command(message)
    # dss.dss_lib.Text_Set_Command(string.encode())
    logger.debug(f"Sent message to OpenDSS: {message}")
    logger.debug(f"Result: {dss.Text.Result()}")


def _get_voltage_bases():
    """
    Get the voltage bases of all buses (in V, line to line)
    WARNING: this method does not work with external multiprocessing
    :return: dictionary with bus name as key and voltage base as value
    """
    results = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        results[bus_name] = dss.Bus.kVBase() * 1000 * math.sqrt(3)
    logger.debug(f"Voltage bases of all busses: {results}")
    return results


def get_properties():
    """
    Convert the opendss network to pandas dataframes, saved in a dictionary with type as key.
    Does not work if there is no active circuit (e.g. if the elements were created in different processes using multiprocessing)
    :return: dictionary with type as key and dataframe as value
    """
    results = {}
    from dss import DSSException
    try:
        dss.Circuit.Name()
    except DSSException:
        logger.warning("Attempting to get OpenDSS properties: No active circuit")
        return results
    results['loads'] = dss.utils.class_to_dataframe("Load")
    results['lines'] = dss.utils.class_to_dataframe("Line")
    results['transformers'] = dss.utils.class_to_dataframe("Transformer")
    results['generators'] = dss.utils.class_to_dataframe("Generator")
    results['pvsystems'] = dss.utils.class_to_dataframe("PVsystem")
    results['vsources'] = dss.utils.class_to_dataframe("Vsource")
    results['monitors'] = dss.utils.class_to_dataframe("Monitor")
    # results['bus'] = dss.utils.to_dataframe(dss.Bus)  # dss.utils.class_to_dataframe("Bus") does not work here
    results['capacitors'] = dss.utils.class_to_dataframe("Capacitor")
    results['fuses'] = dss.utils.class_to_dataframe("Fuse")
    results['spectrum'] = dss.utils.class_to_dataframe("Spectrum")
    results['linecodes'] = dss.utils.class_to_dataframe("LineCode")
    # 3-phase sparse matrix returned as tuple
    ymatrix = dss.YMatrix.getYsparse(True)
    from scipy.sparse import csc_matrix
    results['ymatrix_desparsified'] = csc_matrix(ymatrix)
    node_names = dss.Circuit.AllNodeNames()
    phase_indices = {i: [] for i in range(3)}
    for i, name in enumerate(node_names):
        phase = int(name.split('.')[1]) - 1
        phase_indices[phase].append(i)

    for phase in phase_indices.keys():
        indices = sorted(phase_indices[phase], key=lambda i: int(node_names[i].split('.')[0]))
        matrix_phase = results['ymatrix_desparsified'][indices, :][:, indices]
        results['ymatrix_desparsified_phase_' + str(phase + 1)] = pd.DataFrame(matrix_phase.A)
    results['ymatrix_desparsified'] = pd.DataFrame(results['ymatrix_desparsified'].A)
    return results


def filter_harmonics_in_dataframe(df: pd.DataFrame, target_orders: list[float],
                                  tolerance: float = 1e-4) -> pd.DataFrame:
    """
    Filters out harmonic orders from the dataframe that are not within a specified tolerance of the target orders.
    :param df: DataFrame containing a column 'harmonic' with harmonic orders.
    :param target_orders: List of harmonic orders to retain.
    If target_orders is None, the whole dataframe is returned without modification.
    If target_orders is an empty list, the resulting dataframe will be empty.
    :param tolerance: Tolerance for comparing harmonic orders.
    :return pd.DataFrame: DataFrame with only the rows where harmonic orders match the specified list within the given tolerance.
    """
    if target_orders is None:
        return df
    # Create an array where each element is True if the 'harmonic' is close to any value in target_orders
    mask = np.isclose(df['harmonic'].values[:, None], target_orders, atol=tolerance).any(axis=1)
    filtered_df = df[mask]

    # Check if each target order has been matched at least once
    matched_orders = np.unique(filtered_df['harmonic'][
                                   np.isclose(filtered_df['harmonic'].values[:, None], target_orders,
                                              atol=tolerance).any(axis=1)])
    if len(matched_orders) != len(target_orders):
        logger.warning(
            f"Not all specified frequencies were matched. Matched orders: {matched_orders}, target orders: {target_orders}")

    # Return the filtered DataFrame
    return filtered_df


class OpenDSSSimulation:
    def __init__(self, grid: BaseGrid, simulation_mode: SimulationMode,
                 spectra: dict[str, DynamicSpectrum],
                 generator_profiles: dict[str, pd.DataFrame], load_profiles: dict[str, pd.DataFrame],
                 data_store: DataStore,
                 save_path: str = None, convert=True, step_size_ms: int = 60000):
        """
        :param grid: BaseGrid object
        :param simulation_mode: SimulationMode
        :param spectra: dictionary of varying load spectra for each component with harmonic spectra
        :param generator_profiles: dictionary of generator profiles
        :param load_profiles: dictionary of load profiles
        :param data_store: DataStore object for saving results
        :param save_path: path to openDSS script files
        :param convert: if True convert pandapower net to opendss, else look for the net at save_path
        """
        self.data_store = data_store
        self.debug = SIM_DEBUG_MODE
        if self.debug == DebugMode.VERBOSE:
            logger.warning(
                "Debug mode set to verbose: logging all properties, results and messages to files per simulation step.")
        self.simulation_mode = simulation_mode
        self.grid = grid
        self.save_path = save_path
        self.step_size_ms = step_size_ms
        os.makedirs(save_path, exist_ok=True)
        if convert or any(find_missing_files(save_path, REQUIREDFILES)):
            from OpenDSSWrapper.converter import pandapower_to_opendss
            logger.info("Converting pandapower to openDSS...")
            pandapower_to_opendss(grid.pp_net, grid.name.replace(' ', ''), save_path, monitor_power=cfg.MONITOR_POWER,
                                  monitor_loads=cfg.MONITOR_LOADS, monitor_generators=cfg.MONITOR_GENERATORS)

        self.set_open_dss_grid_scripts()

        self.spectra: dict[str, DynamicSpectrum] = spectra
        # remove all empty spaces from dictionary keys of spectra, generator profiles and load profiles
        # OpenDSS strips empty spaces from names, thus we need to do the same if we want to access profiles and spectra
        self.spectra = {k.replace(' ', '').replace('Load', '').lower(): v for k, v in spectra.items()}
        self.generator_profiles: dict[str, pd.DataFrame] = generator_profiles
        self.generator_profiles = {k.replace(' ', '').lower(): v for k, v in generator_profiles.items()}
        self.load_profiles: dict[str, pd.DataFrame] = load_profiles
        self.load_profiles = {k.replace(' ', '').replace('Load', '').lower(): v for k, v in load_profiles.items()}
        # self.voltage_bases = _get_voltage_bases()
        self.loads = {}
        self.generators = {}
        self.save_default_values()
        self._add_ideal_load_spectrum()

    def set_open_dss_grid_scripts(self):
        """
        Set the openDSS grid scripts
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Grid path {self.save_path} does not exist")
        missing_file_names = find_missing_files(self.save_path, REQUIREDFILES)
        if len(missing_file_names) != 0:
            raise UserWarning(
                "Not all required openDSS files exist at path " + self.save_path
                + ". The following files are missing:" + ", ".join(missing_file_names))
        self.execute_from_file(self.save_path, 'Master.dss')

    def reset_transient_simulation(self, transient_spectrum: Spectrum):
        """
        Resets the simulation to a cleared state and adds the default spectra
        """
        self.set_open_dss_grid_scripts()
        self._add_ideal_load_spectrum()

    def pre_settings_transient_simulation(self, transient_spectrum: Spectrum):
        """
        Sets important consants for a new transient spectrum
        :param transient_spectrum: new transient spectrum for the iteration
        :return:
        """
        self.assign_ideal_load_spectrum()
        self._create_lr_curve(transient_spectrum)

    def _add_ideal_load_spectrum(self):
        """
        Ideal load spectrum should always be known to OpenDSS
        (same as default spectrum 'linear' and 'defaultvsource' but with clearer naming)
        :return:
        """
        ideal_load_spectrum = Spectrum([1.0], [100.0], [0.0])
        send_text_command(ideal_load_spectrum.get_opendss_string('IdealLoad'))

    def assign_ideal_load_spectrum(self):
        """
        Ideal load spectrum shall be set to all loads for Transient Sim
        :return:
        """
        for load_idx in range(1, dss.Loads.Count() + 1):
            dss.Loads.Idx(load_idx)
            load_name = dss.Loads.Name()
            send_text_command(f'Edit Load.{load_name} spectrum=IdealLoad')

    def save_default_values(self):
        """
        Save default load and generator kW and kVar values
        :return:
        """
        for load_idx in range(1, dss.Loads.Count() + 1):
            dss.Loads.Idx(load_idx)
            load_name = dss.Loads.Name()
            self.loads[load_name] = {'kW': dss.Loads.kW(), 'kvar': dss.Loads.kvar()}
        for generator_idx in range(1, dss.Generators.Count() + 1):
            dss.Generators.Idx(generator_idx)
            generator_name = dss.Generators.Name()
            self.generators[generator_name] = {'kW': dss.Generators.kW(), 'kvar': dss.Generators.kvar()}
        for pv_idx in range(1, dss.PVsystems.Count() + 1):
            dss.PVsystems.Idx(pv_idx)
            pv_name = dss.PVsystems.Name()
            self.generators[pv_name] = {'kW': dss.PVsystems.kW(), 'kvar': dss.PVsystems.kvar()}

    def execute_from_file(self, path: str, filename: str):
        """
        Execute a file from the given path
        :param path: path to openDSS file
        :param filename: name of the file
        :return:
        """
        file = open(os.path.join(path, filename), 'r')
        commands = file.readlines()
        for line in commands:
            new_line = line.replace('\n', '')
            if "Redirect" in new_line:
                split_line = new_line.split(' ')
                self.execute_from_file(path, split_line[1])
            else:
                send_text_command(new_line)

    def add_frequency_sweep(self, harmonics: Sequence[float | int] = None, start_frequency: float = None,
                            stop_frequency: float = None, step_frequency: float = None,
                            fund_frequency: float = 50.):
        """
        Either harmonics or start_frequency, stop_frequency and step_frequency must be given
        From Documentation: "Note that to perform a frequency scan of a network,
        you would define a Spectrum object with a small frequency increment and assign it to
        either an Isource or Vsource object, as appropriate."
        :param harmonics: harmonics to include in the spectrum
        :param start_frequency: starting frequency
        :param stop_frequency: stopping frequency
        :param step_frequency: frequency step
        :param fund_frequency: fundamental frequency of the network (magnitude of the fundamental frequency is always 1)
        :return:
        """
        if harmonics is None:
            if start_frequency is None or stop_frequency is None or step_frequency is None:
                raise ValueError("Either harmonics or start_frequency, stop_frequency and step_frequency must be given")
            freq = np.arange(start_frequency, stop_frequency + step_frequency, step_frequency)
            harmonics = freq / fund_frequency

        mag = np.zeros(len(harmonics))
        # set magnitude at fundamental frequency to 1
        mask = np.isclose(harmonics, 1.0)  # fundamental frequency mask
        mag[mask] = 1.0
        phase = np.zeros(len(harmonics))
        # Create Spectrum object
        spectrum = Spectrum(harmonics, mag, phase)
        send_text_command(spectrum.get_opendss_string('freq_sweep'))

        c = dss.Vsources.Count()
        if c > 0:
            dss.Vsources.Idx(1)
            send_text_command(f'Edit Vsource.{dss.Vsources.Name()} spectrum=freq_sweep')
        else:
            dss.Isource.Idx(1)
            send_text_command(f'Edit Isource.{dss.Isource.Name()} spectrum=freq_sweep')

    def update_device(self, device_type: str, step: int, debug_list: list[str]):
        """
        Update the device type (PVsystem, Generator, Load) at the given step
        :param device_type: device type
        :param step: time step
        :param debug_list: list to append debug messages to
        :return:
        """
        device_count_func = getattr(getattr(dss, device_type + 's'), 'Count')
        device_idx_func = getattr(getattr(dss, device_type + 's'), 'Idx')
        device_name_func = getattr(getattr(dss, device_type + 's'), 'Name')

        for idx in range(1, device_count_func() + 1):
            device_idx_func(idx)
            name = device_name_func()
            profiles = self.load_profiles if device_type == 'Load' else self.generator_profiles
            if name not in profiles:  # if no profile is defined, factor is 1
                factor = 1
                debug_list.append(f"Device {name} not found in profiles")
            else:
                # modulo operation to prevent index out of bounds if for some reason profile is shorter than steps (e.g. if steps don't start at 0)
                profile_idx = step%profiles[name].__len__()
                # for now use only one phase. Each phase is normalized to a range 0..1 (except for vsource)
                factor = profiles[name].iloc[profile_idx]['pl1']
                factor = round(max(factor, 0.0001),6)
            if device_type == 'Vsource':
                self.handle_spectrum(name, 0, 0, device_type, debug_list, step, pu=factor)
                debug_list.append(f'Edit {device_type}.{name} pu={str(factor)}')
                dss.Vsources.PU(factor)  # set pu to factor
                continue
            device_kw = self.generators[name]['kW'] if device_type == 'PVsystem' else self.loads[name]['kW']
            device_kvar = self.generators[name]['kvar'] if device_type == 'PVsystem' else self.loads[name]['kvar']
            # update the device values; prevent values below 1e-100 which cause errors in the simulation
            kw_update = round(max(device_kw * factor, 0.001), 6)
            kvar_update = round(max(device_kvar * factor, 0.001), 6)
            if device_type == "PVsystem":
                kva_update = round(max(math.sqrt(kw_update ** 2 + kvar_update ** 2), 0.001), 6)
                device_set_kva = getattr(getattr(dss, device_type + 's'), 'kVARated')
                device_set_kva(kva_update)
                debug_list.append(
                    f'Edit {device_type}.{name} kva={str(kva_update)}')
            else:
                # TODO check if these updates apply to all device types
                device_set_kvar = getattr(getattr(dss, device_type + 's'), 'kvar')
                device_set_kw = getattr(getattr(dss, device_type + 's'), 'kW')
                device_set_kw(kw_update)
                device_set_kvar(kvar_update)
                debug_list.append(
                    f'Edit {device_type}.{name} kW={str(kw_update)} kvar={str(kvar_update)}')

            self.handle_spectrum(name, kw_update, kvar_update, device_type, debug_list, step)

    def handle_spectrum(self, name: str, kw_update: float, kvar_update: float, device_type: str, debug_list: list[str],
                        step, pu=1):
        """
        For the given device type (PVsystem, Generator, Load) update the spectrum
        :param name: device name
        :param kw_update: updated kW value
        :param kvar_update: updated kVar value
        :param device_type: device type
        :param debug_list: list to append debug messages to
        :param pu: pu value for Vsource
        :return:
        """
        if name not in self.spectra:
            debug_list.append(f"Device {name} not found in spectra")
            return
        dyn_spectrum = self.spectra[name]
        # in the database, single phase spectra are saved along with single-phase loads
        # to get the equivalent spectrum for a three-phase load, multiply the load by 1/3rd
        factor = 1 / 3
        if dyn_spectrum.phase == -1:  # for 3-phase spectra, the load is correctly defined
            factor = 1
        # TODO: for single-phase loads (once defined), use single-phase spectrum without factor
        if device_type == 'Load':
            interpolated_spectrum = dyn_spectrum.get_spectrum(load_power_watt=kw_update * 1000 * factor, time_step=step,
                                                              target_step_size_ms=self.step_size_ms)
        elif device_type == 'Generator' or device_type == 'PVsystem':  # power is defined negative for generation in database
            interpolated_spectrum = dyn_spectrum.get_spectrum(load_power_watt=-kw_update * 1000 * factor,
                                                              time_step=step, target_step_size_ms=self.step_size_ms)
        elif device_type == 'Vsource':  # voltage source spectrum load is defined in pu
            interpolated_spectrum = dyn_spectrum.get_spectrum(load_power_watt=pu, time_step=step,
                                                              target_step_size_ms=self.step_size_ms)
        else:
            raise KeyError(f"Device type <{device_type}> unknown.")
        # There might be a DC current in the spectrum (harmonic = 0), however OpenDSS only allows non-zero values
        interpolated_spectrum.harmonics = [cfg.DC_FREQUENCY if np.isclose(x, 0, rtol=1e-5) else x for x in
                                           interpolated_spectrum.harmonics]
        send_text_command(interpolated_spectrum.get_opendss_string(name))
        debug_list.append(f"{interpolated_spectrum.get_opendss_string(name)}")
        send_text_command(f'Edit {device_type}.{name} spectrum={name}')
        debug_list.append(f'Edit {device_type}.{name} spectrum={name}')
        if SIM_DEBUG_PLOT:
            from OpenDSSWrapper.harmonic_spectra import plot_spectrum
            save_path = os.path.join(self.data_store.base_path, 'tmp', str(step))
            plot_spectrum(interpolated_spectrum, save_path=save_path, show_plot=False, name=name)

    def update_generators(self, step: int):
        """
        Update the generators at the given step with generator profiles and spectra
        :param step: time step
        :return:
        """
        debug_save = []
        self.update_device('PVsystem', step, debug_save)
        self.update_device('Generator', step, debug_save)
        self.update_device('Vsource', step, debug_save)
        if self.debug == DebugMode.VERBOSE:
            self.data_store.export_debug_object('generator_debug', debug_save, str(step))

    def update_loads(self, step: int):
        """
        Update the loads at the given step with load profiles and spectra
        :param step: time step
        :return:
        """
        load_debug = []
        self.update_device('Load', step, load_debug)
        if self.debug == DebugMode.VERBOSE:
            self.data_store.export_debug_object('load_debug', load_debug, str(step))

    def get_monitor_data(self, keep_orders=None) -> pd.DataFrame:
        """
        Get the monitor data for the current time step as dictionary with bus number as key and
        dataframe as value
        :param: keep_orders: list of the orders, that shall be kept in the results
        :return: multi-indexed pandas dataframe with indices 'bus', 'element', 'element_type', 'Freq'
        """
        # Initialize empty DataFrames for power, currents, and voltages
        power_df_list = []
        current_voltage_df_list = []
        index_list = []  # To collect index information (bus, element, element_type, Freq)
        for i in range(1, dss.Monitors.Count() + 1):
            dss.Monitors.Idx(i)  # Set active monitor
            name = dss.Monitors.Name()
            header = dss.Monitors.Header()  # column names
            # to lower case
            header = [x.lower() for x in header]
            if self.simulation_mode == SimulationMode.POWERFLOW:
                additional_columns = ['hour', 't(sec)']
            else:  # Assuming dynamic or harmonic simulation modes
                additional_columns = ['freq', 'harmonic']
            columns = additional_columns + header
            data = dss.Monitors.AsMatrix()
            df = pd.DataFrame(data, columns=columns)

            if self.simulation_mode == SimulationMode.TRANSIENTS:
                # only bus voltages are needed for TSE
                if 'monitoratbus' not in name:
                    continue
                # drop the rows, which are not wanted (Transient Simulation only)
                df = filter_harmonics_in_dataframe(df, keep_orders)

            # drop columns that are monitored but not required:
            df.drop(['harmonic'], axis=1, inplace=True)
            if 'v4' in columns:  # for loads and generators neutral is saved as well
                df.drop(['v4', 'vangle4', 'i4', 'iangle4'], axis=1, inplace=True)
            if 's4 (kva)' in columns:
                df.drop(['s4 (kva)', 'ang4'], axis=1, inplace=True)
            # Extract bus number from monitor name
            if dss.Monitors.Mode() == 1:
                name = name.replace('power', '')
            if 'monitoratbus' in name:
                element = name.replace('monitoratbus', '')
                bus = int(element)
                element = 'bus' + element
                element_type = ElementType.LINE
            elif 'monitoratload' in name:
                element = name.replace('monitoratload', '')
                bus = self.grid.get_bus_for_load(element)
                element_type = ElementType.LOAD
            elif 'monitoratgenerator' in name:
                element = name.replace('monitoratgenerator', '')
                bus = self.grid.get_bus_for_generator(element)
                element_type = ElementType.GENERATOR
            elif 'monitoratvsource' in name:
                element = name.replace('monitoratvsource', '')
                bus = int(element)
                element_type = ElementType.VSOURCE
            else:
                raise UserWarning("Unknown monitor name")
            monitor_mode = dss.Monitors.Mode()
            if monitor_mode == 1:  # 1 = Powers at designated Terminal
                power_df_list.append(df.assign(bus=element))
            elif monitor_mode == 0:  # 0 = Voltages and currents at designated Terminal
                current_voltage_df_list.append(df.assign(bus=element))
                for index, row in df.iterrows():
                    # Create a tuple for the MultiIndex
                    index_info = (bus, element, element_type.value, row['freq'])
                    index_list.append(index_info)
            else:
                raise UserWarning(f"Monitor mode {monitor_mode} not supported")
        # Convert lists to DataFrames
        current_voltage_df = pd.concat(current_voltage_df_list, ignore_index=True)
        if power_df_list:
            power_df = pd.concat(power_df_list, ignore_index=True)
            # Combine DataFrames
            combined_df = pd.merge(power_df, current_voltage_df, on=['bus', 'freq'], how='outer')
        else:
            combined_df = current_voltage_df

        # Set the DataFrame's index
        combined_df.index = pd.MultiIndex.from_tuples(index_list,
                                                      names=['bus', 'element', 'element_type', 'freq'])
        combined_df.drop(['freq', 'bus'], axis=1, inplace=True)
        return combined_df

    def run_simulation(self, start_step: int, end_step: int):
        """
        Run a simulation from start_step to end_step with the given step size
        :param start_step: start step
        :param end_step: end step
        :return: a dictionary with bus number as key and dataframe as value
        """
        # DO NOT use parallel mode of OpenDSS with quick for loops!
        # Active elements will not be updated correctly
        # dss.Parallel.ActiveParallel(1)  # multi processing
        # Instead use multiprocessing module from outside of this class
        for i in tqdm(range(start_step, end_step), ascii=True,
                      desc=f"Running Simulation steps {start_step}-{end_step} with "
                           f"{self.step_size_ms}ms time step"):
            self.run_single_simulation(i)

    def run_single_simulation(self, step: int):
        """
        Run a single time step simulation
        :param step: iteration index
        :return: a dictionary with bus number as key and dataframe as value
        """
        # SavedVoltages.dbl is created when solution mode is changed to Harmonics
        # Ensure this thread is the only one accessing this specific file by setting the datapath
        # to the current step
        step_path = os.path.join(self.data_store.base_path, "tmp", str(step))
        os.makedirs(step_path, exist_ok=True)
        send_text_command('Set Datapath=[{dpath}]'.format(dpath=step_path))

        # update spectra and profiles
        self.update_loads(step)
        self.update_generators(step)

        # Solve the base solution
        dss.Solution.Mode(SolutionMode.Snap.value)  # mode MUST be set by integer ID
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            logger.error(f"OpenDSS Snap solution did not converge at step {step}")

        if self.simulation_mode != SimulationMode.POWERFLOW:
            # Solve Harmonics flow
            dss.Solution.Mode(SolutionMode.Harmonic.value)  # mode MUST be set by integer ID
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                logger.error(f"OpenDSS Harmonic solution did not converge at step {step}")

            if self.debug == DebugMode.VERBOSE:  # for debugging purposes export as CSV files
                for i in range(1, dss.Monitors.Count() + 1):
                    dss.Monitors.Idx(i)  # set active Monitor
                    send_text_command('Export monitor ' + dss.Monitors.Name())
        result = self.get_monitor_data()

        self.data_store.add_data(step, result)
        if self.debug == DebugMode.VERBOSE:
            self.data_store.export_debug_object('monitor_results', result, str(step), as_pickle=True)
            for key, value in get_properties().items():
                self.data_store.export_debug_object(key, value, str(step))
        if SIM_DEBUG_PLOT:
            self.data_store.plot_data_3d(result, str(step), target_phase=1, target_mode='voltage', pu=False,
                                         drop_frequencies=[self.grid.pp_net.f_hz])
            self.data_store.plot_data_2d_heatmap(result, str(step), target_phase=1, target_mode='voltage', pu=False,
                                                 drop_frequencies=[self.grid.pp_net.f_hz])
            # convert to per-unit
            from OpenDSSWrapper.simulation import PostProcessor
            post_processor = PostProcessor(self.grid.pp_net, self.grid.get_base_kvs(), 20.)
            df = post_processor.to_pu(result)
            self.data_store.plot_data_3d(df, str(step), target_phase=1, target_mode='voltage', pu=True,
                                         drop_frequencies=[self.grid.pp_net.f_hz])
            self.data_store.plot_data_2d_heatmap(df, str(step), target_phase=1, target_mode='voltage', pu=True,
                                                 drop_frequencies=[self.grid.pp_net.f_hz])

    def run_transient_simulation(self, source_power_MVA: float, bus_number: int,
                                 spectrum: Spectrum, filter_relevant_harmonics=True) -> bool:
        """
        Run a transient simulation
        :param source_power_MVA: Transient source power in MVA
        :param bus_number: Bus number to place the transient source at
        :param spectrum: transient Spectrum
        :return: False if the results contain NaN values (failure in simulation), else True
        """
        # directory for saved voltages dbl file
        step_dir = os.path.join(spectrum.name, str(bus_number), str(source_power_MVA))
        step_path = os.path.join(self.data_store.base_path, "tmp", step_dir)
        os.makedirs(step_path, exist_ok=True)
        send_text_command('Set Datapath=[{dpath}]'.format(dpath=step_path))
        transient_spectrum_string = spectrum.get_opendss_string(TRANSIENTNAMESOURCE + "spectrum")

        # Solve the base solution
        dss.Solution.MaxIterations(100)
        dss.Solution.LoadModel(LoadModel.POWERFLOW.value)
        dss.Solution.Mode(SolutionMode.Snap.value)
        dss.Solution.Solve()

        if not dss.Solution.Converged():
            logger.error(f"Base solve did not converge for Transient spectrum {transient_spectrum_string} with Power "
                         f"{str(source_power_MVA)} at bus {bus_number}.")

        # As the power flow is calculated, the Vsource for transients can be placed with magnitude and angle
        send_text_command(transient_spectrum_string)
        voltage_rms = self.update_transient_source(source_power_MVA, bus_number, TRANSIENTNAMESOURCE + "spectrum")
        enable_transient_source(True)

        # Solve Harmonics flow once to read all frequencies
        dss.Solution.Mode(SolutionMode.Harmonic.value)
        dss.Solution.Solve()

        self.update_transient_damping_fundamental(source_power_MVA, bus_number,
                                                  TRANSIENTNAMESOURCE + "spectrum", voltage_rms)

        # Now solve again with the set reactor element
        dss.Solution.Mode(SolutionMode.Harmonic.value)
        dss.Solution.Solve()

        if not dss.Solution.Converged():
            logger.error(f"Base solve did not converge for Transient spectrum {transient_spectrum_string} with Power "
                         f"{str(source_power_MVA)} at bus {bus_number}.")
        if self.debug == DebugMode.VERBOSE:  # for debugging purposes export as CSV files
            for i in range(1, dss.Monitors.Count() + 1):
                dss.Monitors.Idx(i)  # set active Monitor
                send_text_command('Export monitor ' + dss.Monitors.Name())

        # remove Sources for next iteration, so the powerflow can be calculated without former sources
        enable_transient_source(False)

        # filter orders, which are relevant for the transient. monitors provide the orders with 6 digits precision
        # after decimal
        if filter_relevant_harmonics:
            keep_orders = [round(o, 6) for o in spectrum.harmonics]
        else:
            keep_orders = None
        result = self.get_monitor_data(keep_orders)

        # check if there are nan values in the result
        if result.isnull().values.any():
            logger.error(f"NaN values found in result at step {step_dir}")
            return False
        identifier = f"{spectrum.name.replace('_', '')}-{str(bus_number)}-{str(source_power_MVA)}"
        self.data_store.add_data(identifier, result)
        return True

    def update_transient_source(self, transient_source_power_mva: float, bus_number: int, spectrum_name: str):
        """
        Update the transient source at the given bus with the given spectrum and power
        :param transient_source_power_mva: transient source power in MVA
        :param bus_number: bus number to place the transient source at
        :param spectrum_name: Transient Spectrum name
        :return: bus voltage RMS
        """

        dss.Circuit.SetActiveBus(str(bus_number))
        base_kv = dss.Bus.kVBase()
        # LN Voltages
        pu_voltages = dss.Bus.puVmagAngle()
        voltages = dss.Bus.Voltages()
        columns = ['V1Mag', 'V1Angle', 'V2Mag', 'V2Angle', 'V3Mag', 'V3Angle']
        df = pd.DataFrame([pu_voltages], columns=columns)
        pu = df['V1Mag'].values[0]
        angle = df['V1Angle'].values[0]
        df = pd.DataFrame([voltages], columns=columns)
        voltage = df['V1Mag'].values[0]

        # Manipulate the angles here
        transient_source_command = (
            f'New VSource.{TRANSIENTNAMESOURCE} x1r1=0.001 x0r0=0.001 Phases=1.0 MVAsc1={transient_source_power_mva} '
            f'basekv={base_kv} pu={pu} bus1={bus_number}.1.0 spectrum={spectrum_name} angle={angle}')

        if TRANSIENTNAMESOURCE in str([dss.Vsources.AllNames()]):
            transient_source_command = transient_source_command.replace('New', 'Edit')

        # Add transient on bus
        send_text_command(transient_source_command)

        return voltage

    def update_transient_damping_fundamental(self, transient_source_power_mva: float,
                                             bus_number: int, spectrum_name: str, voltage_rms_at_bus: float):
        """
        Update the transient source at the given bus with the given spectrum and power
        :param transient_source_power_mva: transient source power in MVA
        :param bus_number: bus number to place the transient source at
        :param spectrum_name: Transient Spectrum name
        :param voltage_rms_at_bus: This is the voltage at the bus from the PF simulation
        :return:
        """
        dss.Circuit.SetActiveBus(str(bus_number))
        base_kv = dss.Bus.kVBase()
        # Add equivalent load to reduce 50Hz effect of Source
        frequencies = read_frequencies_from_opendss()
        self._edit_lr_curve(frequencies)

        reactor_imp = (1000000 * transient_source_power_mva) / (voltage_rms_at_bus ** 2)
        transient_load_command = (
            f'New Reactor.{TRANSIENTNAMELOAD}  phases=1 R={reactor_imp} X=1000 '
            f'kv={base_kv} bus1={bus_number}.1.0 RCurve={TRANSIENTRCURVE} LCurve={TRANSIENTLCURVE}')
        if TRANSIENTNAMELOAD in str([dss.Reactors.AllNames()]):
            transient_load_command = transient_load_command.replace('New', 'Edit')

        send_text_command(transient_load_command)

    def save_pandas(self):
        """
        Convert the opendss network to pandas dataframes and save those
        class_to_dataframe sometimes allows extracting additional information e.g. leadLag of Transformers
        :return: pandas dataframe
        """
        # df_loads = dss.utils.to_dataframe(dss.Loads)
        df_loads = dss.utils.loads_to_dataframe()
        df_loads.to_csv(os.path.join(self.data_store.base_path, 'loads.csv'))

        # df_lines = dss.utils.to_dataframe(dss.Lines)
        df_lines = dss.utils.lines_to_dataframe()
        df_lines.to_csv(os.path.join(self.data_store.base_path, 'lines.csv'))

        df_linecodes = dss.utils.to_dataframe(dss.LineCodes)
        df_linecodes.to_csv(os.path.join(self.data_store.base_path, 'linecodes.csv'))

        # df_transformers = dss.utils.to_dataframe(dss.Transformers)
        df_transformers = dss.utils.transformers_to_dataframe()
        df_transformers.to_csv(os.path.join(self.data_store.base_path, 'transformers.csv'))

        # df_generators = dss.utils.to_dataframe(dss.Generators)
        df_generators = dss.utils.generators_to_dataframe()
        df_generators.to_csv(os.path.join(self.data_store.base_path, 'generators.csv'))

        # df_pvsystems = dss.utils.to_dataframe(dss.PVsystems)
        df_pvsystems = dss.utils.pvsystems_to_dataframe()
        df_pvsystems.to_csv(os.path.join(self.data_store.base_path, 'pvsystems.csv'))

        # df_vsources = dss.utils.to_dataframe(dss.Vsources)
        df_vsources = dss.utils.vsources_to_dataframe()
        df_vsources.to_csv(os.path.join(self.data_store.base_path, 'vsources.csv'))

        # df_monitors = dss.utils.to_dataframe(dss.Monitors)
        df_monitors = dss.utils.monitors_to_dataframe()
        df_monitors.to_csv(os.path.join(self.data_store.base_path, 'monitors.csv'))

        df_bus = dss.utils.to_dataframe(dss.Bus)
        df_bus.to_csv(os.path.join(self.data_store.base_path, 'bus.csv'))

        # df_capacitors = dss.utils.to_dataframe(dss.Capacitors)
        df_capacitors = dss.utils.capacitors_to_dataframe()
        df_capacitors.to_csv(os.path.join(self.data_store.base_path, 'capacitors.csv'))

        # df_fuses = dss.utils.to_dataframe(dss.Fuses)
        df_fuses = dss.utils.fuses_to_dataframe()
        df_fuses.to_csv(os.path.join(self.data_store.base_path, 'fuses.csv'))

        # This allows reading default spectra as well!
        df_spectrum = dss.utils.class_to_dataframe('Spectrum')
        df_spectrum.to_csv(os.path.join(self.data_store.base_path, 'spectrum.csv'))

    def _create_lr_curve(self, spectrum: Spectrum, fund_freq=50.0):
        """
        Prepare the default frequency curves for the reactor object
        By these curves only the 50Hz component in the transient damping reactor is set
        :param spectrum: Spectrum to be used for this simulation
        :param fund_freq: Fundamental frequency for the solution
        """
        frequencies = [round(h * fund_freq, 6) for h in spectrum.harmonics]
        self._set_lr_curve_params(frequencies, 'new', fund_freq=fund_freq)

    def _edit_lr_curve(self, frequencies, fund_freq=50.0):
        """
        Manipulate the frequency curves for the reactor object
        By these curves only the 50Hz component in the transient damping reactor is set
        :param frequencies: frequencies obtained during simulation
        :param fund_freq: Fundamental frequency for the solution
        """
        self._set_lr_curve_params(frequencies, 'edit', fund_freq=fund_freq)


    @staticmethod
    def _set_lr_curve_params(frequencies, edit_or_new, fund_freq=50.0):
        """
        Set the frequency curves for the reactor object. either manipulate or create a fresh object.
        :param frequencies: frequencies obtained during simulation or default values.
        :param edit_or_new: whether to edit or create the new frequency curves
        :param fund_freq: Fundamental frequency for the solution.
        """
        pu_factors_r = [1000000 for _ in frequencies]
        pu_factors_l = pu_factors_r.copy()
        pu_factors_r[frequencies.index(fund_freq)] = 1.0
        x = ''
        y_l = ''
        y_r = ''
        for f, r, l in zip(frequencies, pu_factors_r, pu_factors_l):
            x += '{:.12f}'.format(f) + ','
            y_r += '{:.12f}'.format(r) + ','
            y_l += '{:.12f}'.format(l) + ','

        x = x[:-1]
        y_r = y_r[:-1]
        y_l = y_l[:-1]
        command_xy_l = f'{edit_or_new} XYCurve.{TRANSIENTLCURVE} npts={len(frequencies)} Yarray=({y_l}) Xarray=({x})'
        send_text_command(command_xy_l)
        command_xy_r = f'{edit_or_new} XYCurve.{TRANSIENTRCURVE} npts={len(frequencies)} Yarray=({y_r}) Xarray=({x})'
        send_text_command(command_xy_r)

