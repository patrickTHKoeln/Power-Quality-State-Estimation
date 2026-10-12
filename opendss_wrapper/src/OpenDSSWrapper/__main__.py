import glob
import itertools
import logging
import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from OpenDSSWrapper.converter.grids import BaseGrid, MicrogridTHKoeln, IEEE33TestFeeder, CigreLVDistPV
from OpenDSSWrapper.harmonic_spectra import Spectrum, SpectraStoreSQLite
from OpenDSSWrapper.power_profiles import PowerProfileStoreSQLite
from OpenDSSWrapper.simulation import DataStore, PostgreSQLStore, CSVStore, PickleStore
from OpenDSSWrapper.simulation import SimulationMode, OpenDSSSimulation, DebugMode
from OpenDSSWrapper.simulation.simulation import get_properties
from OpenDSSWrapper.tools import config as cfg
from OpenDSSWrapper.tools.logging import setup_logging

setup_logging(cfg.log_path)
logger = logging.getLogger(__name__)

SEED = 1  # Seed for random noise

# columns = ['v1', 'vangle1', 'v2', 'vangle2',
#            'v3', 'vangle3', 'i1', 'iangle1', 'i2', 'iangle2',
#            'i3', 'iangle3']
columns = ['v1', 'vangle1', 'v2', 'vangle2', 'v3', 'vangle3',]


def sim_worker(i):
    """
    Worker function for multiprocessing
    :param i: Simulation number
    :return:
    """
    sim.run_single_simulation(i)


def sim_worker_transient(power_mva: float, bus: int, spectrum: Spectrum):
    """
    Worker function for multiprocessing
    :param power_mva: applied Power for spectral source (in MVA)
    :param bus: Busbar where the spectrum schall be injected
    :param spectrum: Spectrum, which represents the transient effect
    :return:
    """
    sim.run_transient_simulation(power_mva, bus, spectrum)


def init_dss_pool(grid: BaseGrid, sim_mode: SimulationMode, spectra: dict, generator_profiles: dict,
                  load_profiles: dict, data_exporter: DataStore, grid_path: str, step_size_ms: int):
    """
    Worker initialization function for multiprocessing
    Only initialize openDSS once per worker
    :param grid: BaseGrid object
    :param sim_mode: SimulationMode
    :param spectra: Dictionary of spectra
    :param generator_profiles: Dictionary of generator profiles
    :param load_profiles: Dictionary of load profiles
    :param data_exporter: DataExporter object
    :param grid_path: Path to the OpenDSS folder
    :param step_size_ms: Time step in milliseconds
    :return:
    """
    global sim

    sim = OpenDSSSimulation(grid, sim_mode, spectra, generator_profiles, load_profiles,
                            data_exporter, grid_path, step_size_ms)


def simulate_time_series(grid: BaseGrid, spectra: dict, generator_profiles: dict, load_profiles: dict,
                         data_store: DataStore,
                         start_at_step=1,
                         stop_at_step=525600, step_size_ms=60000, multi_proc=True, grid_path: str = None):
    """
    if multi_proc is True: Using multiprocessing, multiple instances of openDSS are started.
    This takes additional start up time (2-3 minutes) and lots of RAM
    :param grid: BaseGrid object
    :param spectra: Dictionary of spectra
    :param generator_profiles: Dictionary of generator profiles
    :param load_profiles: Dictionary of load profiles
    :param data_store: DataStore object
    :param start_at_step: [optional] Start simulation at this step, default 1
    :param stop_at_step: [optional] Stop simulation at this step, default 527026 (1 year)
    :param step_size_ms: [optional] Time step in milliseconds, default 60000 (1 minute)
    :param multi_proc: boolean, if enabled uses all the available minus 2 cores, default True
    :param grid_path: Path to the OpenDSS folder
    :return:
    """
    logger.info(f"Simulating grid {grid.name} from {start_at_step} to {stop_at_step}")
    # Optional logging of all spectra and profiles used for this simulation
    # data_store.export_debug_object('spectra.pic', spectra, as_pickle=True)
    # data_store.export_debug_object('generator_profiles.pic', generator_profiles, as_pickle=True)
    # data_store.export_debug_object('load_profiles.pic', load_profiles, as_pickle=True)
    if multi_proc:
        n_cores = cpu_count() - 4  # not using every core so that we don't run into scheduling issues
        with Pool(n_cores, initializer=init_dss_pool,
                  initargs=(
                          grid, SimulationMode.HARMONICS, spectra, generator_profiles, load_profiles, data_store,
                          grid_path, step_size_ms)) as p:
            list(tqdm(p.imap(
                sim_worker,
                range(start_at_step, stop_at_step)
            ), total=stop_at_step - start_at_step))
    else:
        sim = OpenDSSSimulation(grid, SimulationMode.HARMONICS, spectra, generator_profiles, load_profiles,
                                data_store, grid_path, step_size_ms)
        # sim.add_frequency_sweep(1000, 20000, 50, 50)
        sim.run_simulation(start_at_step, stop_at_step)


def simulate_transient_series(grid: BaseGrid, spectra: dict, generator_profiles: dict, load_profiles: dict,
                              data_store: DataStore,
                              list_of_nodes: [int],
                              list_nom_power_mva: [float],
                              transient_spectra: [Spectrum],
                              multi_proc=False, grid_path: str = None):
    """
    if multi_proc is True: Using multiprocessing, multiple instances of openDSS are started.
    This takes additional start up time (2-3 minutes) and lots of RAM
    :param grid: BaseGrid object
    :param spectra: Dictionary of spectra
    :param generator_profiles: Dictionary of generator profiles
    :param load_profiles: Dictionary of load profiles
    :param data_store: DataStore object
    :param list_of_nodes: A List of int number to which the transient shall be injected to
    :param list_nom_power_mva: A list of mva scaled powers, the spectrum shall have
    :param transient_spectra:  The Transient spectra to be used
    :param multi_proc: boolean, if enabled uses all the available minus 2 cores, default True
    :param grid_path: Path to the OpenDSS folder
    :return:
    """

    debug_mode = DebugMode.NONE
    if multi_proc:
        n_cores = cpu_count() - 4  # not using every core so that we don't run into scheduling issues
        paramlist = list(itertools.product(transient_spectra, list_of_nodes, list_nom_power_mva))
        with Pool(n_cores, initializer=init_dss_pool,
                  initargs=(grid, SimulationMode.TRANSIENTS, spectra, generator_profiles, load_profiles,
                            data_store, debug_mode, grid_path)) as p:
            list(tqdm(p.imap(
                sim_worker_transient,
                paramlist
            )))
    else:
        sim = OpenDSSSimulation(grid, SimulationMode.TRANSIENTS, spectra, generator_profiles, load_profiles,
                                data_store, grid_path)

        for transient in tqdm(transient_spectra):
            # Specific settings for a new spectrum
            sim.pre_settings_transient_simulation(transient)

            for bus in list_of_nodes:
                for power in list_nom_power_mva:
                    sim.run_transient_simulation(power, bus, transient)
            # Reset the simulation after every new spectrum.
            # DSS is confused after couple of modifications on the same spectrum
            sim.reset_transient_simulation(transient)


def load_transient_spectra(path: str):
    """
    :param path: Here are the spectra stored
    :param spectra: The string names of these
    :return: A List of Spectra
    """
    return_spectra = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    os.chdir(path)
    all_csv_spectra = glob.glob("*.csv")
    if len(all_csv_spectra) == 0:
        raise FileNotFoundError(f"No csv files found in {path}")
    for spec in all_csv_spectra:
        new_spec = Spectrum.from_csv(path, spec)
        if new_spec.number_of_harmonics < 2:
            print(f"Spectrum {spec} has just fundamental frequency. Spectrum deleted.")
            os.remove(os.path.join(path, spec))
        else:
            return_spectra.append(new_spec)
    return return_spectra


def filter_already_simulated_transients(results_folder, transient_spectra_list, filter_folder=False,
                                        filter_word: str = None, indices=(0, -1)):
    """
    Reduce the spectra-list to the cases, that were not simulated so far
    :param results_folder: The path, where the results are stored
    :param transient_spectra_list: The overall-list
    :param filter_folder: Boolean: True if results shall be checked, if False all cases will be simulated.
    :param filter_word: If not None, only Spectra containing this keyword are used.
    :param indices: Also filtering by range in list is possible.
    """
    if filter_folder:
        list_spectra_names = [t.name.replace('_', '') for t in transient_spectra_list]
        all_csv_spectra = next(os.walk(results_folder))[1]
        result_list = list(set(list_spectra_names) - set(all_csv_spectra))
        transient_spectra_list = [t for t in transient_spectra_list if t.name.replace('_', '') in result_list]
    if filter_word is not None:
        transient_spectra_list = [t for t in transient_spectra_list if filter_word in t.name]
    return transient_spectra_list[indices[0]: indices[1]]


def sim_grid(grid: BaseGrid, name: str, dataset: str = 'train',
             sim_mode: SimulationMode = SimulationMode.HARMONICS,
             transient_powers_mva: list[float] = None, start_at_step=0, stop_at_step=525600):
    """
    Simulate the CIGRE LV grid
    :param grid: BaseGrid object
    :param name: Name of the grid
    :param dataset: train, test or validation
    :param sim_mode: Which type of simulation shall be executed
    :param transient_powers_mva: Which powers shall be applied to the transient source
    :return:
    """
    logger.info(f"Simulating {name} with {sim_mode.name} mode")
    # data_store = HDF5Store(data_path, f'{name}_{dataset}.h5')
    # data_store = SQLiteStore(data_path, f'{name}_{dataset}.db', columns)
    if sim_mode == SimulationMode.HARMONICS:
        table_name = f'harmonics_{name}_{dataset}'
        data_store = PostgreSQLStore(cfg.data_path, columns, cfg.postgres_config, step_dtype='INTEGER',
                                     table_name=table_name)
        delete_confirmation = input(f"Delete table {table_name} hosted at {cfg.postgres_config['host']}? [y/N]")
        if delete_confirmation.lower() in ['yes', 'ye', 'y']:
            data_store.empty_table()
        else:
            logger.info(f"Appending data to table {table_name} hosted at {cfg.postgres_config['host']}")
    elif sim_mode == SimulationMode.TRANSIENTS:
        # data_store = PostgreSQLStore(data_path, columns, postgres_config, step_dtype='TEXT', table_name='transients')
        # data_store = CSVStore(cfg.data_path, cfg.results_sub_path, f'{name}_{dataset}', columns)
        data_store = PickleStore(cfg.data_path, f'{name}_{dataset}', columns)
    else:
        raise NotImplementedError(f"SimulationMode {sim_mode} not implemented")

    # data_store = PickleStore(cfg.data_path, f'{name}_{dataset}.pkl')
    # data_store = DataStore(cfg.data_path, columns=columns)
    # data_store = CSVStore(cfg.data_path, results_sub_path, f'{name}_{dataset}', columns)

    # get spectra
    spectra = {}
    spectra_store = SpectraStoreSQLite(os.path.join(cfg.data_path, cfg.spectrum_db_name))
    # spectra_store = SpectraStorePostgres(postgres_config)
    power_profile_store = PowerProfileStoreSQLite(os.path.join(cfg.data_path, cfg.power_profile_db_name))
    # power_profile_store = PowerProfileStorePostgres(postgres_config)
    load_spectra_configs = grid.get_load_spectra(dataset)
    generator_spectra_configs = grid.get_generator_spectra(dataset)
    # get spectra data from sql db
    for bus, spectrum_config in generator_spectra_configs.items():
        spectra[bus] = spectra_store.get_spectra(spectrum_config)
    for bus, spectrum_config in load_spectra_configs.items():
        spectra[bus] = spectra_store.get_spectra(spectrum_config)
    # get power profiles from sql db
    load_profiles = {}
    load_profile_configs = grid.get_load_profiles(dataset, n_samples=stop_at_step-start_at_step)
    for bus, load_profile_config in load_profile_configs.items():
        load_profiles[bus] = power_profile_store.retrieve_profile_values(load_profile_config)
    generator_profiles = {}
    generator_profile_configs = grid.get_generator_profiles(dataset, n_samples=stop_at_step-start_at_step)
    for bus, generator_profile_config in generator_profile_configs.items():
        generator_profiles[bus] = power_profile_store.retrieve_profile_values(generator_profile_config)

    grid_path = os.path.join(cfg.data_path, name)

    # Which Simulation type shall be used?
    match sim_mode:
        case SimulationMode.HARMONICS:
            simulate_time_series(grid, spectra, generator_profiles, load_profiles, data_store,
                                 start_at_step=start_at_step, stop_at_step=stop_at_step, step_size_ms=60000,
                                 multi_proc=cfg.MULTI_PROCESSING, grid_path=grid_path)

        case SimulationMode.TRANSIENTS:
            numbered_busbars = grid.get_nodes_as_list_of_int()
            transient_spectra_list = load_transient_spectra(cfg.transient_path)
            transient_spectra_list = filter_already_simulated_transients(os.path.join(cfg.data_path, cfg.results_sub_path),
                                                                         transient_spectra_list, True,
                                                                         filter_word="",
                                                                         indices=(start_at_step, stop_at_step))

            simulate_transient_series(grid, spectra, generator_profiles, load_profiles, data_store,
                                      numbered_busbars, transient_powers_mva, transient_spectra_list,
                                      multi_proc=cfg.MULTI_PROCESSING, grid_path=grid_path)
    data_store.close()
    # data = data_store.read_data()
    # data_np, axis_desc = DataStore.df_to_numpy(data)
    res = get_properties()  # does not work if simulation was done using multiprocessing
    pass

experiments = []
exp0 = {
    "name": "THKoelnNetwork_delta_ni",
    "with_grid_impedance": True,
    "ignore_vsource_angles": False,
    "line1.5": None,
    "line10": None
}
exp1 = {
    "name": "THKoelnNetwork",
    "with_grid_impedance": True,
    "ignore_vsource_angles": False,
    "line1.5": None,
    "line10": None
}
exp2 = {
    "name": "THKoelnNetwork_angles_zero_ni",
    "with_grid_impedance": False,
    "ignore_vsource_angles": True,
    "line1.5": None,
    "line10": None
}
exp3 = {
    "name": "THKoelnNetwork_R_L_ni",
    "with_grid_impedance":False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 152.63,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km':  11.9067,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.01,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 47.6267,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.039612,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 203.11,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 1.786,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.43,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 7.144,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.049769,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}

exp4 = {
    "name":"THKoelnNetwork_R0_4R1_ni",
    "with_grid_impedance":False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 152.63,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.01,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 59.1652,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.039612,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 203.11,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.43,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 8.228,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.049769,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}
exp5 = {
    "name": "THKoelnNetwork_X_L_ni",
    "with_grid_impedance": False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 152.63,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.08,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.01,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 56.5108,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.24,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 203.11,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.08,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.43,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 8.80465,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.24,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}

exp6 = {
    "name": "THKoelnNetwork_X0_3X1_ni",
    "with_grid_impedance": False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 0,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
         'c0_nf_per_km': 0,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 56.5108,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.044277,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 0,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
         'c0_nf_per_km': 0,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 8.80465,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.052668,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}
exp7 = {
    "name": "THKoelnNetwork_no_C_ni",
    "with_grid_impedance": False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 0,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
         'c0_nf_per_km': 0,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 56.5108,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.039612,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 0,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
         'c0_nf_per_km': 0,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 8.80465,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.049769,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}
exp8 = {
    "name": "THKoelnNetwork_no_impedance",
    "with_grid_impedance": False,
    "ignore_vsource_angles": False,
    "line1.5":{
        'c_nf_per_km': 152.63,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km -> 11.907 Ohm/km?
         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.01,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 56.5108,  # zero line resistance                       Ω/km -> 15.876 Ohm/km?
         'x0_ohm_per_km': 0.039612,  # zero line reactance                        Ω/km
         'max_i_ka': 0.016,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    },
    "line10":{
        'c_nf_per_km': 203.11,  # line capacitance (line-to-earth)      nF/km
         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km -> 1.786 Ohm/km?
         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
         'c0_nf_per_km': 49.43,  # zero line capacitance (line-to-earth)      nF/km
         'r0_ohm_per_km': 8.80465,  # zero line resistance                       Ω/km -> 2.381 Ohm/km?
         'x0_ohm_per_km': 0.049769,  # zero line reactance                      Ω/km -> 0.00377 Ohm/km?
         'max_i_ka': 0.06,  # maximal thermal current               kA
         'type': 'cs'  # cs - underground cable system
    }
}
experiments.append(exp0)
# experiments.append(exp1)
# experiments.append(exp2)
# experiments.append(exp3)
# experiments.append(exp4)
# experiments.append(exp5)
# experiments.append(exp6)
# experiments.append(exp7)
# experiments.append(exp8)

if __name__ == '__main__':
    transient_source_power = [0.1, 1.0]
    # transient_source_power = [1.0]
    # cigre_lv = CigreLVDistPV()

    # ieee33 = IEEE33TestFeeder()
    sim_dataset_config = {  # define length (and time of the year) of training, test and validation set
        'train': (0, 100000),
        # 'test': (0, 100),
        #'validation': (0, 10000)
    }
    for exp in experiments:
        for dataset, steps in sim_dataset_config.items():
            microgrid_th = MicrogridTHKoeln(line1=exp['line1.5'], line10=exp['line10'],
                                            with_grid_impedance=exp['with_grid_impedance'],
                                            ignore_source_angles=exp['ignore_vsource_angles'])
            # cigre_lv = CigreLVDistPV()
            sim_grid(microgrid_th,
                 # name='CIGRELVDist',
                 name=exp.get('name'),
                 # name='IEEE33',
                 dataset=dataset,
                 sim_mode=SimulationMode.HARMONICS,
                 transient_powers_mva=transient_source_power,
                 start_at_step=steps[0],
                 stop_at_step=steps[1])
