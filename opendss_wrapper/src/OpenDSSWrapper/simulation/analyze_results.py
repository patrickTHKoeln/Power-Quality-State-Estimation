import pandas as pd
import numpy as np
import os
from OpenDSSWrapper.tools import config as cfg
from OpenDSSWrapper.simulation.plot_data import plot_nodal_voltage_distribution

def copy_bus_data_df(df_all: pd.DataFrame, source_bus: int, new_bus: int) -> pd.DataFrame:
    """
    Copies bus data from one bus to another and attaches the data to the dataframe
    :param df_all: DataFrame containing original bus data
    :param source_bus: Index of bus data to copy
    :param new_bus: The new bus index, which should be created
    :return: DataFrame containing copied bus data and original data
    """
    # Wähle die Zeilen mit 'bus' == source_bus
    df_bus1 = df_all.xs(source_bus, level='bus').copy()
    # Ändere den 'bus'-Index zu 1

    new_index = pd.MultiIndex.from_product([df_all.index.levels[0], [new_bus], df_all.index.levels[2]],
                                           names=df_all.index.names)

    df_bus1 = df_bus1.set_index(new_index)

    # Füge die neuen Zeilen dem ursprünglichen DataFrame hinzu
    return pd.concat([df_all, df_bus1])


def swap_sequence_buses_df(df_all: pd.DataFrame, source_buses: [int]) -> pd.DataFrame:
    """
    Swaps bus phase angles of L2 and L3 from several buses
    :param df_all: DataFrame containing original bus data
    :param source_buses: for which buses the sequence should be swapped?
    :return: df with swapped sequences
    """
    df_bus_swap_phases = df_all.loc[df_all.index.get_level_values('bus').isin(source_buses)]

    # Tausche die Spalten 'A' und 'B' für diese Zeilen
    df_bus_swap_phases[['v3', 'v2']] = df_bus_swap_phases[['v2', 'v3']]
    df_bus_swap_phases[['vangle3', 'vangle2']] = df_bus_swap_phases[['vangle2', 'vangle3']]

    # Aktualisiere den ursprünglichen DataFrame
    df_all.update(df_bus_swap_phases)
    return df_all


def read_lab_data(data_path: str, f_name: str, frequencies=None, swap_sequence_buses=None):
    """
    Read "live" data of lab measurements from csv
    :param data_path: path to data directory
    :param f_name: file name of csv results
    :param frequencies: A list holding the frequencies, which are needed
    :param swap_sequence_buses: if negative sequence is measured, these buses L2 and L3 are swapped.
    :return: formatted dataframe holding lab measurements
    """
    df = pd.read_csv(os.path.join(data_path, f_name), index_col=[0, 1, 2])
    df['vangle1'] = df['vangle1'] + df['phasor_offset']
    df['vangle2'] = df['vangle2'] + df['phasor_offset']
    df['vangle3'] = df['vangle3'] + df['phasor_offset']
    # temp swap phases for
    if not swap_sequence_buses:
        swap_sequence_buses = [0, 3, 6, 7]
    df = swap_sequence_buses_df(df, swap_sequence_buses)
    # data format (batch / step , frequency, node)
    df.drop(columns=['phasor_offset'], inplace=True)

    # nodes 0, 1, 2 are actually the same measurement and just connected by switches, copy measurement 0
    df = copy_bus_data_df(df, 0, 1)
    df = copy_bus_data_df(df, 0, 2)

    # order after adding bus data
    df.sort_index(level=['step', 'bus', 'freq'], inplace=True)

    # Add dummy columns for the simulation data format
    new_index = pd.MultiIndex.from_product([df.index.levels[0], df.index.levels[1], df.index.levels[2], [0], [0]],
                                           names=df.index.names + ['element', 'element_type'])
    expanded_df = df.reindex(new_index)

    return expanded_df[expanded_df.index.get_level_values('freq').isin(frequencies)]


if __name__ == '__main__':

    frequencies = list(np.arange(50, 1300, 50))
    figure_path = os.path.join(cfg.data_path, 'lab_evaluation', 'random_angles_3ph', 'eberle_nodes')

    # load data from lab
    data_lab = read_lab_data(cfg.data_path, 'SE_Results_24-7-2025.csv', frequencies)
    data_lab = data_lab.reset_index(['element', 'element_type'], drop=True)
    # convert angles to degrees if needed
    data_lab['vangle1'] = data_lab['vangle1'] * 180 / np.pi
    data_lab['vangle2'] = data_lab['vangle2'] * 180 / np.pi
    data_lab['vangle3'] = data_lab['vangle3'] * 180 / np.pi
    # convert % values to voltages

    # load data from db
    from OpenDSSWrapper.simulation.store import PostgreSQLStore

    columns = ['v1', 'vangle1', 'v2', 'vangle2', 'v3', 'vangle3']

    train_store = PostgreSQLStore('data', columns, cfg.postgres_config,
                                  table_name=f'harmonics_thkoelnnetwork_train')
    data_sim = train_store.read_data(steps=list(np.arange(4001, 5000, 1)), columns=columns, element_types=[0],
                                       frequencies=frequencies)
    data_sim = data_sim.reset_index(['element', 'element_type'], drop=True)

    # compare magnitude and angle distribution per phase and per harmonic
    for f in data_lab.index.get_level_values('freq').unique().tolist():
        harmonic_path = os.path.join(figure_path, str(f))
        if not os.path.exists(harmonic_path):
            os.makedirs(harmonic_path)
        for bus in [0, 3, 6, 7]:  # data_lab.index.get_level_values('bus').unique().tolist():
            harmonic = f / 50
            lab_magnitude = data_lab.loc[pd.IndexSlice[:, bus, f], :]
            sim_magnitude = data_sim.loc[pd.IndexSlice[:, bus, f], :]

            plot_nodal_voltage_distribution(bus, harmonic, 'magnitude',
                                            lab_magnitude[['v1', 'v2', 'v3']].to_numpy(),
                                            sim_magnitude[['v1', 'v2', 'v3']].to_numpy(),
                                            harmonic_path, False)
            plot_nodal_voltage_distribution(bus, harmonic, 'phase',
                                            lab_magnitude[['vangle1', 'vangle2', 'vangle3']].to_numpy(),
                                            sim_magnitude[['vangle1', 'vangle2', 'vangle3']].to_numpy(),
                                            harmonic_path, False)
    # compare the phase angle diffs across the grid
