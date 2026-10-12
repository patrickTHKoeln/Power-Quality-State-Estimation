import logging
import os
import sqlite3
from multiprocessing import Manager, Lock
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)


class DataStore:
    """
    Base class for data exporters
    If used as a standalone class, it will store data in memory and provide methods to read it back.
    Reading in a whole dataset is limited by the available memory.
    """

    def __init__(self, base_path: str = None, columns: list = None):
        """
        Initialize DataExporter
        :param base_path: base path
        :param columns: columns to export, default None (all columns)
        """
        self.base_path = base_path
        self.columns = columns
        if base_path is not None:
            os.makedirs(self.base_path, exist_ok=True)
        self.data = pd.DataFrame()

    def export_debug_object(self, file_name: str, debug_object, sub_path: str = "", as_pickle: bool = False):
        """
        Export debug object to a file
        :param file_name: file name
        :param debug_object: debug object to export
        :param sub_path: [optional] sub path
        :param as_pickle: [optional] export as pickle file
        """
        path = os.path.join(self.base_path, "tmp", sub_path)
        os.makedirs(path, exist_ok=True)
        if as_pickle:
            with open(os.path.join(path, file_name + ".pic"), "wb") as f:
                import pickle
                pickle.dump(debug_object, f)
                return
        if isinstance(debug_object, pd.DataFrame):
            debug_object.to_csv(os.path.join(path, file_name + ".csv"))
        elif isinstance(debug_object, list):
            with open(os.path.join(path, file_name + ".txt"), "w") as f:
                f.write("\n".join(debug_object))
        else:
            with open(os.path.join(path, file_name + ".txt"), "w") as f:
                f.write(str(debug_object))

    def add_data(self, step: int | str, data: pd.DataFrame):
        """
        Add data to the exporter
        :param step: time step or other identifier (will be named 'step' in the multi-indexed dataframe)
        :param data: data as pandas data frame
        """
        if self.columns:
            data = data[self.columns]
        tuples = [(step, *rest) for rest in data.index]
        data.index = pd.MultiIndex.from_tuples(tuples,
                                               names=['step'] + data.index.names if data.index.names else ['step'])
        self.data = pd.concat([self.data, data])

    def read_data(self, steps: list[int | str] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types=None, columns=None) -> pd.DataFrame:
        """
        Read data from file/memory
        :param steps: list of time steps to read
        :param busses: bus ids to read
        :param frequencies: frequencies to read
        :param elements: elements list of strings
        :param element_types: list of objects of ElementType or integer.
        If element_type is 0 (or line), line currents are returned along with bus voltages
        if element_type is 1 (or load), data for loads is returned
        if element_type is 2 (or generator), data for generators is returned
        if element_type is 3 (or vsource), data for voltage sources is returned
        :param columns: dataframe columns
        :return: data as pandas dataframe (multi-indexed)
        """
        filtered_data = DataStore.filter_dataframe(self.data, steps, busses, frequencies, elements, element_types,
                                                   columns)
        return filtered_data

    def close(self):
        """
        Close the exporter
        """
        pass

    @staticmethod
    def filter_dataframe(data: pd.DataFrame, steps: list[int | str] = None, busses: list[int] = None,
                         frequencies: list[float | int] = None,
                         elements: list[str] = None, element_types=None, columns=None) -> pd.DataFrame:
        """
        Filter the given multi-indexed pandas dataframe
        :param data: pandas dataframe to filter
        :param steps: list of time steps to read
        :param busses: bus ids to read
        :param frequencies: frequencies to read
        :param elements: elements list of strings
        :param element_types: list of objects of ElementType or integer.
        If element_type is 0 (or line), line currents are returned along with bus voltages
        if element_type is 1 (or load), data for loads is returned
        if element_type is 2 (or generator), data for generators is returned
        :param columns: dataframe columns
        :return: data as pandas dataframe (multi-indexed)
        """
        if data.index.empty:
            logger.error('Attempting to filter dataframe. Data empty; no filter applied')
            return data

        # Filter by 'step' if provided
        if steps is not None:
            data = data[data.index.get_level_values('step').isin(steps)]

        # Filter by 'bus' if provided
        if busses is not None:
            data = data[data.index.get_level_values('bus').isin(busses)]

        # Filter by 'element' if provided
        if elements is not None:
            data = data[data.index.get_level_values('element').isin(elements)]

        # Filter by 'element_type', allowing for both enum and int values
        if element_types is not None:
            # Convert enum to its value if needed
            from OpenDSSWrapper.simulation import ElementType
            measurement_values = [m.value if isinstance(m, ElementType) else m for m in element_types]
            data = data[
                data.index.get_level_values('element_type').isin(measurement_values)]

        # Filter by 'Freq' if provided
        if frequencies is not None:
            data = data[data.index.get_level_values('freq').isin(frequencies)]

        # more efficient implementation that however requires sorting
        # # Convert MeasurementType enum to integer values if necessary
        # if element_types is not None:
        #     element_types = [m.value if isinstance(m, MeasurementType) else m for m in element_types]
        #
        # # Use slice(None) if a parameter is None, else use the parameter itself for slicing
        # steps_slice = slice(None) if steps is None else slice(min(steps), max(steps))
        # busses_slice = slice(None) if busses is None else slice(min(busses), max(busses))
        # elements_slice = slice(None) if elements is None else elements
        # element_types_slice = slice(None) if element_types is None else element_types
        # frequencies_slice = slice(None) if frequencies is None else frequencies
        #
        # # Filter the DataFrame using .loc with slices
        # filtered_data = self.data.loc[
        #                 (steps_slice, busses_slice, elements_slice, element_types_slice, frequencies_slice),
        #                 :
        #                 ]

        # drop undesired columns
        if columns:
            data = data[columns]
        return data

    @staticmethod
    def get_bus_data(df: pd.DataFrame, phases: list[int] = None) -> pd.DataFrame:
        """
        Get bus voltages and bus currents (i.e. sum of all load, generator and vsource currents attached to this bus)
        from the given dataframe

        :param df: multi-indexed pandas Dataframe
        :param phases: list of phases to consider, default None (all phases)
        :return: dataframe with bus voltages and currents
        """
        # Ensure phases are correctly formatted
        if phases is None:
            phases = ['1', '2', '3']
        else:
            phases = [str(phase) for phase in phases]

        current_mag_columns = [f'i{phase}' for phase in phases]
        current_angle_columns = [f'iangle{phase}' for phase in phases]
        current_columns = current_mag_columns + current_angle_columns
        voltage_columns = [f'v{phase}' for phase in phases] + [f'vangle{phase}' for phase in phases]

        # Function to convert currents to complex representation for a given phase
        def to_complex(row, phase):
            magnitude = row[f'i{phase}']
            angle = np.radians(row[f'iangle{phase}'])  # Convert angle to radians
            return magnitude * np.exp(1j * angle)

        # Adjustments for multi-indexed DataFrame
        # Assuming 'element_type' is part of the multi-index
        voltages = df.loc[df.index.get_level_values('element_type') == 0].copy()
        currents = df.loc[df.index.get_level_values('element_type').isin([1, 2, 3])].copy()

        # Initialize current columns to 0 in complex form for selected phases
        for col in current_columns:
            voltages[col] = 0 + 0j

        if not currents.empty:
            for phase in phases:
                col = f'i{phase}'
                currents[col + '_complex'] = currents.apply(lambda row: to_complex(row, phase), axis=1)

            # Adjust GroupBy for multi-index
            agg_dict = {col + '_complex': 'sum' for col in current_mag_columns}
            currents_sum = currents.groupby(level=['step', 'bus', 'freq']).agg(agg_dict).reset_index()

            # Merge needs adjustment for multi-index
            merged_df = pd.merge(voltages.reset_index(), currents_sum, on=['step', 'bus', 'freq'],
                                 how='left').set_index(['step', 'bus', 'element', 'element_type', 'freq'])

            for phase in phases:
                col = f'i{phase}'
                merged_df[col] = np.abs(merged_df[col + '_complex'].fillna(0))
                merged_df[f'iangle{phase}'] = np.angle(merged_df[col + '_complex'].fillna(0), deg=True)
                merged_df.drop(columns=[col + '_complex'], inplace=True)
        else:
            merged_df = voltages

        selected_columns = voltage_columns + current_columns
        non_duplicate_indices = ~merged_df.index.duplicated(keep=False)
        result_df = merged_df[non_duplicate_indices][selected_columns]

        # Ensure the result is multi-indexed as required
        result_df = result_df.reset_index().set_index(['step', 'bus', 'element', 'element_type', 'freq'])
        return result_df

    @staticmethod
    def df_to_numpy(df: pd.DataFrame, data_format:str="channels_last") -> tuple[np.ndarray, tuple]:
        """
        Convert a multi-indexed pandas dataframe to a n-dimensional numpy array
        dimensions: step, freq, bus, columns
        sorts by bus, so ensure that bus is unique (eliminate duplicates caused by generator /load measurements)
        :param df: pandas dataframe
        :param data_format: "channels_last" or "channels_first"
        e.g. if channels_last, the data format will be (step, node, frequency, complex component)
        if channels_first (step, frequency, complex_component, node)
        :return: numpy array, tuple of axis descriptions
        """
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("data_format must be either channels_last or channels_first")
        df = df.sort_index(level=['step', 'bus', 'freq'], ascending=True)
        df = df.reset_index(['element', 'element_type'], drop=True)
        # Get unique values for dimensions
        steps = df.index.get_level_values('step').unique()
        elements = df.index.get_level_values('bus').unique()
        freqs = df.index.get_level_values('freq').unique()
        columns = df.columns
        df_grouped = df.groupby(['step', 'bus', 'freq']).agg('first')
        data_array = df_grouped.to_numpy()
        result_array = data_array.reshape((len(steps), len(elements), len(freqs), -1))
        if data_format == "channels_first":
            result_array = np.transpose(result_array, (0, 2, 3, 1))
        return result_array, (list(steps), list(elements), list(freqs), list(columns))

    def plot_data_3d(self, df: pd.DataFrame, sub_path: str = "", target_phase: int = 1, target_mode: str = 'Voltage',
                     pu: bool = False, drop_frequencies: list[int | float] = None):
        """
        Create a 3D bar plot for simulation results with frequencies on the x-axis, bus on the y-axis, and magnitude on the z-axis.
        The plot will be saved in the specified sub_path.
        :param df: Multi-indexed DataFrame with indices 'step', 'bus', 'element', 'element_type', 'freq'
                   and columns 'v1' for magnitudes and 'vangle1' for phase angles.
        :param sub_path: path to save the plot
        :param target_phase: phase to plot
        :param target_mode: 'Voltage' or 'Current' or 'Power' (apparent power)
        :param pu: if True, data is provided in per-unit and axis descriptions will be adjusted
        :param drop_frequencies: list of frequencies to drop from the plot
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        save_path = os.path.join(self.base_path, "tmp", sub_path)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        mode_map = {'voltage': 'v', 'v': 'v', 'current': 'i', 'c': 'i', 'power': 's', 'p': 's', 's': 's'}
        short_label = mode_map.get(target_mode.lower())

        magn_column = f'{short_label}{target_phase}'
        angle_column = f'{short_label}angle{target_phase}'
        df = self.filter_dataframe(df, columns=[magn_column, angle_column], element_types=[0])
        if drop_frequencies:
            df = df[~df.index.get_level_values('freq').isin(drop_frequencies)]
        # Preparing data
        grouped = df.groupby(level=['bus', 'freq'])
        xpos = []
        ypos = []
        zpos = []
        dz = []
        dz_angle = []

        for (bus, freq), group in grouped:
            magnitude = group[magn_column].values[0]  # only 1 value exists
            angle = group[angle_column].values[0]

            xpos.append(freq)
            ypos.append(bus)
            zpos.append(0)
            dz.append(magnitude)
            dz_angle.append(angle)

        # Normalize the phase angles to map to colors
        angle_norm = Normalize(min(dz_angle), max(dz_angle))
        angle_colors = plt.cm.viridis(angle_norm(dz_angle))

        ax.bar3d(xpos, ypos, zpos, 0.4, 0.4, dz, shade=True, color=angle_colors)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Bus')
        pu_suffix = ""
        if pu:
            pu_suffix = "pu"
        ax.set_zlabel(f'Magnitude [{short_label.upper()} {pu_suffix}]')
        ax.set_title(f'Simulation Results {sub_path} {short_label.upper()}{target_phase} {pu_suffix}')

        mappable = plt.cm.ScalarMappable(norm=angle_norm, cmap=plt.cm.viridis)
        mappable.set_array(dz_angle)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label('Phase Angle [degrees]')

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"simulation_3d_{short_label.upper()}{target_phase}{pu_suffix}.png"))
        plt.close()

    def plot_data_2d_heatmap(self, df: pd.DataFrame, sub_path: str = "", target_phase: int = 1,
                             target_mode: str = 'Voltage', pu: bool = False,
                             drop_frequencies: list[int | float] = None):
        """
        Create 2D heatmap plots for simulation results where the x-axis is frequencies, y-axis is nodes,
        and color represents magnitudes or phase angles.
        :param df: Multi-indexed DataFrame with indices 'step', 'bus', 'element', 'element_type', 'freq'
                   and columns like 'v1', 'vangle1' depending on phase and mode.
        :param sub_path: path to save the plot
        :param target_phase: phase to plot
        :param target_mode: 'Voltage', 'Current', or 'Power' (apparent power)
        :param pu: if True, indicates data is provided in per-unit
        :param drop_frequencies: list of frequencies to drop from the plot
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_path = os.path.join(self.base_path, "tmp", sub_path)
        mode_map = {'voltage': 'v', 'v': 'v', 'current': 'i', 'c': 'i', 'power': 's', 'p': 's', 's': 's'}
        short_label = mode_map.get(target_mode.lower())

        if not short_label:
            logger.error(f'Unknown mode {target_mode}')
            return

        magn_column = f'{short_label}{target_phase}'
        angle_column = f'{short_label}angle{target_phase}'

        df = self.filter_dataframe(df, columns=[magn_column, angle_column], element_types=[0])
        if drop_frequencies:
            df = df[~df.index.get_level_values('freq').isin(drop_frequencies)]
        df.reset_index(inplace=True)
        pivot_magn = df.pivot(index='bus', columns='freq', values=magn_column)
        pivot_angle = df.pivot(index='bus', columns='freq', values=angle_column)

        # Plot magnitude heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_magn, annot=False, cmap='viridis',
                    cbar_kws={'label': f'Magnitude [{short_label.upper()} {"pu" if pu else ""}]'})
        plt.title(f'Magnitude Heatmap for {target_mode} Phase {target_phase} {"Per Unit" if pu else ""}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Node')
        plt.savefig(
            os.path.join(save_path, f"heatmap_magnitude_{short_label.upper()}{target_phase}{'_pu' if pu else ''}.png"))
        plt.close()

        # Plot angle heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_angle, annot=False, cmap='viridis', cbar_kws={'label': 'Phase Angle [degrees]'})
        plt.title(f'Phase Angle Heatmap for {target_mode} Phase {target_phase} {"Per Unit" if pu else ""}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Node')
        plt.savefig(
            os.path.join(save_path, f"heatmap_angle_{short_label.upper()}{target_phase}{'_pu' if pu else ''}.png"))
        plt.close()


class PickleStore(DataStore):
    """
    Export Data in Pickle format.
    Caution: use only for small simulations or debugging
    Data is accumulated in a multiprocessing queue and written to the file when the exporter is closed.
    This leads to high memory usage and long file write times at the end of the simulation.
    """

    def __init__(self, base_path: str, file_name: str, columns: list = None):
        """
        Initialize Pickle exporter
        :param base_path: base path
        :param file_name: file name
        :param columns: columns to export, default None (all columns)
        """
        super().__init__(base_path, columns)
        self.file_name = file_name
        manager = Manager()
        self.queue = manager.Queue()

    def add_data(self, step: int | str, data: pd.DataFrame):
        if self.columns:
            data = data[self.columns]
        self.queue.put((step, data))

    def close(self):
        data_to_pickle = pd.DataFrame()
        while not self.queue.empty():
            step, data = self.queue.get()
            tuples = [(step, *rest) for rest in data.index]
            data.index = pd.MultiIndex.from_tuples(tuples,
                                                   names=['step'] + data.index.names if data.index.names else ['step'])
            data_to_pickle = pd.concat([data_to_pickle, data])
        data_to_pickle.to_pickle(os.path.join(self.base_path, self.file_name))

    def read_data(self, steps: list[int | str] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types=None, columns=None):
        data = pd.read_pickle(os.path.join(self.base_path, self.file_name))
        filtered_data = DataStore.filter_dataframe(data, steps, busses, frequencies, elements, element_types, columns)
        return filtered_data


class HDF5Store(DataStore):
    """
    Export Data in HDF5 format
    HDF5 only supports multiprocessing using MPI which I try to avoid since programs need to be specifically run
    through MPI and implementation differs for each OS.
    Therefore, a multiprocessing queue is used to accumulate data and write it to the file when the exporter is closed.
    This leads to high memory usage and long file write times at the end of the simulation.
    """

    def __init__(self, base_path: str, file_name: str, columns: list = None):
        super().__init__(base_path, columns)
        self.columns = columns
        self.base_path = base_path
        self.file_name = file_name
        manager = Manager()
        self.buffer = manager.list()

    def add_data(self, step: int | str, data: pd.DataFrame):
        """
        Add data to the exporter
        If columns are specified, only these columns are exported
        :param step: time step
        :param data: data dictionary of bus id and data frame
        :return:
        """
        if self.columns:
            data = data[self.columns]
        tuples = [(step, *rest) for rest in data.index]
        data.index = pd.MultiIndex.from_tuples(tuples,
                                               names=['step'] + data.index.names if data.index.names else ['step'])
        self.buffer.append(data)

    def _write_data(self):
        file_path = os.path.join(self.base_path, self.file_name)
        combined_df = pd.concat(list(self.buffer), ignore_index=True)
        combined_df.to_hdf(file_path, key='data', mode='w', format='table', data_columns=True, complevel=9,
                           complib='blosc')

    def close(self):
        self._write_data()
        # Clear the buffer by replacing it with a new Manager list
        self.buffer = Manager().list()

    def read_data(self, steps: list[int] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types=None, columns=None):
        file_path = os.path.join(self.base_path, self.file_name)
        with pd.HDFStore(file_path, 'r') as hdf_file:
            df = hdf_file['data']
            filtered_data = DataStore.filter_dataframe(df, steps, busses, frequencies, elements, element_types,
                                                       columns)
        return filtered_data


class SQLiteStore(DataStore):
    """
    Export Data to SQLite database
    Caution: SQLite does not allow parallel writes, so a lock mechanism is used which slows down processing
    """

    def __init__(self, base_path: str, file_name: str, columns: list, batch_size=10000, step_dtype='INTEGER',
                 table_name='data'):
        """
        Initialize SQL exporter
        :param base_path: base path
        :param file_name: file name
        :param columns: columns (required for this data store)
        :param batch_size: batch size for bulk inserts (currently not used)
        :param step_dtype: data type for step column (INTEGER or TEXT)
        :param table_name: name of the table
        """
        super().__init__(base_path, columns)
        self.file_name = file_name
        self.lock = Lock()
        self.step_dtype = step_dtype
        # Define table schema based on columns
        columns_sql = ', '.join([f"{col} REAL" for col in self.columns])
        self.table_name = table_name
        self.table_schema = f"CREATE TABLE IF NOT EXISTS {self.table_name} (step {step_dtype}, bus INTEGER, element TEXT, element_type INTEGER, freq REAL, {columns_sql})"

        # Initialize database and create table
        self._init_db()

    def _init_db(self):
        # Initialize persistent database connection
        self.conn = sqlite3.connect(os.path.join(self.base_path, self.file_name))
        with self.conn:
            self.conn.execute(self.table_schema)

    def add_data(self, step: int | str, data: pd.DataFrame):
        if isinstance(step, int) and not self.step_dtype == 'INTEGER':
            raise ValueError(f"step type is {type(step)} but step_dtype is {self.step_dtype}.")
        if isinstance(step, str) and not self.step_dtype == 'TEXT':
            raise ValueError(f"step type is {type(step)} but step_dtype is {self.step_dtype}.")
        # Reset index to turn the multi-level index into columns
        if self.columns:
            data = data[self.columns]
        df_reset = data.reset_index()
        # Add 'step' column
        df_reset.insert(0, 'step', step)
        rows_to_insert = []
        for row in df_reset.itertuples(index=False, name=None):
            rows_to_insert.append(tuple(row))

        # Perform bulk insert within a transaction
        if rows_to_insert:
            value_placeholder = ','.join('?' * len(rows_to_insert[0]))
            with self.lock:  # acquire lock to prevent concurrent writes (not supported by SQLite)
                with self.conn:
                    self.conn.executemany(f'INSERT INTO {self.table_name} VALUES ({value_placeholder})', rows_to_insert)

    def close(self):
        self.conn.close()

    def read_data(self, steps: list[int | str] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types=None, columns=None):
        step_str = "AND step IN ({})".format(','.join(map(str, steps))) if steps else ""
        bus_str = "AND bus IN ({})".format(','.join(map(str, busses))) if busses else ""
        freq_str = "AND freq IN ({})".format(','.join(map(str, frequencies))) if frequencies else ""
        element_types_str = "AND element_type in ({})".format(
            ','.join(map(str, element_types))) if element_types else ""
        element_str = "AND element in ({})".format(','.join(map(str, elements))) if elements else ""
        column_str = '*'
        if columns is not None:
            column_str = ', '.join(columns)

        with sqlite3.connect(os.path.join(self.base_path, self.file_name)) as conn:
            query = f"SELECT {column_str} FROM {self.table_name} WHERE 1=1 {step_str} {bus_str} {freq_str} {element_str} {element_types_str}"
            df = pd.read_sql_query(query, conn)
            # drop columns that are not in self.columns
            if columns is not None:
                columns = columns + ['step', 'bus', 'element', 'element_type', 'freq']  # keep some
                df = df[columns]
        df_reindexed = df.set_index(['step', 'bus', 'element', 'element_type', 'freq'])
        return df_reindexed


connection_pool = None


class PostgreSQLStore(DataStore):
    """
    Manually create database first!
    PostgreSQL allows parallel writes and is more robust than SQLite.
    This should be your preferred choice for large simulations.
    """

    def __init__(self, base_path: str, columns: list, db_config: dict, step_dtype='INTEGER', table_name='data'):
        """
        Initialize SQL exporter for PostgreSQL
        :param base_path: base path
        :param columns: columns (required for this data store)
        :param db_config: Dictionary containing database configuration parameters
                          Example: {"dbname": "mydb", "user": "username", "password": "password", "host": "localhost"}
        :param step_dtype: data type for step column (INTEGER or TEXT)
        :param table_name: name of the table
        """
        super().__init__(base_path, columns)
        self.db_config = db_config
        self.step_dtype = step_dtype
        # Define table schema based on columns
        columns_sql = ', '.join([f"{col} REAL" for col in self.columns])
        self.table_schema = f"CREATE TABLE IF NOT EXISTS {table_name}  (step {step_dtype}, bus INTEGER, element VARCHAR(8), element_type INTEGER, freq REAL, {columns_sql})"
        self.table_name = table_name
        # Initialize database and create table
        self._init_db()

    def _check_connection(self, connection):
        return ThreadedConnectionPool(1, 10, **self.db_config)


    def _init_db(self):
        # Initialize persistent database connection
        global connection_pool
        connection_pool = self._check_connection(connection_pool)
        with connection_pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self.table_schema)
                conn.commit()
            connection_pool.putconn(conn)

    def add_data(self, step: int | str, data: pd.DataFrame):
        if isinstance(step, int) and not self.step_dtype == 'INTEGER':
            raise ValueError(f"step type is {type(step)} but step_dtype is {self.step_dtype}.")
        if isinstance(step, str) and not self.step_dtype == 'TEXT':
            raise ValueError(f"step type is {type(step)} but step_dtype is {self.step_dtype}.")
        # Reset index to turn the multi-level index into columns
        if self.columns:
            data = data[self.columns]
        df_reset = data.reset_index()
        # Add 'step' column
        df_reset.insert(0, 'step', step)
        rows_to_insert = [tuple(row) for row in df_reset.itertuples(index=False, name=None)]

        # Perform bulk insert within a transaction
        if rows_to_insert:
            columns_placeholder = ', '.join(['step', 'bus', 'element', 'element_type', 'freq'] + self.columns)
            insert_query = f'INSERT INTO {self.table_name} ({columns_placeholder}) VALUES %s'
            global connection_pool
            connection_pool = self._check_connection(connection_pool)
            with connection_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    execute_values(cursor, insert_query, rows_to_insert, template=None, page_size=100)
                    conn.commit()
                connection_pool.putconn(conn)


    def close(self):
        global connection_pool
        connection_pool = self._check_connection(connection_pool)
        connection_pool.closeall()
        connection_pool = None

    def read_data(self, steps: list[int | str] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types: list[int] = None, columns: list[str] = None):
        step_str = "AND step IN ({})".format(','.join(map(str, steps))) if steps else ""
        bus_str = "AND bus IN ({})".format(','.join(map(str, busses))) if busses else ""
        freq_str = "AND freq IN ({})".format(','.join(map(str, frequencies))) if frequencies else ""
        element_types_str = "AND element_type in ({})".format(
            ','.join(map(str, element_types))) if element_types else ""
        element_str = "AND element in ({})".format(','.join(map(str, elements))) if elements else ""
        column_str = '*'
        if columns is not None:
            columns = columns + ['step', 'bus', 'element', 'element_type', 'freq']  # keep some
        if columns is not None:
            column_str = ', '.join(columns)

        query = f"SELECT {column_str} FROM {self.table_name} WHERE 1=1 {step_str} {bus_str} {freq_str} {element_str} {element_types_str}"
        logger.debug(f"Querying database: {query}")
        global connection_pool
        connection_pool = self._check_connection(connection_pool)
        with connection_pool.getconn() as conn:
            df = pd.read_sql_query(query, conn)
            connection_pool.putconn(conn)
        # drop columns that are not in self.columns
        if columns is not None:
            df = df[columns]
        df_reindexed = df.set_index(['step', 'bus', 'element', 'element_type', 'freq'])
        return df_reindexed

    def empty_table(self):
        """
        Drop the table (self.table_name) if it exists.
        :return:
        """
        global connection_pool
        connection_pool = self._check_connection(connection_pool)
        with connection_pool.getconn() as conn:
            with conn.cursor() as cursor:
                logger.info(
                    f"Emptying table {self.table_name} hosted on {self.db_config['host']}:{self.db_config['port']}")
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                cursor.execute(self.table_schema)
                conn.commit()
            connection_pool.putconn(conn)


class CSVStore(DataStore):
    """
    Export Data in CSV format
    Human-readable format, but leads to many files. Make sure the file system can handle it.
    """

    def __init__(self, base_path: str, sub_path: str, grid_name: str, columns: list = None):
        """
        Export Data in CSV format as a single CSV file per step. The resulting CSV file is multi-indexed with indices
        ['bus', 'element', 'element_type', 'freq']
        :param base_path: base path
        :param sub_path: sub path to export to (so that a million files don't end up in the base folder)
        :param grid_name: grid name
        :param columns: columns to export, default None (all columns)
        """
        super().__init__(base_path, columns)
        self.sub_path = sub_path
        self.grid_name = grid_name

    def add_data(self, step: int | str, data: pd.DataFrame):
        path = os.path.join(self.base_path, self.sub_path, f"{str(step).split('-')[0]}")
        os.makedirs(path, exist_ok=True)
        if self.columns is not None:
            data = data[self.columns]
        filename = f'{self.grid_name}_step_{step}.csv'
        # Ensure index is saved
        data.to_csv(os.path.join(path, filename), index=True)

    def close(self):
        # not necessary here
        pass

    def read_data(self, steps: list[int | str] = None, busses: list[int] = None, frequencies: list[float | int] = None,
                  elements: list[str] = None, element_types=None, columns=None):
        df_combined = pd.DataFrame()
        if steps is None:
            steps = [filename.split('_')[3].replace('.csv', '') for filename in
                     os.listdir(os.path.join(self.base_path, self.sub_path)) if filename.endswith('.csv')]
        for step in steps:
            filename = f'{self.grid_name}_step_{step}.csv'
            sub_path = os.path.join(self.sub_path, str(step).split('-')[0])

            # Read CSV without specifying index_col to keep all data in columns
            df = pd.read_csv(os.path.join(self.base_path, sub_path, filename))

            # Reconstruct multi-index including step, bus, element, element_type, and freq
            df.set_index(['bus', 'element', 'element_type', 'freq'], append=False, inplace=True)
            df.index = pd.MultiIndex.from_tuples([(step,) + idx for idx in df.index],
                                                 names=['step', 'bus', 'element', 'element_type', 'freq'])

            df_combined = pd.concat([df_combined, df])

        filtered_data = self.filter_dataframe(df_combined, steps, busses, frequencies, elements, element_types, columns)
        return filtered_data
