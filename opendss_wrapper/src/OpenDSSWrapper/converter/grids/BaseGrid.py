import pandapower as pp
import pandas as pd
import numpy as np


class BaseGrid:
    descriptive_nodes = {}
    name = None
    short_name = None
    n_nodes = 0
    SWITCH_STRING = 'std_switch'

    def __init__(self):
        self.descriptive_nodes = {}
        self.n_nodes = 0
        self.name = "Base Grid"
        self.pp_net = self._get_pandapower()
        self.add_std_types()

    def add_std_types(self):
        """
        Add standard types to the pandapower network
        :return:
        """
        # Line properties between the buses in the net, when used as switch
        line_data = {'c_nf_per_km': 0.0001,                        # line capacitance (line-to-earth)      nF/km
                     'r_ohm_per_km': 0.0001,                      # line resistance                       Ω/km
                     'x_ohm_per_km': 0.0001,                      # line reactance                        Ω/km
                     'max_i_ka': 0.06,                          # maximal thermal current               kA
                     'type': 'cs'                               # cs - underground cable system
                     }

        # Creation of the standard type stdswitch
        pp.create_std_type(self.pp_net, line_data, name=self.SWITCH_STRING, element='line')

    def _get_pandapower(self):
        """
        Do not use this method from other classes, use grid.pp_net instead
        otherwise standard types are not added
        :return:
        """
        return pp.create_empty_network(self.name)

    def get_load_profiles(self, dataset='train'):
        """
        Get load profiles for the given dataset
        :param dataset: one of 'train', 'validation', 'test'
        :return:
        """
        raise NotImplementedError

    def get_generator_profiles(self, dataset='train'):
        """
        Get generator profiles for the given dataset
        :param dataset: one of 'train', 'validation', 'test'
        :return:
        """
        raise NotImplementedError

    def get_load_spectra(self, dataset='train'):
        """
        Get load spectra for the given dataset
        :param dataset: one of 'train', 'validation', 'test'
        :return:
        """
        raise NotImplementedError

    def get_generator_spectra(self, dataset='train'):
        """
        Get generator spectra for the given dataset
        :param dataset: one of 'train', 'validation', 'test'
        :return:
        """
        raise NotImplementedError

    def get_nodes_as_list_of_int(self):
        return [int(n) for n in self.descriptive_nodes.keys()]

    def get_description_for_node(self, node: int):
        """
        Get descriptive string for given node
        :param node:
        :return: description if it exists, else string of node
        """
        if len(self.descriptive_nodes) == 0:
            return str(node)
        try:
            return self.descriptive_nodes[str(node)]
        except KeyError:
            return str(node)

    def get_base_kvs(self) -> dict[str, float]:
        """
        Get Base voltages as dictionary of bus as key and base voltage
        in kilovolt and as phase-to-phase voltage as value
        :return: base voltages
        """
        base_kvs = {}

        # Iterate over the rows of net.bus
        for b, bc in self.pp_net.bus.iterrows():
            bus = str(b)
            base_kvs[bus] = bc.vn_kv
        return base_kvs

    def get_bus_for_load(self, load_name:str):
        """
        Get the bus number for the given load name
        :param load_name: Name of the load
        :return: bus number
        """
        df = self.pp_net.load[['name', 'bus']]
        df['name'] = df['name'].str.replace("Load ", "")
        df['name'] = df['name'].str.lower()
        load_name = load_name.lower()
        # Check if load_name is in the stripped names
        if load_name in df['name'].values:
            # Retrieve the 'bus' value for the given load_name
            bus_number = df.loc[df['name'] == load_name, 'bus'].iloc[0]
            return bus_number
        else:
            return "Load name not found"

    def get_bus_for_generator(self, generator_name:str):
        """
        Get the bus number for the given generator name
        :param generator_name: Name of the generator
        :return: bus number
        """
        df = self.pp_net.sgen[['name', 'bus']]
        df['name'] = df['name'].str.replace(' ', '')
        df['name'] = df['name'].str.lower()
        generator_name = generator_name.lower()
        # Check if load_name is in the stripped names
        if generator_name in df['name'].values:
            # Retrieve the 'bus' value for the given load_name
            bus_number = df.loc[df['name'] == generator_name, 'bus'].iloc[0]
            return bus_number
        else:
            return "Generator name not found"

    @staticmethod
    def __get_grid_impedance(grid_name: str, f_max: float):
        """
        Load an impedance measurement from morEnergy ONIS for the grid connecting point
        :param grid_name: name of the grid, the impedance measurement file / entry in DB should be named like this.
        :param f_max: maximum frquency to consider impedance data
        :return: a pandas dataframe containing three phase impedance measurements in polar coords.
        """
        import pandas as pd
        import os
        from OpenDSSWrapper.tools.config import data_path
        imp_path = os.path.join(data_path, f'{grid_name}.csv')
        if os.path.exists(imp_path):
            impedance_data = pd.read_csv(imp_path).T
            first_row = impedance_data.iloc[0].values
            impedance_data.columns = first_row
            impedance_data = impedance_data.iloc[1:]
            return impedance_data.drop(columns=[col for col in impedance_data.columns if int(col) > f_max]).T
        else:
            return None

    @staticmethod
    def __convert_grid_impedance_df_to_complex(df):
        def to_complex(amplitude, phase):
            return amplitude * np.exp(1j * np.pi*phase/180)
        df_complex = pd.DataFrame({
            'z1': to_complex(df.iloc[:, 0], df.iloc[:, 1]),
            'z2': to_complex(df.iloc[:, 2], df.iloc[:, 3]),
            'z3': to_complex(df.iloc[:, 4], df.iloc[:, 5]),
        }, index=df.index)
        return df_complex

    @staticmethod
    def __revert_complex_grid_impedance_to_df(df, no_neutral=False):
        if no_neutral:
            df_reverted = pd.DataFrame({
                'z1': np.abs(df['z1']),
                'z1angle': np.angle(df['z1'], deg=True),
                'z2': np.abs(df['z2']),
                'z2angle': np.angle(df['z2'], deg=True),
                'z3': np.abs(df['z3']),
                'z3angle': np.angle(df['z3'], deg=True),
            }, index=df.index)
        else:
            df_reverted = pd.DataFrame({
                'z1': np.abs(df['z1']),
                'z1angle': np.angle(df['z1'], deg=True),
                'z2': np.abs(df['z2']),
                'z2angle': np.angle(df['z2'], deg=True),
                'z3': np.abs(df['z3']),
                'z3angle': np.angle(df['z3'], deg=True),
                'zN': np.abs(df['zn']),
                'zNangle': np.angle(df['zn'], deg=True),
            }, index=df.index)
        return df_reverted

    @staticmethod
    def __compute_equivalent_grid_impedance(df_ll, df_ln):
        default_eqv_columns = ['z1', 'z2', 'z3', 'zn']
        df_combined = pd.DataFrame(columns=default_eqv_columns)
        if df_ll is None:
            df_ln.columns = ['z1n', 'z2n', 'z3n']
            df_combined['z1'] = df_ln['z1n']
            df_combined['z2'] = df_ln['z2n']
            df_combined['z3'] = df_ln['z3n']

            return df_combined
        else:
            df_ln.columns = ['z1n', 'z2n', 'z3n']
            df_ll.columns = ['z12', 'z23', 'z31']
            df_combined['z1'] = 0.5*(df_ll['z12'] - df_ll['z23'] + df_ll['z31'])
            df_combined['z2'] = 0.5*(df_ll['z23'] - df_ll['z31'] + df_ll['z12'])
            df_combined['z3'] = 0.5*(df_ll['z31'] - df_ll['z12'] + df_ll['z23'])
            'average of the 3 unbalanced neutral impedances'
            df_combined['zn'] = (df_ln['z1n'] - df_combined['z1'] +
                                 df_ln['z2n'] - df_combined['z2'] +
                                 df_ln['z3n'] - df_combined['z3'])/3

            return df_combined

    def _handle_grid_impedance(self, grid_name, f_max):
        add_grid_imp_ll = self.__get_grid_impedance(f"{grid_name}_imp_ll", f_max)
        add_grid_imp_ln = self.__get_grid_impedance(f"{grid_name}_imp_ln", f_max)

        if add_grid_imp_ll is not None and add_grid_imp_ln is not None:
            df_ll = self.__convert_grid_impedance_df_to_complex(add_grid_imp_ll)
            df_ln = self.__convert_grid_impedance_df_to_complex(add_grid_imp_ln)
            df_grid_imp = self.__compute_equivalent_grid_impedance(df_ll, df_ln)
            return self.__revert_complex_grid_impedance_to_df(df_grid_imp).T
        elif add_grid_imp_ln is not None:
            df_ln = self.__convert_grid_impedance_df_to_complex(add_grid_imp_ln)
            df_grid_imp = self.__compute_equivalent_grid_impedance(None, df_ln)
            return self.__revert_complex_grid_impedance_to_df(df_grid_imp, no_neutral=True).T
        else:
            print("no grid impedance found. Use default assumption.")
            return None

    @staticmethod
    def get_cable_resonances(net: pp.pandapowerNet, pos_neg_seq='pos'):
        """
        compute all occurring cable resonances and print them
        :param net: currently formed grid.
        :param pos_neg_seq: which sequence of cable resonances to consider.
        """
        for idx, line in net.line.iterrows():
            length = line['length_km']
            l = line['x_ohm_per_km'] / (2*np.pi*50)
            c = line['c_nf_per_km'] * 10**-9
            f_res = 1/(2*np.pi*length*np.sqrt(l*c/2))
            print(f"Line from {line['from_bus']} to {line['to_bus']}: length={length}, f_res={f_res/1000000} MHz")
        pass

    @staticmethod
    def _refresh_line_pp(net: pp.pandapowerNet, line):
        name = line.loc['name']
        std_type = line.std_type
        from_bus = line.from_bus
        to_bus = line.to_bus
        length_km = line.length_km
        df = line.df
        parallel = line.parallel
        in_service = line.in_service
        pp.create_line(net, name=name, from_bus=from_bus, to_bus=to_bus, length_km=length_km, in_service=in_service,
                       std_type=std_type, parallel=parallel, df=df)


