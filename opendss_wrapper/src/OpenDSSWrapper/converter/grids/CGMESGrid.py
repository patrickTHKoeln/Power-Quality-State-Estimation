import configparser
import os

import numpy as np

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.converter import cgmes_to_pandapower, cimpyorm_to_pandapower
from OpenDSSWrapper.harmonic_spectra import SpectrumConfig
from OpenDSSWrapper.power_profiles import PowerProfileConfig, PowerProfileCategory, PowerProfileSubCategory
import pandapower as pp

config = configparser.ConfigParser()
config.sections()
config_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
config.read(os.path.join(config_path, 'config.ini'))
data_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['data_path'])))
cgmes_path = os.path.join(data_path, 'CGMES')


class CGMESGrid(BaseGrid):
    """
    Class to represent a general CGMES grid, spectra and load profiles are randomly selected
    """

    def __init__(self, name, cgmes_files, load_inj_file=None, gen_inj_file=None):
        self.name = name
        self.cgmes_files = cgmes_files
        self.load_inj_file = load_inj_file
        self.gen_inj_file = gen_inj_file
        super().__init__()
        self.n_nodes = len(self.pp_net.bus)

    def _get_pandapower(self):
        net = cgmes_to_pandapower(self.cgmes_files)
        net = cimpyorm_to_pandapower(net, self.load_inj_file, self.gen_inj_file)
        # Add custom LineTypes for MV-Switches
        lt = net.std_types['line']['NA2XS2Y 1x95 RM/25 12/20 kV']

        line_data = {'c_nf_per_km': lt['c_nf_per_km'], 'r_ohm_per_km': lt['r_ohm_per_km'],
                     'x_ohm_per_km': lt['x_ohm_per_km'], 'max_i_ka': lt['max_i_ka'],
                     'type': 'cs'}
        pp.create_std_type(net, line_data, name='NA2XS2Y', element='line')

        list_busses = []
        # Add specific lines for MV Section instead of switches
        for _, sw in net.switch.iterrows():
            list_busses.append(sw.bus)
            pp.create_line(net, name=sw['name'], from_bus=sw.bus,
                           to_bus=sw.element, length_km=0.3, std_type="NA2XS2Y")

        pp.drop_switches_at_buses(net, list_busses)

        line_idxs_to_drop = []
        # update lines after changes to std_types
        for l in net.line.iterrows():
            line_idxs_to_drop.append(l[0])
            self._refresh_line_pp(net, l[1])
        pp.drop_lines(net, line_idxs_to_drop)
        return net

    def get_load_profiles(self, dataset='train'):
        load_profiles = {}
        load_names = self.pp_net.load['name'].values
        # For now assume only residential loads
        rng = np.random.default_rng()
        random_func = rng.normal
        noise_parameters = {'loc': 0.85, 'scale': 0.05}
        # Load profile is randomly selected from the database based on the following configuration
        ppc_residential = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                             PowerProfileSubCategory.RESIDENTIAL, step_size_ms=60000,
                                             scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        # TODO: filter for EV, commercial (by name?)
        for load_name in load_names:
            load_profiles[load_name] = ppc_residential
        return load_profiles

    def get_generator_profiles(self, dataset='train'):
        generator_profiles = {}
        generator_names = self.pp_net.sgen['name'].values
        # For now assume only pv
        rng = np.random.default_rng()
        random_func = rng.normal
        noise_parameters = {'loc': 0.85, 'scale': 0.05}
        # Generator profile is randomly selected from the database based on the following configuration
        ppc_pv = PowerProfileConfig(None, PowerProfileCategory.GENERATOR,
                                    PowerProfileSubCategory.PHOTOVOLTAIC_SYSTEM, step_size_ms=60000,
                                    scaling_factor=1, random_func=random_func, noise_params=noise_parameters)

        for generator_name in generator_names:
            generator_profiles[generator_name] = ppc_pv

        return generator_profiles

    def get_load_spectra(self, dataset='train'):
        load_spectra = {}
        load_names = self.pp_net.load['name'].values
        default_config = SpectrumConfig()  # default config = no spectrum
        spectrum_config = SpectrumConfig(None, None, "ELECTRIC_VEHICLES", "ELECTRIC_CAR")
        for load_name in load_names:
            # random choice between default and electric vehicles
            load_spectra[load_name] = np.random.choice([default_config, spectrum_config])
        return load_spectra

    def get_generator_spectra(self, dataset='train'):
        generator_spectra = {}
        generator_names = self.pp_net.sgen['name'].values
        spectrum_config = SpectrumConfig(None, None, "GENERATION", "PHOTOVOLTAIC_SYSTEM")
        for generator_name in generator_names:
            generator_spectra[generator_name] = spectrum_config
        return generator_spectra


if __name__ == '__main__':
    example_cgmes = os.path.join(cgmes_path, '20240227_CGMES_example_geographical_location_profile.zip')
    gen_path = os.path.join(cgmes_path, 'Anonymisiertes_Netz_nur_einspeiser_mit_reservierungen', 'out.db')
    load_path = os.path.join(cgmes_path, 'Anonymisiertes_Netz_nur_lasten_mit_reservierungen', 'out.db')
    cg = CGMESGrid('example', example_cgmes, load_path, gen_path)
    pass
