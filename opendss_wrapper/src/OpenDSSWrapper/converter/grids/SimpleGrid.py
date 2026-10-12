import numpy as np
import pandapower as pp

from OpenDSSWrapper.converter import BaseGrid
from OpenDSSWrapper.harmonic_spectra import SpectrumConfig, MeasurementType
from OpenDSSWrapper.power_profiles import PowerProfileConfig, PowerProfileSubCategory, PowerProfileCategory


class SimpleGrid(BaseGrid):
    def __init__(self):
        super().__init__()
        self.name = "Simple Grid"
        self.n_nodes = 2
        self.descriptive_nodes = {0: "Source", 1: "Load"}

    def _get_pandapower(self):
        net = pp.create_empty_network(self.name)
        bus0 = pp.create_bus(net, name="Source", vn_kv=0.4)
        bus1 = pp.create_bus(net, name="Bus1", vn_kv=0.4)
        pp.create_ext_grid(net, bus=bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=0.974529347,
                           s_sc_min_mva=0.974529347, rx_max=0.245307125, rx_min=0.245307125)
        l1 = pp.create_load(net, name="Load 1", bus=bus1, p_mw=0.1, q_mvar=0.05)
        line_data = {'c_nf_per_km': 150,  # line capacitance (line-to-earth)      nF/km
                     'r_ohm_per_km': 1.78,  # line resistance                       Ω/km
                     'x_ohm_per_km': 0.12,  # line reactance                        Ω/km
                     'max_i_ka': 0.06,  # maximal thermal current               kA
                     'type': 'cs'  # cs - underground cable system
                     }
        pp.create_std_type(net, line_data, name='LtE10', element='line')
        pp.create_line(net, name='Line1', from_bus=bus0, to_bus=bus1, length_km=0.1, std_type="LtE10")
        return net

    def get_load_profiles(self, dataset='train'):
        ppc_residential = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                             PowerProfileSubCategory.RESIDENTIAL, step_size_ms=60000,
                                             scaling_factor=1)
        load_profiles = {}
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        for load_name in load_names:
            load_profiles[load_name] = ppc_residential
        return load_profiles

    def get_generator_profiles(self, dataset='train'):
        generator_profiles = {}
        rng = np.random.default_rng()
        # voltage source profile is centered around 1 with a standard deviation of 0.15
        # default profile is defined as always 1, so we need to add rng to it
        vsource_profile = PowerProfileConfig('Default', PowerProfileCategory.SOURCE,
                                             PowerProfileSubCategory.EXTERNAL_GRID, step_size_ms=60000,
                                             scaling_factor=1, normalize=False, rng=rng, rng_mean=1, rng_scale=0.05)
        generator_profiles["source"] = vsource_profile
        return generator_profiles

    def get_load_spectra(self, dataset='train'):
        return {}

    def get_generator_spectra(self, dataset='train'):
        # add a spectrum to external grid
        generator_spectra = {}
        spectrum_config = SpectrumConfig("MicrogridlabSupplySource", manufacturer=None, category="GENERATION",
                                         subcategory="GRID", measurement_type=MeasurementType.VOLTAGE)
        generator_spectra["Source"] = spectrum_config
        return generator_spectra
