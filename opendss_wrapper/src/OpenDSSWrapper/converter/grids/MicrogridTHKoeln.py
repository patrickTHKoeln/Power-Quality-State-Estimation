import numpy as np
import pandapower as pp

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.harmonic_spectra import SpectrumConfig, MeasurementType, DynamicSpectrumType
from OpenDSSWrapper.power_profiles import PowerProfileConfig, PowerProfileCategory, PowerProfileSubCategory
import OpenDSSWrapper.tools.config as cfg

class MicrogridTHKoeln(BaseGrid):
    def __init__(self, line1, line10, with_grid_impedance, ignore_source_angles):
        self.line1 = line1
        self.line10 = line10
        self.with_grid_impedance = with_grid_impedance
        self.ignore_source_angles = ignore_source_angles
        super().__init__()
        self.samples_val = None
        self.n_nodes = 15
        self.name = "Laboratory Microgrid of TH-Koeln"
        self.short_name = "microgrid_thk"
        self.descriptive_nodes = {'0': '0',
                                  '1': 'R1',
                                  '2': 'R2',
                                  '3': 'R3',
                                  '4': 'R4',
                                  '5': 'R5',
                                  '6': 'R6',
                                  '7': 'R7',
                                  '8': 'R8',
                                  '9': 'WR1',
                                  '10': 'WR2',
                                  '11': 'WR3',
                                  '12': 'WR4',
                                  '13': 'WR5',
                                  '14': 'WR6'
                                  }
        self.samples = None

    def _get_pandapower(self):
        """
        includes the TH-Koeln laboratory microgrid
        :return: net
        """

        # generator is connected? True/False
        slack = True
        generator = False

        # define pandapower empty network
        net = pp.create_empty_network()

        # define all net-buses
        bus0 = pp.create_bus(net, name='Bus 0', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(5, 10))
        bus1 = pp.create_bus(net, name='Bus R1', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(3, 9))
        bus2 = pp.create_bus(net, name='Bus R2', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(7, 9))
        bus3 = pp.create_bus(net, name='Bus R3', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(3, 8))
        bus4 = pp.create_bus(net, name='Bus R4', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(7, 8))
        bus5 = pp.create_bus(net, name='Bus R5', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(3, 7))
        bus6 = pp.create_bus(net, name='Bus R6', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(7, 7))
        bus7 = pp.create_bus(net, name='Bus R7', vn_kv=0.4, type='b', zone='THKoelnNetwork',
                             geodata=(3, 6))  # connect variable resistor here
        bus8 = pp.create_bus(net, name='Bus R8', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(7, 6))

        # define the buses for laboratory inverters
        bus_wr1 = pp.create_bus(net, name='Bus WR1', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(2, 9))
        bus_wr2 = pp.create_bus(net, name='Bus WR2', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(2, 8))
        bus_wr3 = pp.create_bus(net, name='Bus WR3', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(2, 7))
        bus_wr4 = pp.create_bus(net, name='Bus WR4', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(8, 8))
        bus_wr5 = pp.create_bus(net, name='Bus WR5', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(8, 7))
        bus_wr6 = pp.create_bus(net, name='Bus WR6', vn_kv=0.4, type='b', zone='THKoelnNetwork', geodata=(8, 6))

        # Define line parameters
        # Line properties from bus to the inverters
        if self.line1 is None:
            self.line1 = {'c_nf_per_km': 152.63,  # line capacitance (line-to-earth)      nF/km
                         'r_ohm_per_km': 14.7913,  # line resistance                       Ω/km
                         'x_ohm_per_km': 0.014759,  # line reactance                        Ω/km
                         'c0_nf_per_km': 49.01,  # zero line capacitance (line-to-earth)      nF/km
                         'r0_ohm_per_km': 56.5108,  # zero line resistance                       Ω/km
                         'x0_ohm_per_km': 0.039612,  # zero line reactance                        Ω/km
                         'max_i_ka': 0.016,  # maximal thermal current               kA
                         'type': 'cs'  # cs - underground cable system
                         }

        # Creation of the standard type LtE1.5
        pp.create_std_type(net, self.line1, name='LtH1_5', element='line')

        # Line properties between the buses in the net
        if self.line10 is None:
            self.line10 = {'c_nf_per_km': 203.11,  # line capacitance (line-to-earth)      nF/km
                         'r_ohm_per_km': 2.057,  # line resistance                       Ω/km
                         'x_ohm_per_km': 0.017556,  # line reactance                        Ω/km
                         'c0_nf_per_km': 49.43,  # zero line capacitance (line-to-earth)      nF/km
                         'r0_ohm_per_km': 8.80465,  # zero line resistance                       Ω/km
                         'x0_ohm_per_km': 0.049769,  # zero line reactance                      Ω/km
                         'max_i_ka': 0.06,  # maximal thermal current               kA
                         'type': 'cs'  # cs - underground cable system
                         }

        # Creation of the standard type LtH10
        pp.create_std_type(net, self.line10, name='LtE10', element='line')

        # Define line parameters
        # Left string
        pp.create_line(net, name="Line LtE5", from_bus=bus1, to_bus=bus3, length_km=0.056, std_type="LtE10")
        pp.create_line(net, name="Line LtE6", from_bus=bus3, to_bus=bus5, length_km=0.073, std_type="LtE10")
        pp.create_line(net, name="Line LtE3", from_bus=bus5, to_bus=bus7, length_km=0.033, std_type="LtE10")

        pp.create_line(net, name="Line LtH1", from_bus=bus1, to_bus=bus_wr1, length_km=0.010, std_type="LtH1_5")
        pp.create_line(net, name="Line LtH2", from_bus=bus3, to_bus=bus_wr2, length_km=0.012, std_type="LtH1_5")
        pp.create_line(net, name="Line LtH3", from_bus=bus5, to_bus=bus_wr3, length_km=0.014, std_type="LtH1_5")

        # Right String
        pp.create_line(net, name="Line LtE4", from_bus=bus2, to_bus=bus4, length_km=0.043, std_type="LtE10")
        pp.create_line(net, name="Line LtE1", from_bus=bus4, to_bus=bus6, length_km=0.020, std_type="LtE10")
        pp.create_line(net, name="Line LtE2", from_bus=bus6, to_bus=bus8, length_km=0.025, std_type="LtE10")

        pp.create_line(net, name="Line LtH4", from_bus=bus4, to_bus=bus_wr4, length_km=0.017, std_type="LtH1_5")
        pp.create_line(net, name="Line LtH5", from_bus=bus6, to_bus=bus_wr5, length_km=0.021, std_type="LtH1_5")
        pp.create_line(net, name="Line LtH6", from_bus=bus8, to_bus=bus_wr6, length_km=0.025, std_type="LtH1_5")

        # define the switches
        pp.create_switch(net, bus=bus0, element=bus1, name="switchA", et="b", closed=True)
        pp.create_switch(net, bus=bus0, element=bus2, name="switchB", et="b", closed=True)
        pp.create_switch(net, bus=bus7, element=bus8, name="openring", et="b", closed=False)

        # External grid (Slack), default according to IEC 60909-0 Beiblatt 4:2009-08
        # S_sc Dyn5 630kVA 6% u_k, 1.2% u_kr => Z_t = 0.01524Ohm , Re{Z_t} = 3.047mOhm
        # Alternative: ISBN 978-3-8007-3700-0 S. 83 S_k'' for LV: 0,02GVA -> 20MVA
        if slack:
            pp.create_ext_grid(net, bus=bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=20.0,
                               s_sc_min_mva=20.0, rx_max=0.1, rx_min=0.1)
            net['is_asymmetric'] = True

        # Laboratory Generator
        if generator:
            pp.create_ext_grid(net, bus=bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=0.114501334,
                               s_sc_min_mva=0.114501334, rx_max=1.429698229, rx_min=1.429698229)

        # Loads
        pp.create_load(net, bus_wr1, p_mw=0.015, name='Load R1')
        pp.create_load(net, bus_wr2, p_mw=0.015, name='Load R2')
        pp.create_load(net, bus_wr3, p_mw=0.015, name='Load R3')
        pp.create_load(net, bus_wr4, p_mw=0.015, name='Load R4')
        pp.create_load(net, bus_wr5, p_mw=0.015, name='Load R5')
        pp.create_load(net, bus_wr6, p_mw=0.015, name='Load R6')

        # PV
        pp.create_sgen(net, bus=bus_wr1, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 1', type="PV")
        pp.create_sgen(net, bus=bus_wr2, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 2', type="PV")
        pp.create_sgen(net, bus=bus_wr3, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 3', type="PV")
        pp.create_sgen(net, bus=bus_wr4, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 4', type="PV")
        pp.create_sgen(net, bus=bus_wr5, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 5', type="PV")
        pp.create_sgen(net, bus=bus_wr6, p_mw=0.011, q_mvar=0, sn_mva=0.011, name='PV 6', type="PV")

        # Attach frequency dependent grid impedance
        add_grid_imp = self._handle_grid_impedance('microgrid_thk', 2500)
        if add_grid_imp is not None and self.with_grid_impedance:
            net['grid_impedance'] = add_grid_imp
        return net

    def get_load_profiles(self, dataset="train", n_samples=1000):
        if self.samples is None:
            self.samples = generate_inverter_samples(n_samples, dim=6)
        if self.samples_val is None:
            self.samples_val = generate_inverter_samples(n_samples, dim=6)
        load_profiles = {}
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        rng = np.random.default_rng(1)
        random_func = rng.normal
        noise_parameters = {'loc': 0.85, 'scale': 0.05}
        ppc_residential = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                             PowerProfileSubCategory.RESIDENTIAL, step_size_ms=60000,
                                             scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        ppc_commercial = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                            PowerProfileSubCategory.COMMERCIAL, step_size_ms=60000,
                                            scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        beta_func = rng.beta
        beta_noise_parameters = {'a': .5, 'b': .5}
        ppc_random = PowerProfileConfig("Default", PowerProfileCategory.DEFAULT, PowerProfileSubCategory.DEFAULT,
                                        step_size_ms=60000,
                                        scaling_factor=1, normalize=True, random_func=beta_func,
                                        noise_params=beta_noise_parameters)

        ppc_samples = PowerProfileConfig("Samples", PowerProfileCategory.SAMPLED, PowerProfileSubCategory.DEFAULT,
                                         step_size_ms=60000, scaling_factor=1, normalize=False)
        if dataset == "train":
            load_profiles[load_names[0]] = ppc_samples.clone(samples=self.samples[0][0])
            load_profiles[load_names[1]] = ppc_samples.clone(samples=self.samples[0][1])
            load_profiles[load_names[2]] = ppc_samples.clone(samples=self.samples[0][2])
            load_profiles[load_names[3]] = ppc_samples.clone(samples=self.samples[0][3])
            load_profiles[load_names[4]] = ppc_samples.clone(samples=self.samples[0][4])
            load_profiles[load_names[5]] = ppc_samples.clone(samples=self.samples[0][5])
        elif dataset == 'test':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_residential.clone(name="LP07")
            load_profiles[load_names[1]] = ppc_residential.clone(name="LP08")
            load_profiles[load_names[2]] = ppc_residential.clone(name="LP10")
            load_profiles[load_names[3]] = ppc_residential.clone(name="LP11")
            load_profiles[load_names[4]] = ppc_residential.clone(name="LP13")
            load_profiles[load_names[5]] = ppc_residential.clone(name="LP15")
        elif dataset == 'validation':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_samples.clone(samples=self.samples_val[0][0])
            load_profiles[load_names[1]] = ppc_samples.clone(samples=self.samples_val[0][1])
            load_profiles[load_names[2]] = ppc_samples.clone(samples=self.samples_val[0][2])
            load_profiles[load_names[3]] = ppc_samples.clone(samples=self.samples_val[0][3])
            load_profiles[load_names[4]] = ppc_samples.clone(samples=self.samples_val[0][4])
            load_profiles[load_names[5]] = ppc_samples.clone(samples=self.samples_val[0][5])
        else:
            raise ValueError('Mode must be either train, test or validation')
        return load_profiles

    def get_generator_profiles(self, dataset='train', n_samples=1000):
        if self.samples is None:
            self.samples = generate_inverter_samples(n_samples, dim=6)
        if self.samples_val is None:
            self.samples_val = generate_inverter_samples(n_samples, dim=6)
        sgen_names = self.pp_net.sgen['name'].values
        generator_profiles = {}
        rng = np.random.default_rng(2)
        random_func = rng.normal
        noise_parameters = {'loc': 0.85, 'scale': 0.05}
        ppc_pv = PowerProfileConfig(None, PowerProfileCategory.GENERATOR,
                                    PowerProfileSubCategory.PHOTOVOLTAIC_SYSTEM, step_size_ms=60000,
                                    scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        beta_func = rng.beta
        beta_noise_parameters = {'a': .5, 'b': .5}
        ppc_random = PowerProfileConfig("Default", PowerProfileCategory.DEFAULT, PowerProfileSubCategory.DEFAULT,
                                        step_size_ms=60000,
                                        scaling_factor=1, normalize=True, random_func=beta_func,
                                        noise_params=beta_noise_parameters)
        ppc_samples = PowerProfileConfig("Samples", PowerProfileCategory.SAMPLED, PowerProfileSubCategory.DEFAULT,
                                         step_size_ms=60000, scaling_factor=1, normalize=False)
        if dataset == 'train':
            generator_profiles[sgen_names[0]] = ppc_samples.clone(samples=self.samples[1][0])
            generator_profiles[sgen_names[1]] = ppc_samples.clone(samples=self.samples[1][1])
            generator_profiles[sgen_names[2]] = ppc_samples.clone(samples=self.samples[1][2])
            generator_profiles[sgen_names[3]] = ppc_samples.clone(samples=self.samples[1][3])
            generator_profiles[sgen_names[4]] = ppc_samples.clone(samples=self.samples[1][4])
            generator_profiles[sgen_names[5]] = ppc_samples.clone(samples=self.samples[1][5])
        elif dataset == 'test':
            generator_profiles[sgen_names[0]] = ppc_pv.clone(name="SMA_PV_2013")
            generator_profiles[sgen_names[1]] = ppc_pv.clone(name="SMA_PV_2014")
            generator_profiles[sgen_names[2]] = ppc_pv.clone(name="SMA_PV_2015")
            generator_profiles[sgen_names[3]] = ppc_pv.clone(name="PV_sol_EFH - NW")
            generator_profiles[sgen_names[4]] = ppc_pv.clone(name="PV_sol_EFH - SW")
            generator_profiles[sgen_names[5]] = ppc_pv.clone(name="PV_sol_RH - NW")
        elif dataset == 'validation':
            generator_profiles[sgen_names[0]] = ppc_samples.clone(samples=self.samples_val[1][0])
            generator_profiles[sgen_names[1]] = ppc_samples.clone(samples=self.samples_val[1][1])
            generator_profiles[sgen_names[2]] = ppc_samples.clone(samples=self.samples_val[1][2])
            generator_profiles[sgen_names[3]] = ppc_samples.clone(samples=self.samples_val[1][3])
            generator_profiles[sgen_names[4]] = ppc_samples.clone(samples=self.samples_val[1][4])
            generator_profiles[sgen_names[5]] = ppc_samples.clone(samples=self.samples_val[1][5])
        else:
            raise ValueError('Mode must be either train, test or validation')

        # default profile is defined as always 1, so we need to add rng to it centered around 1 with a small standard deviation
        noise_parameters = {'loc': 1, 'scale': 0.03}
        vsource_profile = PowerProfileConfig('Default', PowerProfileCategory.DEFAULT,
                                             PowerProfileSubCategory.DEFAULT, step_size_ms=60000,
                                             scaling_factor=1, normalize=False, random_func=random_func,
                                             noise_params=noise_parameters)
        generator_profiles["source"] = vsource_profile # opendss will always remane the L1 to "source"
        generator_profiles["source2"] = vsource_profile
        generator_profiles["source3"] = vsource_profile
        return generator_profiles

    def get_load_spectra(self, dataset='train'):
        load_spectra = {}
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        spectrum_config = SpectrumConfig("MicrogridlabInverter", None, "GENERATION", "PHOTOVOLTAIC_SYSTEM", phase=1)
        rng = np.random.default_rng(3)
        random_func = rng.random
        noise_params = {}
        random_config = SpectrumConfig(None, None, "DEFAULT", "DEFAULT", dynamic_type=DynamicSpectrumType.RANDOM,
                                       random_func=random_func, noise_params=noise_params,
                                       max_harmonic_values=cfg.MAX_HARMONIC_VALUES)
        if dataset == 'train':
            load_spectra[load_names[0]] = random_config
            load_spectra[load_names[1]] = random_config
            load_spectra[load_names[2]] = random_config
            load_spectra[load_names[3]] = random_config
            load_spectra[load_names[4]] = random_config
            load_spectra[load_names[5]] = random_config
        else:
            load_spectra[load_names[0]] = spectrum_config.clone(name="MicrogridlabInverter_rack1")
            load_spectra[load_names[1]] = spectrum_config.clone(name="MicrogridlabInverter_rack2")
            load_spectra[load_names[2]] = spectrum_config.clone(name="MicrogridlabInverter_rack3")
            load_spectra[load_names[3]] = spectrum_config.clone(name="MicrogridlabInverter_rack4")
            load_spectra[load_names[4]] = spectrum_config.clone(name="MicrogridlabInverter_rack5")
            load_spectra[load_names[5]] = spectrum_config.clone(name="MicrogridlabInverter_rack6")
        return load_spectra

    def get_generator_spectra(self, dataset='train'):
        sgen_names = self.pp_net.sgen['name'].values
        rng = np.random.default_rng(4)
        random_func = rng.random
        noise_params = {}
        random_config = SpectrumConfig(None, None, "DEFAULT", "DEFAULT", dynamic_type=DynamicSpectrumType.RANDOM,
                                       random_func=random_func, noise_params=noise_params,
                                       max_harmonic_values=cfg.MAX_HARMONIC_VALUES)

        random_vsource_config_1 = SpectrumConfig(None, None, "DEFAULT", "DEFAULT",
                                                 dynamic_type=DynamicSpectrumType.RANDOM, phase=1,
                                                 random_func=random_func, noise_params=noise_params,
                                                 max_harmonic_values=cfg.MAX_HARMONIC_VALUES,
                                                 ignore_angles=self.ignore_source_angles)
        random_vsource_config_2 = SpectrumConfig(None, None, "DEFAULT", "DEFAULT",
                                                 dynamic_type=DynamicSpectrumType.RANDOM, phase=2,
                                                 random_func=random_func, noise_params=noise_params,
                                                 max_harmonic_values=cfg.MAX_HARMONIC_VALUES,
                                                 ignore_angles=self.ignore_source_angles)
        random_vsource_config_3 = SpectrumConfig(None, None, "DEFAULT", "DEFAULT",
                                                 dynamic_type=DynamicSpectrumType.RANDOM, phase=3,
                                                 random_func=random_func, noise_params=noise_params,
                                                 max_harmonic_values=cfg.MAX_HARMONIC_VALUES,
                                                 ignore_angles=self.ignore_source_angles)

        generator_spectra = {}
        if dataset == 'train':
            generator_spectra[sgen_names[0]] = random_config
            generator_spectra[sgen_names[1]] = random_config
            generator_spectra[sgen_names[2]] = random_config
            generator_spectra[sgen_names[3]] = random_config
            generator_spectra[sgen_names[4]] = random_config
            generator_spectra[sgen_names[5]] = random_config
            generator_spectra["source"] = random_vsource_config_1  # opendss will always remane the L1 to "source"
            generator_spectra["source2"] = random_vsource_config_2
            generator_spectra["source3"] = random_vsource_config_3
        elif dataset == "test" or dataset == "validation":
            spectrum_config = SpectrumConfig("MicrogridlabInverter", None, "GENERATION", "PHOTOVOLTAIC_SYSTEM", phase=1)
            generator_spectra[sgen_names[0]] = spectrum_config.clone(name="MicrogridlabInverter_rack1")
            generator_spectra[sgen_names[1]] = spectrum_config.clone(name="MicrogridlabInverter_rack2")
            generator_spectra[sgen_names[2]] = spectrum_config.clone(name="MicrogridlabInverter_rack3")
            generator_spectra[sgen_names[3]] = spectrum_config.clone(name="MicrogridlabInverter_rack4")
            generator_spectra[sgen_names[4]] = spectrum_config.clone(name="MicrogridlabInverter_rack5")
            generator_spectra[sgen_names[5]] = spectrum_config.clone(name="MicrogridlabInverter_rack6")
            v_source_spectrum_config = SpectrumConfig("MicrogridlabSupplySource", manufacturer=None, category="GENERATION",
                                             subcategory="GRID", measurement_type=MeasurementType.VOLTAGE,
                                             dynamic_type=DynamicSpectrumType.TIME_SERIES)
            generator_spectra["source"] = v_source_spectrum_config  # opendss will always remane the L1 to "source"
            generator_spectra["source2"] = v_source_spectrum_config
            generator_spectra["source3"] = v_source_spectrum_config

        return generator_spectra


def generate_inverter_samples(num_samples, dim=6, scramble=True):
    """
    Generate load and generator samples that together mimic the behavior of inverters.
    :param num_samples: amount of samples to draw
    :param dim: amount of inverters
    :param scramble:
    :return: tuple of load_samples and generator_samples with dimensions (dim, num_samples)
    """
    from scipy.stats import qmc
    # Initialize Sobol sampler for 6 dimensions.
    sampler = qmc.Sobol(d=dim, scramble=scramble)
    # Generate num_samples points in the unit cube [0, 1]^6.
    unit_samples = sampler.random(num_samples)
    # Scale samples to the range [-1, 1] for each inverter.
    # Transpose the matrix so that n_dim is the first dimension.
    scaled_samples = qmc.scale(unit_samples, l_bounds=[-1] * dim, u_bounds=[1] * dim).T

    # Create load samples: positive values are kept; negative values are set to zero.
    load_samples = np.where(scaled_samples > 0, scaled_samples, 0)

    # Create generator samples: negative values are converted to positive (sign change);
    # positive values are set to zero.
    generator_samples = np.where(scaled_samples < 0, -scaled_samples, 0)

    return load_samples, generator_samples


if __name__ == '__main__':
    microgrid_th = MicrogridTHKoeln(None, None, False, False)
    pp_net = microgrid_th.pp_net
    microgrid_th.get_cable_resonances(pp_net)
    load_names = pp_net.load['name'].values
    sgen_names = pp_net.sgen['name'].values
    pass
