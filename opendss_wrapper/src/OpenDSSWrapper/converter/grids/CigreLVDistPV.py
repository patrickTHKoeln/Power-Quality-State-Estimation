import os.path

import numpy as np
import pandapower as pp

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.harmonic_spectra import SpectrumConfig
from OpenDSSWrapper.power_profiles import PowerProfileConfig, PowerProfileCategory, PowerProfileSubCategory


class CigreLVDistPV(BaseGrid):
    def __init__(self):
        super().__init__()
        self.n_nodes = 44
        self.name = "CIGRE Low Voltage Distribution Grid"
        self.short_name = "cigrelv"
        self.descriptive_nodes = {'0': '0',
                                  '1': 'R0',
                                  '2': 'R1',
                                  '3': 'R2',
                                  '4': 'R3',
                                  '5': 'R4',
                                  '6': 'R5',
                                  '7': 'R6',
                                  '8': 'R7',
                                  '9': 'R8, PV1',
                                  '10': 'R9',
                                  '11': 'R10',
                                  '12': 'R11',
                                  '13': 'R12',
                                  '14': 'R13',
                                  '15': 'R14',
                                  '16': 'R15',
                                  '17': 'R16',
                                  '18': 'R17',
                                  '19': 'R18',
                                  '20': 'I0',
                                  '21': 'I1',
                                  '22': 'I2',
                                  '23': 'C0',
                                  '24': 'C1',
                                  '25': 'C2',
                                  '26': 'C3',
                                  '27': 'C4',
                                  '28': 'C5',
                                  '29': 'C6',
                                  '30': 'C7',
                                  '31': 'C8',
                                  '32': 'C9, PV2',
                                  '33': 'C10',
                                  '34': 'C11',
                                  '35': 'C12, PV3',
                                  '36': 'C13',
                                  '37': 'C14',
                                  '38': 'C15',
                                  '39': 'C16',
                                  '40': 'C17',
                                  '41': 'C18',
                                  '42': 'C19',
                                  '43': 'C20'
                                  }

    def _get_pandapower(self):
        net = pp.networks.create_cigre_network_lv()
        # Add PV manually
        # NOTE: if type is not set here, it is either set to wye or delta for connection type automatically
        pp.create_sgen(net, bus=9, p_mw=0.01, q_mvar=0, sn_mva=0.01, name='PV 1', type="PV")
        pp.create_sgen(net, bus=32, p_mw=0.035, q_mvar=0, sn_mva=0.035, name='PV 2', type="PV")
        pp.create_sgen(net, bus=35, p_mw=0.015, q_mvar=0, sn_mva=0.015, name='PV 3', type="PV")

        # update line types for PQ-Simulations by capacities
        line_data = {'c_nf_per_km': 1100.0, 'r_ohm_per_km': 0.162,
                     'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                     'type': 'cs'}
        pp.delete_std_type(net, name='UG1', element='line')
        pp.create_std_type(net, line_data, name='UG1', element='line')
        # UG2
        line_data = {'c_nf_per_km': 1100.0, 'r_ohm_per_km': 0.2647,
                     'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                     'type': 'cs'}
        pp.delete_std_type(net, name='UG2', element='line')
        pp.create_std_type(net, line_data, name='UG2', element='line')

        # UG3
        line_data = {'c_nf_per_km': 1100.0, 'r_ohm_per_km': 0.822,
                     'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                     'type': 'cs'}
        pp.delete_std_type(net, name='UG3', element='line')
        pp.create_std_type(net, line_data, name='UG3', element='line')

        # OH1
        line_data = {'c_nf_per_km': 210, 'r_ohm_per_km': 0.4917,
                     'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                     'type': 'ol'}
        pp.delete_std_type(net, name='OH1', element='line')
        pp.create_std_type(net, line_data, name='OH1', element='line')

        # OH2
        line_data = {'c_nf_per_km': 210, 'r_ohm_per_km': 1.3207,
                     'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                     'type': 'ol'}
        pp.delete_std_type(net, name='OH2', element='line')
        pp.create_std_type(net, line_data, name='OH2', element='line')

        # OH3
        line_data = {'c_nf_per_km': 210, 'r_ohm_per_km': 2.0167,
                     'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                     'type': 'ol'}
        pp.delete_std_type(net, name='OH3', element='line')
        pp.create_std_type(net, line_data, name='OH3', element='line')

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

        # set all tap pos to 1
        for i in net.trafo.index:
            net.trafo.at[i, 'tap_pos'] = 1
        return net

    def get_load_profiles(self, dataset="train", n_samples=1000):
        load_profiles = {}
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        rng = np.random.default_rng(1)
        random_func = rng.normal
        noise_parameters = {'loc': 0.85, 'scale': 0.05}
        # Test and validation sets are realistic profiles
        ppc_residential = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                             PowerProfileSubCategory.RESIDENTIAL, step_size_ms=60000,
                                             scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        ppc_commercial = PowerProfileConfig(None, PowerProfileCategory.LOAD,
                                            PowerProfileSubCategory.COMMERCIAL, step_size_ms=60000,
                                            scaling_factor=1, random_func=random_func, noise_params=noise_parameters)
        # Training set is fully randomized (per profile)
        uniform_func = rng.uniform
        uniform_noise_parameters = {'low': 0, 'high': 1}

        # Original issue: if every load profile is randomized, on average the most common case will be the average case
        # this results in a normal distribution around 50% load. To avoid this, we skew the distribution towards the
        # extremes. This is done by using a beta distribution with a=0.5, b=0.5. This distribution is symmetric around
        # 0.5, but has a higher probability for values close to 0 and 1. This way, we can achieve a more even distribution.
        # Each individual profile is skewed towards extreme values so that overall we achieve an even distribution
        beta_func = rng.beta
        beta_noise_parameters = {'a': .5, 'b': .5}
        ppc_random = PowerProfileConfig("Default", PowerProfileCategory.DEFAULT, PowerProfileSubCategory.DEFAULT,
                                        step_size_ms=60000,
                                        scaling_factor=1, normalize=True, random_func=beta_func,
                                        noise_params=beta_noise_parameters)
        if dataset == "train":
            load_profiles[load_names[0]] = ppc_random
            load_profiles[load_names[1]] = ppc_random
            load_profiles[load_names[2]] = ppc_random
            load_profiles[load_names[3]] = ppc_random
            load_profiles[load_names[4]] = ppc_random
            load_profiles[load_names[5]] = ppc_random
            load_profiles[load_names[6]] = ppc_random
            load_profiles[load_names[7]] = ppc_random
            load_profiles[load_names[8]] = ppc_random
            load_profiles[load_names[9]] = ppc_random
            load_profiles[load_names[10]] = ppc_random
            load_profiles[load_names[11]] = ppc_random
            load_profiles[load_names[12]] = ppc_random
            load_profiles[load_names[13]] = ppc_random
            load_profiles[load_names[14]] = ppc_random

        elif dataset == 'test':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_residential.clone(name="HTW_Berlin_household_1")
            load_profiles[load_names[1]] = ppc_residential.clone(name="HTW_Berlin_household_2")
            load_profiles[load_names[2]] = ppc_residential.clone(name="HTW_Berlin_household_3")
            load_profiles[load_names[3]] = ppc_residential.clone(name="HTW_Berlin_household_4")
            load_profiles[load_names[4]] = ppc_residential.clone(name="HTW_Berlin_household_5")
            load_profiles[load_names[5]] = ppc_residential.clone(name="HTW_Berlin_household_6")
            load_profiles[load_names[6]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_10")
            load_profiles[load_names[7]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_11")
            load_profiles[load_names[8]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_12")
            load_profiles[load_names[9]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_13")
            load_profiles[load_names[10]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_14")
            load_profiles[load_names[11]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_15")
            load_profiles[load_names[12]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_16")
            load_profiles[load_names[13]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_17")
            load_profiles[load_names[14]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_18")

        elif dataset == 'validation':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_residential.clone(name="HTW_Berlin_household_7")
            load_profiles[load_names[1]] = ppc_residential.clone(name="HTW_Berlin_household_8")
            load_profiles[load_names[2]] = ppc_residential.clone(name="HTW_Berlin_household_9")
            load_profiles[load_names[3]] = ppc_residential.clone(name="HTW_Berlin_household_10")
            load_profiles[load_names[4]] = ppc_residential.clone(name="HTW_Berlin_household_11")
            load_profiles[load_names[5]] = ppc_residential.clone(name="HTW_Berlin_household_12")
            load_profiles[load_names[6]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_19")
            load_profiles[load_names[7]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_20")
            load_profiles[load_names[8]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_21")
            load_profiles[load_names[9]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_22")
            load_profiles[load_names[10]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_23")
            load_profiles[load_names[11]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_24")
            load_profiles[load_names[12]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_25")
            load_profiles[load_names[13]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_26")
            load_profiles[load_names[14]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_27")
        else:
            raise ValueError('Mode must be either train, test or validation')
        return load_profiles

    def get_generator_profiles(self, dataset='train', n_samples=1000):
        sgen_names = self.pp_net.sgen['name'].values
        generator_profiles = {}
        rng = np.random.default_rng(4)
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
        if dataset == 'train':
            generator_profiles[sgen_names[0]] = ppc_random
            generator_profiles[sgen_names[1]] = ppc_random
            generator_profiles[sgen_names[2]] = ppc_random
        elif dataset == 'test':
            generator_profiles[sgen_names[0]] = ppc_pv.clone(name="SMA_PV_2013")
            generator_profiles[sgen_names[1]] = ppc_pv.clone(name="SMA_PV_2014")
            generator_profiles[sgen_names[2]] = ppc_pv.clone(name="SMA_PV_2015")
        elif dataset == 'validation':
            generator_profiles[sgen_names[0]] = ppc_pv.clone(name="PV_sol_EFH - SW")
            generator_profiles[sgen_names[1]] = ppc_pv.clone(name="PV_sol_EFH - NW")
            generator_profiles[sgen_names[2]] = ppc_pv.clone(name="PV_sol_RH - NW")
        else:
            raise ValueError('Mode must be either train, test or validation')
        return generator_profiles

    def get_load_spectra(self, dataset='train'):
        load_spectra = {}
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        spectrum_config = SpectrumConfig(None, None, "ELECTRIC_VEHICLES", "ELECTRIC_CAR")
        default_config = SpectrumConfig()
        if dataset == 'train':
            load_spectra[load_names[0]] = default_config  # R1
            load_spectra[load_names[1]] = default_config  # R11
            load_spectra[load_names[2]] = spectrum_config.clone(name="2015 Mercedes B-Class", manufacturer="Mercedes", nominal_voltage=240)  # R15
            load_spectra[load_names[3]] = spectrum_config.clone(name="2015 Nissan Leaf", manufacturer="Nissan", nominal_voltage=208)  # R16
            load_spectra[load_names[4]] = default_config  # R17
            load_spectra[load_names[5]] = default_config  # R18
            load_spectra[load_names[6]] = default_config  # I2
            load_spectra[load_names[7]] = default_config  # C1
            load_spectra[load_names[8]] = default_config  # C12
            load_spectra[load_names[9]] = default_config  # C13
            load_spectra[load_names[10]] = spectrum_config.clone(name="2015 Mercedes B-Class", manufacturer="Mercedes", nominal_voltage=240)  # C14
            load_spectra[load_names[11]] = default_config  # C17
            load_spectra[load_names[12]] = spectrum_config.clone(name="Tesla S85", manufacturer="Tesla", nominal_voltage=230)  # C18
            load_spectra[load_names[13]] = spectrum_config.clone(name="2015 Nissan Leaf", manufacturer="Nissan", nominal_voltage=208)  # C19
            load_spectra[load_names[14]] = default_config  # C20
        else:
            load_spectra[load_names[0]] = default_config  # R1
            load_spectra[load_names[1]] = default_config  # R11
            load_spectra[load_names[2]] = spectrum_config.clone(name="2012 Nissan Leaf", manufacturer="Nissan", nominal_voltage=208)  # R15
            load_spectra[load_names[3]] = spectrum_config.clone(name="2014 BMW i3", manufacturer="BMW", nominal_voltage=240)  # R16
            load_spectra[load_names[4]] = default_config  # R17
            load_spectra[load_names[5]] = default_config  # R18
            load_spectra[load_names[6]] = default_config  # I2
            load_spectra[load_names[7]] = default_config  # C1
            load_spectra[load_names[8]] = default_config  # C12
            load_spectra[load_names[9]] = default_config  # C13
            load_spectra[load_names[10]] = spectrum_config.clone(name="2012 Nissan Leaf", manufacturer="Nissan", nominal_voltage=208)  # C14
            load_spectra[load_names[11]] = default_config  # C17
            load_spectra[load_names[12]] = spectrum_config.clone(name="2012 Chevrolet Volt", manufacturer="Chevrolet", nominal_voltage=208)  # C18
            load_spectra[load_names[13]] = spectrum_config.clone(name="2014 BMW i3", manufacturer="BMW", nominal_voltage=240)  # C19
            load_spectra[load_names[14]] = default_config  # C20
        return load_spectra

    def get_generator_spectra(self, dataset='train'):
        sgen_names = self.pp_net.sgen['name'].values
        generator_spectra = {}
        spectrum_config = SpectrumConfig(None, None, "GENERATION", "PHOTOVOLTAIC_SYSTEM")
        if dataset == 'train' or dataset == "test" or dataset == "validation":
            generator_spectra[sgen_names[0]] = spectrum_config.clone(name="PVInverter33kW")
            generator_spectra[sgen_names[1]] = spectrum_config.clone(name="PVInverter33kW")
            generator_spectra[sgen_names[2]] = spectrum_config.clone(name="PVInverter33kW")
        return generator_spectra


if __name__ == '__main__':
    cigre_lv = CigreLVDistPV()
    pp_net = cigre_lv.pp_net
    from OpenDSSWrapper.converter import pandapower_to_opendss
    cigre_lv.get_cable_resonances(pp_net)
    pandapower_to_opendss(pp_net, 'cigreLVTest', os.path.join('C:', 'Temp'))
    load_names = pp_net.load['name'].values
    sgen_names = pp_net.sgen['name'].values
    pass
