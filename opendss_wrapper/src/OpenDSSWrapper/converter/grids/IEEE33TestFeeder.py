import os

import numpy as np
import pandapower as pp

from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.harmonic_spectra import SpectrumConfig, DynamicSpectrumType
from OpenDSSWrapper.power_profiles import PowerProfileConfig, PowerProfileCategory, PowerProfileSubCategory
import OpenDSSWrapper.tools.config as cfg

class IEEE33TestFeeder(BaseGrid):
    def __init__(self):
        super().__init__()
        self.n_nodes = 33
        self.name = "IEEE 33 BUS DISTRIBUTION TEST SYSTEM"
        self.short_name = "IEEE33"
        self.descriptive_nodes = {'0': '1',
                                  '1': '2',
                                  '2': '3',
                                  '3': '4',
                                  '4': '5',
                                  '5': '6',
                                  '6': '7',
                                  '7': '8',
                                  '8': '9',
                                  '9': '10',
                                  '10': '11',
                                  '11': '12',
                                  '12': '13',
                                  '13': '14 (EVSE)',
                                  '14': '15',
                                  '15': '16',
                                  '16': '17',
                                  '17': '18 (EVSE)',
                                  '18': '19',
                                  '19': '20',
                                  '20': '21 (EVSE)',
                                  '21': '22',
                                  '22': '23',
                                  '23': '24 (EVSE)',
                                  '24': '25',
                                  '25': '26',
                                  '26': '27',
                                  '27': '28',
                                  '28': '29 (EVSE)',
                                  '29': '30',
                                  '30': '31',
                                  '31': '32',
                                  '32': '33'
                                  }

    def _get_pandapower(self):
        net = pp.networks.case33bw()
        # names and types are all None, set to some default value
        for idx, ld in net.load.iterrows():
            if ld['name'] is None:
                net.load.loc[idx, 'name'] = "Load " + str(ld['bus'])
            if ld['type'] is None:
                net.load.loc[idx, 'type'] = 'wye'
        for idx, li in net.line.iterrows():
            if li['name'] is None:
                net.line.loc[idx, 'name'] = f"{li['from_bus']}-{li['to_bus']}"
        from OpenDSSWrapper.converter.convert_cgmes import line_parameters_to_std_type
        net = line_parameters_to_std_type(net)
        return net

    def get_load_profiles(self, dataset="train"):
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
        beta_noise_parameters = {'a': .1, 'b': .5}
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
            load_profiles[load_names[15]] = ppc_random
            load_profiles[load_names[16]] = ppc_random
            load_profiles[load_names[17]] = ppc_random
            load_profiles[load_names[18]] = ppc_random
            load_profiles[load_names[19]] = ppc_random
            load_profiles[load_names[20]] = ppc_random
            load_profiles[load_names[21]] = ppc_random
            load_profiles[load_names[22]] = ppc_random
            load_profiles[load_names[23]] = ppc_random
            load_profiles[load_names[24]] = ppc_random
            load_profiles[load_names[25]] = ppc_random
            load_profiles[load_names[26]] = ppc_random
            load_profiles[load_names[27]] = ppc_random
            load_profiles[load_names[28]] = ppc_random
            load_profiles[load_names[29]] = ppc_random
            load_profiles[load_names[30]] = ppc_random
            load_profiles[load_names[31]] = ppc_random
        elif dataset == 'test':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_residential.clone(name="HTW_Berlin_household_1")
            load_profiles[load_names[1]] = ppc_residential.clone(name="HTW_Berlin_household_2")
            load_profiles[load_names[2]] = ppc_residential.clone(name="HTW_Berlin_household_3")
            load_profiles[load_names[3]] = ppc_residential.clone(name="HTW_Berlin_household_4")
            load_profiles[load_names[4]] = ppc_residential.clone(name="HTW_Berlin_household_5")
            load_profiles[load_names[5]] = ppc_residential.clone(name="HTW_Berlin_household_6")
            load_profiles[load_names[6]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_1")
            load_profiles[load_names[7]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_1")
            load_profiles[load_names[8]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_2")
            load_profiles[load_names[9]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_3")
            load_profiles[load_names[10]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_4")
            load_profiles[load_names[11]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_5")
            load_profiles[load_names[12]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_6")
            load_profiles[load_names[13]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_7")
            load_profiles[load_names[14]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_8")
            load_profiles[load_names[15]] = ppc_residential.clone(name="HTW_Berlin_household_7")
            load_profiles[load_names[16]] = ppc_residential.clone(name="HTW_Berlin_household_8")
            load_profiles[load_names[17]] = ppc_residential.clone(name="HTW_Berlin_household_9")
            load_profiles[load_names[18]] = ppc_residential.clone(name="HTW_Berlin_household_10")
            load_profiles[load_names[19]] = ppc_residential.clone(name="HTW_Berlin_household_11")
            load_profiles[load_names[20]] = ppc_residential.clone(name="HTW_Berlin_household_12")
            load_profiles[load_names[21]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_9")
            load_profiles[load_names[22]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_10")
            load_profiles[load_names[23]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_11")
            load_profiles[load_names[24]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_12")
            load_profiles[load_names[25]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_13")
            load_profiles[load_names[26]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_14")
            load_profiles[load_names[27]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_15")
            load_profiles[load_names[28]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_16")
            load_profiles[load_names[29]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_17")
            load_profiles[load_names[30]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_18")
            load_profiles[load_names[31]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_19")
        elif dataset == 'validation':
            ppc_residential.rng = np.random.default_rng(2)
            ppc_commercial.rng = np.random.default_rng(3)
            load_profiles[load_names[0]] = ppc_residential.clone(name="HTW_Berlin_household_13")
            load_profiles[load_names[1]] = ppc_residential.clone(name="HTW_Berlin_household_14")
            load_profiles[load_names[2]] = ppc_residential.clone(name="HTW_Berlin_household_15")
            load_profiles[load_names[3]] = ppc_residential.clone(name="HTW_Berlin_household_16")
            load_profiles[load_names[4]] = ppc_residential.clone(name="HTW_Berlin_household_17")
            load_profiles[load_names[5]] = ppc_residential.clone(name="HTW_Berlin_household_18")
            load_profiles[load_names[6]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_20")
            load_profiles[load_names[7]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_21")
            load_profiles[load_names[8]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_22")
            load_profiles[load_names[9]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_23")
            load_profiles[load_names[10]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_24")
            load_profiles[load_names[11]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_25")
            load_profiles[load_names[12]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_26")
            load_profiles[load_names[13]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_27")
            load_profiles[load_names[14]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_28")
            load_profiles[load_names[15]] = ppc_residential.clone(name="HTW_Berlin_household_20")
            load_profiles[load_names[16]] = ppc_residential.clone(name="HTW_Berlin_household_21")
            load_profiles[load_names[17]] = ppc_residential.clone(name="HTW_Berlin_household_22")
            load_profiles[load_names[18]] = ppc_residential.clone(name="HTW_Berlin_household_23")
            load_profiles[load_names[19]] = ppc_residential.clone(name="HTW_Berlin_household_24")
            load_profiles[load_names[20]] = ppc_residential.clone(name="HTW_Berlin_household_25")
            load_profiles[load_names[21]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_26")
            load_profiles[load_names[22]] = ppc_commercial.clone(name="LoadProfile_20IPs_2016_27")
            load_profiles[load_names[23]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_28")
            load_profiles[load_names[24]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_29")
            load_profiles[load_names[25]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_30")
            load_profiles[load_names[26]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_31")
            load_profiles[load_names[27]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_32")
            load_profiles[load_names[28]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_33")
            load_profiles[load_names[29]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_34")
            load_profiles[load_names[30]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_35")
            load_profiles[load_names[31]] = ppc_commercial.clone(name="LoadProfile_20IPs_2017_36")
        else:
            raise ValueError('Mode must be either train, test or validation')
        return load_profiles

    def get_generator_profiles(self, dataset='train'):
        sgen_names = self.pp_net.sgen['name'].values
        generator_profiles = {}
        return generator_profiles

    def get_load_spectra(self, dataset='train'):
        load_spectra = {}
        rng = np.random.default_rng(1)
        random_func = rng.random
        noise_params = {}
        random_config = SpectrumConfig(None, None, "DEFAULT", "DEFAULT", dynamic_type=DynamicSpectrumType.RANDOM,
                                       random_func=random_func, noise_params=noise_params, max_harmonic_values=cfg.MAX_HARMONIC_VALUES)
        # Get load names from the net.load DataFrame
        load_names = self.pp_net.load['name'].values
        spectrum_config = SpectrumConfig(None, None, "ELECTRIC_VEHICLES", "ELECTRIC_CAR")
        default_config = SpectrumConfig()
        if dataset == 'train':
            load_spectra[load_names[0]] = default_config  # 1
            load_spectra[load_names[1]] = default_config  # 2
            load_spectra[load_names[2]] = default_config  # 3
            load_spectra[load_names[3]] = default_config  # 4
            load_spectra[load_names[4]] = default_config  # 5
            load_spectra[load_names[5]] = default_config  # 6
            load_spectra[load_names[6]] = default_config  # 7
            load_spectra[load_names[7]] = default_config  # 8
            load_spectra[load_names[8]] = default_config  # 9
            load_spectra[load_names[9]] = default_config  # 10
            load_spectra[load_names[10]] = default_config  # 11
            load_spectra[load_names[11]] = default_config  # 12
            load_spectra[load_names[12]] = default_config  # 13
            load_spectra[load_names[13]] = random_config
            load_spectra[load_names[14]] = default_config  # 15
            load_spectra[load_names[15]] = default_config  # 16
            load_spectra[load_names[16]] = default_config  # 17
            load_spectra[load_names[17]] = random_config
            load_spectra[load_names[18]] = default_config  # 19
            load_spectra[load_names[19]] = default_config  # 20
            load_spectra[load_names[20]] = random_config
            load_spectra[load_names[21]] = default_config  # 22
            load_spectra[load_names[22]] = default_config  # 23
            load_spectra[load_names[23]] = random_config
            load_spectra[load_names[24]] = default_config  # 25
            load_spectra[load_names[25]] = default_config  # 26
            load_spectra[load_names[26]] = default_config  # 27
            load_spectra[load_names[27]] = default_config  # 28
            load_spectra[load_names[28]] = random_config
            load_spectra[load_names[29]] = default_config  # 30
            load_spectra[load_names[30]] = default_config  # 31
            load_spectra[load_names[31]] = default_config  # 32
        else:
            load_spectra[load_names[0]] = default_config  # 1
            load_spectra[load_names[1]] = default_config  # 2
            load_spectra[load_names[2]] = default_config  # 3
            load_spectra[load_names[3]] = default_config  # 4
            load_spectra[load_names[4]] = default_config  # 5
            load_spectra[load_names[5]] = default_config  # 6
            load_spectra[load_names[6]] = default_config  # 7
            load_spectra[load_names[7]] = default_config  # 8
            load_spectra[load_names[8]] = default_config  # 9
            load_spectra[load_names[9]] = default_config  # 10
            load_spectra[load_names[10]] = default_config  # 11
            load_spectra[load_names[11]] = default_config  # 12
            load_spectra[load_names[12]] = default_config  # 13
            load_spectra[load_names[13]] = spectrum_config.clone(name="2012 Nissan Leaf", manufacturer="Nissan",
                                                                 nominal_voltage=208)  # 14
            load_spectra[load_names[14]] = default_config  # 15
            load_spectra[load_names[15]] = default_config  # 16
            load_spectra[load_names[16]] = default_config  # 17
            load_spectra[load_names[17]] = spectrum_config.clone(name="2015 Mercedes B-Class", manufacturer="Mercedes",
                                                                 nominal_voltage=240)  # 18
            load_spectra[load_names[18]] = default_config  # 19
            load_spectra[load_names[19]] = default_config  # 20
            load_spectra[load_names[20]] = spectrum_config.clone(name="2012 Chevrolet Volt", manufacturer="Chevrolet",
                                                                 nominal_voltage=208)  # 21
            load_spectra[load_names[21]] = default_config  # 22
            load_spectra[load_names[22]] = default_config  # 23
            load_spectra[load_names[23]] = spectrum_config.clone(name="2014 BMW i3", manufacturer="BMW",
                                                                 nominal_voltage=208)  # 24
            load_spectra[load_names[24]] = default_config  # 25
            load_spectra[load_names[25]] = default_config  # 26
            load_spectra[load_names[26]] = default_config  # 27
            load_spectra[load_names[27]] = default_config  # 28
            load_spectra[load_names[28]] = spectrum_config.clone(name="2015 Nissan Leaf", manufacturer="Nissan",
                                                                 nominal_voltage=208)  # 29
            load_spectra[load_names[29]] = default_config  # 30
            load_spectra[load_names[30]] = default_config  # 31
            load_spectra[load_names[31]] = default_config  # 32
        return load_spectra

    def get_generator_spectra(self, dataset='train'):
        sgen_names = self.pp_net.sgen['name'].values
        generator_spectra = {}

        return generator_spectra


if __name__ == '__main__':
    ieee33 = IEEE33TestFeeder()
    pp_net = ieee33.pp_net
    load_names = pp_net.load['name'].values
    sgen_names = pp_net.sgen['name'].values
    pass
