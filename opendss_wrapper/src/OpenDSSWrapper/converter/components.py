import logging
import math

import numpy as np
import pandapower as pp
import pandas as pd

from OpenDSSWrapper.tools import config as cfg

logger = logging.getLogger(__name__)


class BaseComponent:
    def __init__(self, df: pd.Series, vn_kv: float | int, fn_hz: float | int):
        """
        Base class for all components in the network
        :param df: pandas Series with the component parameters
        :param vn_kv: nominal voltage in kV where the element resides (for transformers: high voltage bus)
        :param fn_hz: nominal frequency in Hz
        """
        self.df = df
        self.original_columns = list(df.index)
        self.vn_kv = vn_kv
        self.fn_hz = fn_hz
        self._calculate_parameters()

    def get_updated_df(self) -> tuple[pd.Series, list[str]]:
        """
        Get the updated Series with the calculated parameters and the additional columns
        :return: updated Series, additional columns
        """
        additional_columns = [col for col in list(self.df.index) if col not in self.original_columns]
        return self.df, additional_columns

    def _calculate_parameters(self):
        """
        Calculate additional parameters for the component
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def to_opendss(self, **kwargs) -> str:
        """
        Convert the component to an OpenDSS string
        :param kwargs: additional parameters
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        """
        Get opendss strings for monitors at this element
        :return: monitors, power_monitors, each a list of tuple of bus index and openDSS string
        """
        raise NotImplementedError("Implement in subclass")


class BranchComponent(BaseComponent):
    def __init__(self, df: pd.Series, vn_kv: float | int, fn_hz: float | int, from_bus: int, to_bus: int):
        """
        Base class for all components between two busses in the network. Extension of BaseComponent.
        :param df: pandas DataFrame with the component parameters
        :param vn_kv: nominal voltage in kV where the element resides (for transformers: high voltage bus)
        :param fn_hz: nominal frequency in Hz
        :param from_bus: origin bus
        :param to_bus: target bus
        """
        super().__init__(df, vn_kv, fn_hz)
        self.from_bus = from_bus
        self.to_bus = to_bus

    def get_pi_branch_admittance(self, vt_kv: float | int, f_hz: float | int, zero_seq: bool = False):
        """
        Calculate the branch admittance for a given voltage and frequency
        :param vt_kv: target voltage level in kV
        :param f_hz: frequency in Hz
        :param zero_seq: If True, the zero sequence data is used
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def get_pi_longitude_admittance(self, vt_kv: float | int, f_hz: float | int, zero_seq: bool = False):
        """
        Calculate the longitude admittance for a given voltage and frequency
        :param vt_kv: target voltage level in kV
        :param f_hz: frequency in Hz
        :param zero_seq: If True, the zero sequence data is used
        :return:
        """
        raise NotImplementedError("Implement in subclass")


class Line(BranchComponent):
    def __init__(self, df, vn_kv, fn_hz):
        from_bus = df['from_bus']
        to_bus = df['to_bus']
        super().__init__(df, vn_kv, fn_hz, from_bus, to_bus)

    def _calculate_parameters(self):
        self.df.loc['l_h_per_km'] = self.df['x_ohm_per_km'] / (2 * np.pi * self.fn_hz)
        self.df.loc['r_ohm'] = self.df['r_ohm_per_km'] * self.df['length_km']
        self.df.loc['l_h'] = self.df['l_h_per_km'] * self.df['length_km']
        if self.df['c_nf_per_km'] == 0:  # Assume default values
            if self.df['type'] == 'ol':
                self.df.loc['c_nf_per_km'] = 210
            elif self.df['type'] == 'cs':
                self.df.loc['c_nf_per_km'] = 1100  # TODO confirm values
            else:
                logger.error(
                    f"Unknown line type {self.df['type']} for line {self.df['name']}. Cannot infer capacitance.")
        self.df.loc['c_f'] = self.df['c_nf_per_km'] * 1e-9 * self.df['length_km']
        self.df.loc['g_s'] = self.df['g_us_per_km'] * 1e-6 * self.df['length_km']
        if self.df['g_s'] == 0:  # Assume default values
            self.df.loc['g_s'] = 0.000001
        # TODO: move to std_type since this is (or should be) equal for all lines of the same type
        if 'r0_ohm_per_km' not in self.df or np.isnan(self.df['r0_ohm_per_km']):
            self.df.loc['r0_ohm_per_km'] = self.df['r_ohm_per_km'] * 4
        if 'x0_ohm_per_km' not in self.df or np.isnan(self.df['x0_ohm_per_km']):
            self.df.loc['x0_ohm_per_km'] = self.df['x_ohm_per_km'] * 3
        if 'c0_nf_per_km' not in self.df or np.isnan(self.df['c0_nf_per_km']):
            self.df.loc['c0_nf_per_km'] = self.df['c_nf_per_km'] / 2

        # Absolute values for zero sequence
        self.df.loc['r0_ohm'] = self.df['r0_ohm_per_km'] * self.df['length_km']
        self.df.loc['l0_h_per_km'] = self.df['x0_ohm_per_km'] / (2 * np.pi * self.fn_hz)
        self.df.loc['l0_h'] = self.df['l0_h_per_km'] * self.df['length_km']
        self.df.loc['c0_f'] = self.df['c0_nf_per_km'] * 1e-9 * self.df['length_km']

    def get_pi_branch_admittance(self, vt_kv, f_hz, zero_seq=False, **kwargs):
        c_f = self.df['c0_f'] if zero_seq else self.df['c_f']
        xc_ohm = 1 / (2 * np.pi * f_hz * c_f)
        x_ohm = complex(0.5 * (1/self.df['g_s']) * (vt_kv / self.vn_kv) ** 2,
                        0.5 * (-xc_ohm * (vt_kv / self.vn_kv) ** 2))
        return 1/x_ohm

    def get_pi_longitude_admittance(self, vt_kv, f_hz, zero_seq=False, **kwargs):
        l_h = self.df['l0_h'] if zero_seq else self.df['l_h']
        r_ohm = self.df['r0_ohm'] if zero_seq else self.df['r_ohm']
        xl_ohm = l_h * 2 * np.pi * f_hz
        z = complex(r_ohm * (vt_kv / self.vn_kv) ** 2, xl_ohm * (vt_kv / self.vn_kv) ** 2)
        return 1 / z

    def to_opendss(self, **kwargs):
        name = self.df['name'].replace(" ", "").replace("Line", "")
        length_km = self.df['length_km']
        if 'std_type' not in self.df or pd.isna(self.df['std_type']):
            raise ValueError("Standard type must be defined")
        line_code_name = self.df['std_type']
        from_bus = self.df['from_bus']
        to_bus = self.df['to_bus']
        units = 'km'
        # index = next((i for i, item in enumerate(my_list) if item.id == 'specific_id'), -1)
        opendss_str = (f'New Line.{name} Linecode={line_code_name} bus1={str(from_bus)} bus2={str(to_bus)} '
                       f'length={str(length_km)} units={units}')
        return opendss_str

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        monitors = []
        power_monitors = []
        name = self.df['name'].replace(" ", "").replace("Line", "")
        # Add Monitors if not existing
        monitor_name = f'MonitorAtBus{str(self.from_bus)}'
        monitors.append(
            (self.from_bus, f'New Monitor.{monitor_name} element=Line.{name} Terminal=1 mode=0'))
        power_monitors.append(
            (self.from_bus, f'New Monitor.Power{monitor_name} element=Line.{name} Terminal=1 mode=1'))
        monitor_name = f'MonitorAtBus{str(self.to_bus)}'
        monitors.append(
            (self.to_bus, f'New Monitor.{monitor_name} element=Line.{name} Terminal=2 mode=0'))
        power_monitors.append(
            (self.to_bus, f'New Monitor.Power{monitor_name} element=Line.{name} Terminal=2 mode=1'))
        return monitors, power_monitors


class Switch(BranchComponent):
    SWITCH_STRING = 'std_switch'

    def __init__(self, df, vn_kv, fn_hz):
        from_bus = df['bus']
        to_bus = df['element']
        super().__init__(df, vn_kv, fn_hz, from_bus, to_bus)

    def _calculate_parameters(self):
        pass  # nothing to do here

    def is_closed(self):
        return self.df['closed']

    def get_pi_branch_admittance(self, vt_kv, f_h, zero_seq=False):
        return complex(1e-6, 1e-7)


    def get_pi_longitude_admittance(self, vt_kv, f_hz, zero_seq=False):
        if self.df['closed']:
            return complex(1e3, 1e-2)
        else:
            return complex(0, 0)

    def to_opendss(self, **kwargs):
        length_km = '0.00001'
        # nameIndex = li.name
        name = self.df['name'].replace(" ", "").replace("Switch", "")
        line_code_name = Switch.SWITCH_STRING
        from_bus = self.from_bus
        to_bus = self.to_bus
        units = 'km'
        enabled = "True" if self.df['closed'] else "False"
        opendss_str = (f'New Line.{name} Linecode={line_code_name} bus1={str(from_bus)} bus2= {str(to_bus)} '
                       f'length={str(length_km)} units={units} switch=True enabled={enabled}')
        return opendss_str

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        monitors = []
        power_monitors = []
        name = self.df['name'].replace(" ", "").replace("Switch", "")
        monitor_name = 'MonitorAtBus' + str(self.from_bus)
        monitors.append((self.from_bus, f'New Monitor.{monitor_name} element=Line.{name} Terminal=1 mode=0'))
        power_monitors.append((self.from_bus, f'New Monitor.Power{monitor_name} element=Line.{name} Terminal=1 mode=1'))
        monitor_name = 'MonitorAtBus' + str(self.to_bus)
        monitors.append(
            (self.to_bus, f'New Monitor.{monitor_name} element=Line.{name} Terminal=2 mode=0'))
        power_monitors.append((self.to_bus, f'New Monitor.Power{monitor_name} element=Line.{name} Terminal=2 mode=1'))
        return monitors, power_monitors


class StaticGenerator(BaseComponent):
    def __init__(self, df, vn_kv, fn_hz):
        super().__init__(df, vn_kv, fn_hz)
        self.opendss_type = "Generator"

    def _calculate_parameters(self):
        if 'sn_mva' not in self.df or np.isnan(self.df['sn_mva']):
            if 'p_mw' not in self.df or 'q_mvar' not in self.df:
                raise ValueError("Either sn_mva or p_mw and q_mvar must be defined")
            self.df.loc['sn_mva'] = math.sqrt(self.df['p_mw'] ** 2 + self.df['q_mvar'] ** 2)
            logger.debug(
                f"No sn_mva parameter for PVSystem {self.df['name']}. Calculating SN_MVA from p_mw and q_mvar.")

    def to_opendss(self, **kwargs):
        name = self.df['name'].replace(" ", "")
        if self.df['type'] == 'wye' or self.df['type'] == 'delta':
            # ambiguous type property for asymmetric static generators
            # try to derive type from name property
            self.df.loc['type'] = StaticGenerator._derive_sgen_type(self.df['name'])
            logger.warning(f"Type property not properly set for pandapower static generator at bus {self.df['bus']},"
                           f" derived type {self.df['type']} from name property")

        if self.df['type'] == 'PV':
            logger.info(f"PV system found at bus {self.df['bus']} with p_mw={self.df['p_mw']}"
                        f" and q_mvar={self.df['q_mvar']}")
            self.opendss_type = "PVSystem"
            # No model Parameter for PVSystem
            opendss_str = (f"New PVSystem.{name} Phases=3 bus1={self.df['bus']}"
                           f" Kv={self.vn_kv} kVA={self.df['sn_mva'] * 1000} Pf=1.0")  # Kva={self.df.sn_mva * 1000}
        elif self.df.type == 'WP':
            logger.info(f"Wind power system found at bus {self.df['bus']} with p_mw={self.df['p_mw']}"
                        f" and q_mvar={self.df['q_mvar']}")
            # TODO: add duty=wind after defining "new loadshape.wind ..."
            opendss_str = (f"New Generator.{name} Phases=3 bus1={self.df['bus']} Model=1"
                           f" Kv={self.vn_kv} kW={self.df['p_mw'] * 1000} Pf=1.0")
        elif self.df.type == 'CHP':
            logger.info(f"Combined heating and power system found at bus {self.df['bus']} with p_mw={self.df['p_mw']}"
                        f" and q_mvar={self.df['q_mvar']}")
            opendss_str = (f"New Generator.{name} Phases=3 bus1={self.df['bus']} Model=1"
                           f" Kv={self.vn_kv} kW={self.df['p_mw'] * 1000} Pf=1.0")
        else:
            logger.info(f"Unknown type of generator found at bus {self.df['bus']} with p_mw={self.df['p_mw']}"
                        f" and q_mvar={self.df['q_mvar']}. Defining as Generator.")
            opendss_str = (f"New Generator.{name} Phases=3 bus1={self.df['bus']} Model=1"
                           f" Kv={self.vn_kv} kW={self.df['p_mw'] * 1000} Pf=1.0")
        return opendss_str

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        name = self.df['name'].replace(" ", "")
        monitors = []
        power_monitors = []
        if self.df['type'] == 'PV':  # reconfirm type is correctly set (in case to_opendss was not called first)
            self.opendss_type = 'PVSystem'
        monitor_name = f'MonitorAtGenerator{name}'
        monitors.append(
            (self.df['bus'], f'New Monitor.{monitor_name} element={self.opendss_type}.{name} Terminal=1 mode=0'))
        power_monitors.append(
            (self.df['bus'], f'New Monitor.Power{monitor_name} element={self.opendss_type}.{name} Terminal=1 mode=1'))
        return monitors, power_monitors

    @staticmethod
    def _derive_sgen_type(name: str) -> str:
        """
        Derive the type of static generator from its name
        :param name: name of the static generator
        :return: type
        """
        name = name.lower()
        if 'pv' in name:
            t = 'PV'
        elif 'wp' in name or 'wind' in name or 'wka' in name:
            t = 'WP'
        elif 'chp' in name:
            t = 'CHP'
        else:
            t = 'Other'
        return t


class Transformer(BranchComponent):
    def __init__(self, df, vn_kv, fn_hz, ignore_branch_admittances=False, convert_delta=True):
        from_bus = df['hv_bus']
        to_bus = df['lv_bus']
        self.ignore_branch_admittances = ignore_branch_admittances
        self.convert_delta = convert_delta
        super().__init__(df, vn_kv, fn_hz, from_bus, to_bus)

    def _calculate_parameters(self):
        self.df.loc['sn_kva'] = self.df['sn_mva'] * 1000

        # Tap position
        for tap in ['tap', 'tap2']:  # iterate over both taps, e.g. 'tap_pos' and 'tap2_pos'
            if f'{tap}_pos' not in self.df or np.isnan(self.df[f'{tap}_pos']):
                if f'{tap}_neutral' in self.df and not np.isnan(self.df[f'{tap}_neutral']):
                    self.df.loc[f'{tap}_pos'] = self.df[f'{tap}_neutral']
                elif f'{tap}_max' in self.df and not np.isnan(
                        self.df[f'{tap}_max']) and f'{tap}_min' in self.df and not np.isnan(
                    self.df[f'{tap}_min']):
                    # 1 if in range of tap min and tap max
                    if self.df[f'{tap}_max'] >= 1 >= self.df[f'{tap}_min']:
                        self.df.loc[f'{tap}_pos'] = 1.0
                    else:
                        # random integer between min and max
                        self.df.loc[f'{tap}_pos'] = np.random.randint(self.df[f'{tap}_min'],
                                                                      self.df[f'{tap}_max'])
                else:
                    self.df.loc[f'{tap}_pos'] = 1.0

        # Voltage ratio
        self.df.loc['n'] = self.df['vn_lv_kv'] / self.df['vn_hv_kv']

        # Longitudinal component
        self.df.loc['s_kva_per_phase'] = self.df['sn_kva'] / 3
        self.df.loc['vn_v_p2e'] = self.vn_kv * 1000 / np.sqrt(3)
        self.df.loc['in_a'] = self.df['sn_kva'] / (np.sqrt(3) * self.vn_kv)
        self.df.loc['z_sc_ohm'] = 0.01 * self.df['vk_percent'] * self.df['vn_v_p2e'] ** 2 / (
                1000 * self.df['s_kva_per_phase'])
        self.df.loc['z_sc_real'] = 0.01 * self.df['vkr_percent'] * self.df['vn_v_p2e'] ** 2 / (
                1000 * self.df['s_kva_per_phase'])
        self.df.loc['z_sc_imag'] = np.sqrt(self.df['z_sc_ohm'] ** 2 - self.df['z_sc_real'] ** 2)

        self.df.loc['r_sc_ohm'] = self.df['z_sc_real']
        self.df.loc['l_sigma_h'] = self.df['z_sc_imag'] / (2 * np.pi * self.fn_hz)  # Leakage inductance

        # Transversal components

        # open loop losses -> current through R_Fe and x_h ->
        # not given in pandapower, but typically 3%...0,3%
        if 'i0_percent' not in self.df or np.isnan(self.df['i0_percent']) or self.df['i0_percent'] == 0:
            self.df.loc['i0_percent'] = 0.03

        # P_FE typically described by -> 0,3...0,05% = 1/r_FE = P_lr/S_rT
        if 'pfe_kw' not in self.df or np.isnan(self.df['pfe_kw']) or self.df['pfe_kw'] == 0:
            self.df.loc['pfe_kw'] = self.df['sn_kva'] * 0.003

        self.df.loc['pfe_kw_per_phase'] = self.df['pfe_kw'] / 3
        self.df.loc['r_fe_ohm'] = self.df['vn_v_p2e'] ** 2 / (1000 * self.df['pfe_kw_per_phase'])
        # im = U_rT^2/X_h

        self.df.loc['i_0'] = self.df['i0_percent'] * self.df['in_a']  # see above - 3% for smaller transformers
        self.df.loc['s_0_kva_per_phase'] = self.df['vn_v_p2e'] * self.df['i_0'] / 1000
        self.df.loc['q_mag_kvar_per_phase'] = np.sqrt(
            self.df['s_0_kva_per_phase'] ** 2 - self.df['pfe_kw_per_phase'] ** 2)

        self.df.loc['x_h_ohm'] = self.df['vn_v_p2e'] ** 2 / (1000 * self.df['q_mag_kvar_per_phase'])
        self.df.loc['l_h_h'] = self.df['x_h_ohm'] / (2 * np.pi * self.fn_hz)

    def get_pi_branch_admittance(self, vt_kv, f_hz, zero_seq=False):
        if self.ignore_branch_admittances:
            return complex(0, 0)

        x_h_ohm = self.df['l_h_h'] * 2 * math.pi * f_hz
        if zero_seq:
            # This only applies for the LV side of DYN!!
            x0_h_ohm = x_h_ohm * 0.75
            y_siemens = complex((1 / self.df['r_fe_ohm'] * (vt_kv / self.vn_kv) ** 2),
                                (1 / x0_h_ohm * (vt_kv / self.vn_kv) ** 2))
        elif self.convert_delta:
            z_ohm, _, _ = self._get_delta_from_star_element(vt_kv)
            y_siemens = 1 / z_ohm
        else:
            y_siemens = complex(0.5 * (1/self.df['r_fe_ohm'] * (vt_kv / self.vn_kv) ** 2),
                                0.5 * (1/x_h_ohm * (vt_kv / self.vn_kv) ** 2))

        return y_siemens

    def get_pi_longitude_admittance(self, vt_kv, f_hz, zero_seq=False):
        # z_sc_imag = l_sigma_h * 2 * np.pi * f_hz
        if self.convert_delta:
            _, _, z_ohm = self._get_delta_from_star_element(vt_kv)
        else:
            z_ohm = complex(self.df['z_sc_real'] * (vt_kv / self.vn_kv) ** 2,
                            self.df['z_sc_imag'] * (vt_kv / self.vn_kv) ** 2)
        if zero_seq:  # This applies to DYN as it will not permit zero sequences
            return 1E-9 + 1j*1E-9
        return 1 / z_ohm

    def _get_delta_from_star_element(self, vt_kv):
        # ->I_p     ab        <-I_s
        # a ---o---[___]---o--- b
        #      |           |
        #   ac|_|       bc|_|
        #      |           |
        # -----o-----------o----
        #           c
        # like in pandapower the
        conversion_factor = (vt_kv / self.vn_kv) ** 2
        a_b = 0.5 * complex(self.df['z_sc_real'] * conversion_factor,
                            self.df['z_sc_imag'] * conversion_factor)

        c = 1/complex((1/self.df['r_fe_ohm']) * conversion_factor,
                      (1/self.df['x_h_ohm']) * conversion_factor)

        sum_denominator = a_b ** 2 + 2 * c * a_b
        ac = sum_denominator / a_b
        ab = sum_denominator / c
        bc = ac
        logger.debug('PiBranches by star-delta:' + str(ac))
        logger.debug('PiBranches by x_h/r_fe / 2:' + str(c / 2))
        return ac, bc, ab

    def to_opendss(self, **kwargs):
        name = self.df['name'].replace(" ", "").replace("Trafo", "")  # OpenDSS can't handle spaces in names

        # XHL: percent reactance high to low (wdg 1 to wdg 2)
        xhl = np.sqrt(self.df['vk_percent'] ** 2 - self.df['vkr_percent'] ** 2)
        load_loss_percent = self.df['vkr_percent']
        hv_bus = self.df['hv_bus']
        lv_bus = self.df['lv_bus']
        sn_kva = self.df['sn_kva']
        vn_hv_kv = self.df['vn_hv_kv']
        vn_lv_kv = self.df['vn_lv_kv']
        tap1_pos = self.df['tap_pos']
        tap2_pos = self.df['tap2_pos']

        # Beispiel ABB 630kVA: +/-5% Anzapfung, 20kv->400V Dyn5 u_k 5,8%
        # Beispiel Schneider Elektrik Minera 315kVA: +/-2,5% Anzapfung, 20kv->400V Dyn11 u_k 4% ,
        # Kurzschlussverluste = 1900W, Leerlaufverluste = 1320W
        # Beispiel Siemens 250kVA: +/-4% Anzapfung, 20kv->400V Dyn5 u_k 3,96% , Kurzschlussverluste = 2310W

        # Transversal Components
        # current through R_Fe -> not given in pandapower, but typically 3%...0,3%
        # P_FE typically described by -> 0,3...0,05% = 1/r_FE = P_lr/S_rT

        r_fe_percent = 0.3  # FH Strahlsund -> Skript Transformator
        # im = U_rT^2/X_h
        x_h_percent = 3  # estimated i_m=3% for smaller transformers (FH Strahlsund -> Skript Transformator)

        # defaults:  phases=3 windings=2
        opendss_str = (f'New "Transformer.{name}" Windings={str(2)} Basefreq={str(self.fn_hz)}'
                       f' XHL={str(xhl)} %LoadLoss={str(load_loss_percent)} %NoloadLoss={str(r_fe_percent)}'
                       f' %imag={str(x_h_percent)}'
                       f' wdg=1 bus={str(hv_bus)} conn=delta'
                       f' kV={str(vn_hv_kv)} kVA={str(sn_kva)} tap={str(tap1_pos)}'
                       f' wdg=2 bus={str(lv_bus)} conn=wye'
                       f' kV={str(vn_lv_kv)} kVA={str(sn_kva)} tap={str(tap2_pos)}')
        return opendss_str


class Load(BaseComponent):
    def __init__(self, df, vn_kv, fn_hz):
        super().__init__(df, vn_kv, fn_hz)

    def _calculate_parameters(self):
        return  # nothing to do here

    def to_opendss(self, **kwargs):
        name = self.df['name'].replace("Load ", "")
        bus_index = self.df['bus']
        kW = str(self.df['p_mw'] * 1000)
        kVar = str(self.df['q_mvar'] * 1000)
        conn_type = self.df['type']
        opendss_str = (f'New Load.{name} Bus1={str(bus_index)} Phases=3 Conn={conn_type} Model=1 '
                       f'kV={self.vn_kv} kW={kW} kvar={kVar}')
        return opendss_str

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        monitors = []
        power_monitors = []
        name = self.df['name'].replace("Load ", "")
        monitor_name = f'MonitorAtLoad{name}'
        monitors.append((self.df['bus'], f'New Monitor.{monitor_name} element=Load.{name} Terminal=1 mode=0'))
        power_monitors.append(
            (self.df['bus'], f'New Monitor.Power{monitor_name} element=Load.{name} Terminal=1 mode=1'))
        return monitors, power_monitors


class ExternalGrid(BranchComponent):
    def __init__(self, df, vn_kv, fn_hz):
        from_bus = df['bus']
        to_bus = -1
        self.is_asymmetric = False
        self.name = 'source'
        super().__init__(df, vn_kv, fn_hz, from_bus=from_bus, to_bus=to_bus)

    def _calculate_parameters(self):
        if 's_sc_max_mva' not in self.df:
            if not ('max_p_mw' in self.df and 'max_q_mvar' in self.df):
                raise ValueError(
                    "External grid missing required information w.r.t. short circuit: s_sc_max_mva or max_p_mw and max_q_mvar.")
                # TODO assume some default, e.g. through a pandapower short circuit calculation
            self.df.loc['s_sc_max_mva'] = cfg.SHORT_CIRCUIT_EXT_GRID_TOLERANCE * np.sqrt(
                self.df['max_p_mw'] ** 2 + self.df['max_q_mvar'] ** 2)

        if 's_sc_min_mva' not in self.df:
            if not ('min_p_mw' in self.df and 'min_q_mvar' in self.df):
                raise ValueError(
                    "External grid missing required information w.r.t. short circuit. s_sc_min_mva or min_p_mw and min_q_mvar.")
                # TODO assume some default
            self.df.loc['s_sc_min_mva'] = cfg.SHORT_CIRCUIT_EXT_GRID_TOLERANCE * np.sqrt(
                self.df['min_p_mw'] ** 2 + self.df['min_q_mvar'] ** 2)
        self.df.loc['s_sc_mva'] = 0.5 * (self.df['s_sc_max_mva'] + self.df['s_sc_min_mva'])

        self.df.loc['i_sc_max_a'] = self.df['s_sc_max_mva'] * 1000 * 1000 / (self.vn_kv * 1000 * math.sqrt(3))
        self.df.loc['i_sc_min_a'] = self.df['s_sc_min_mva'] * 1000 * 1000 / (self.vn_kv * 1000 * math.sqrt(3))
        self.df.loc['z_max_ohm'] = cfg.SHORT_CIRCUIT_EXT_GRID_TOLERANCE * (self.vn_kv * 1000) / (
                math.sqrt(3) * self.df['i_sc_max_a'])
        self.df.loc['z_min_ohm'] = cfg.SHORT_CIRCUIT_EXT_GRID_TOLERANCE * (self.vn_kv * 1000) / (
                math.sqrt(3) * self.df['i_sc_min_a'])

        if 'rx_max' not in self.df:
            if 'r_max_ohm' not in self.df:
                if 'max_p_mw' not in self.df:
                    raise ValueError("External grid missing required information max_p_mw, rx_max or r_max_ohm.")
                self.df.loc['cos_phi_max'] = self.df['max_p_mw'] / self.df['s_sc_max_mva']
                self.df.loc['r_max_ohm'] = self.df['z_max_ohm'] * self.df['cos_phi_max']
            if 'x_max_ohm' not in self.df:
                self.df.loc['x_max_ohm'] = math.sqrt(self.df['z_max_ohm'] ** 2 - self.df['r_max_ohm'] ** 2)
            self.df.loc['rx_max'] = self.df['r_max_ohm'] / self.df['x_max_ohm']
        if 'rx_min' not in self.df:
            if 'r_min_ohm' not in self.df:
                if 'min_p_mw' not in self.df:
                    raise ValueError("External grid missing required information min_p_mw, rx_min or r_min_ohm.")
                self.df.loc['cos_phi_min'] = self.df['min_p_mw'] / self.df['s_sc_min_mva']
                self.df.loc['r_min_ohm'] = self.df['z_max_ohm'] * self.df['cos_phi_min']
            if 'x_min_ohm' not in self.df:
                self.df.loc['x_min_ohm'] = math.sqrt(self.df['z_min_ohm'] ** 2 - self.df['r_min_ohm'] ** 2)
            self.df.loc['rx_min'] = self.df['r_min_ohm'] / self.df['x_min_ohm']

        ratio_rx = 0.5 * (self.df['rx_max'] + self.df['rx_min'])
        # normalerweise wird die Impedanz über c*U^2/S''k berechnet, um der Quellspannung getreu zu bleiben,
        # wird U/(sqrt(3)*I_k) genutzt
        self.df.loc['i_0_a'] = self.df['s_sc_mva'] * 1000 * 1000 / (self.vn_kv * 1000 * math.sqrt(3))
        i_0_a_complex = complex(self.df['i_0_a'], 0)

        self.df.loc['z_q_ohm'] = (self.vn_kv * 1000) / (math.sqrt(3) * self.df['i_0_a'])

        if ratio_rx <= 0:
            self.df.loc['x_q_ohm'] = 0.995 * self.df['z_q_ohm']
            self.df.loc['r_q_ohm'] = 0.1 * self.df['z_q_ohm']
        else:
            self.df.loc['r_q_ohm'] = self.df['z_q_ohm'] / math.sqrt(1 + ratio_rx ** 2)
            self.df.loc['x_q_ohm'] = ratio_rx * self.df['r_q_ohm']

        self.df.loc['l_q_h'] = self.df['x_q_ohm'] / (2 * np.pi * self.fn_hz)

    def get_i0_at_voltage_level(self, voltage0_complex):
        # complex_current_i0_mag = self.sc_mva * 1000 * 1000 / (voltage0Complex * 1000 * 3)
        complex_current_i0_mag = (voltage0_complex * 1000) / complex(self.df['r_q_ohm'], self.df['x_q_ohm'])
        # (vEbene * math.sqrt(3) * math.sqrt(3))
        return complex_current_i0_mag

    def to_opendss(self, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']

        else:
            self.name = "ExternalGrid"

        if 'is_asymmetric' in kwargs:
            self.is_asymmetric = kwargs['is_asymmetric']

        XR1 = 1 / (0.5 * (self.df['rx_max'] + self.df['rx_min']))
        R1 = round(self.df['z_q_ohm'] / np.sqrt(1 + XR1 ** 2), 6)
        X1 = round(XR1 * R1, 6)
        R0 = round(R1 * 4, 6)
        X0 = round(X1 * 3, 6)
        str_impedances = f"R1={str(R1)} X1={str(X1)} R0={str(R0)} X0={str(X0)}"

        bus = self.df['bus']

        basekv = str(round(self.vn_kv/math.sqrt(3), 6)) if self.is_asymmetric else str(self.vn_kv)

        opendss_str = (f"New Circuit.{self.name} basekv={basekv} bus1={str(bus)} pu={str(self.df['vm_pu'])} "
                       f"{str_impedances}")
        if self.is_asymmetric:
            opendss_str = opendss_str.replace(f'Circuit.{self.name}', f'Circuit.{self.name}1')
            opendss_str = opendss_str.replace(f'bus1={str(bus)}', f'bus1={str(bus)}.1')
            opendss_str += (f" Angle=0 phases=1\n"
                            f"New Vsource.source2 basekv={basekv} phases=1 bus1={str(bus)}.2 Angle=240 {str_impedances}\n"
                            f"New Vsource.source3 basekv={basekv} phases=1 bus1={str(bus)}.3 Angle=120 {str_impedances}")
        return opendss_str

    def get_opendss_monitors(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        monitors = []
        power_monitors = []
        bus = self.df['bus']
        if self.is_asymmetric:
            # There is no way to monitor three single phase devices with one monitor
            """
            for l in range(1, 4):
                elem_name = "source" + str(l) if l > 1 else "source"
                monitors.append(
                    (bus, f'New Monitor.MonitorAtVSource{str(bus)}_{str(l)} element=VSource.{elem_name} Terminal=1 mode=0'))
                power_monitors.append(
                    (bus, f'New Monitor.PowerMonitorAtVSource{str(bus)}_{str(l)} element=VSource.{elem_name} Terminal=1 mode=1'))
                            """
        else:
            monitors.append((bus, f'New Monitor.MonitorAtVSource{str(bus)} element=VSource.source Terminal=1 mode=0'))
            power_monitors.append(
                (bus, f'New Monitor.PowerMonitorAtVSource{str(bus)} element=VSource.source Terminal=1 mode=1'))
        return monitors, power_monitors

    def get_pi_branch_admittance(self, vt_kv, f_hz, zero_seq=False):
        raise NotImplementedError()

    def get_pi_longitude_admittance(self, vt_kv, f_hz, zero_seq=False):
        r_q_ohm = round(self.df['r_q_ohm'] * 4, 6) if zero_seq else self.df['r_q_ohm']
        x_q_ohm = round(self.df['x_q_ohm'] * 3, 6) if zero_seq else self.df['x_q_ohm']

        l_q_h = x_q_ohm / (2 * np.pi * self.fn_hz)
        x_q_ohm = l_q_h * 2 * np.pi * f_hz
        z_ohm = complex(r_q_ohm, x_q_ohm)
        z_ohm *= (vt_kv / self.vn_kv) ** 2
        return 1 / z_ohm


def infer_line_std_types(net: pp.pandapowerNet, eps=1e-7) -> pp.pandapowerNet:
    """
    Often, standard types are not defined for the lines in the network. This function will create a unique standard type
    for each unique combination of line parameters.
    the following columns are relevant for uniqueness: r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka
    r0_ohm_per_km, x0_ohm_per_km, c0_nf_per_km, g0_us_per_km
    TODO: use .loc and ['key'] instead of attribute access
    :param net: pandapower network
    :param eps: tolerance; if the difference between two values is less than eps, they are considered equal
    :return: updated pandapower network with added standard types
    """

    # Step 1: Find all unique combinations of relevant columns
    relevant_columns = ['r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km', 'max_i_ka',
                        'r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km', 'g0_us_per_km']
    for column in relevant_columns:
        if column not in net.line:  # columns must exist in network
            net.line[column] = pd.Series()  # add empty series (nan)
    # Round to avoid floating point errors
    net.line[relevant_columns] = net.line[relevant_columns].round(5)
    unique_combinations = net.line[relevant_columns].drop_duplicates()

    # Step 2: Generate a unique name for each combination and update std_type
    for idx, row in unique_combinations.iterrows():
        std_type_name = f"CustomType_{idx}"
        # Find lines with this unique combination
        condition = (
                ((np.abs(net.line.r_ohm_per_km - row.r_ohm_per_km) < eps) | (
                        pd.isna(net.line.r_ohm_per_km) & pd.isna(row.r_ohm_per_km))) &
                ((np.abs(net.line.x_ohm_per_km - row.x_ohm_per_km) < eps) | (
                        pd.isna(net.line.x_ohm_per_km) & pd.isna(row.x_ohm_per_km))) &
                ((np.abs(net.line.c_nf_per_km - row.c_nf_per_km) < eps) | (
                        pd.isna(net.line.c_nf_per_km) & pd.isna(row.c_nf_per_km))) &
                ((np.abs(net.line.g_us_per_km - row.g_us_per_km) < eps) | (
                        pd.isna(net.line.g_us_per_km) & pd.isna(row.g_us_per_km))) &
                ((np.abs(net.line.max_i_ka - row.max_i_ka) < eps) | (
                        pd.isna(net.line.max_i_ka) & pd.isna(row.max_i_ka))) &
                ((np.abs(net.line.r0_ohm_per_km - row.r0_ohm_per_km) < eps) | (
                        pd.isna(net.line.r0_ohm_per_km) & pd.isna(row.r0_ohm_per_km))) &
                ((np.abs(net.line.x0_ohm_per_km - row.x0_ohm_per_km) < eps) | (
                        pd.isna(net.line.x0_ohm_per_km) & pd.isna(row.x0_ohm_per_km))) &
                ((np.abs(net.line.c0_nf_per_km - row.c0_nf_per_km) < eps) | (
                        pd.isna(net.line.c0_nf_per_km) & pd.isna(row.c0_nf_per_km))) &
                ((np.abs(net.line.g0_us_per_km - row.g0_us_per_km) < eps) | (
                        pd.isna(net.line.g0_us_per_km) & pd.isna(row.g0_us_per_km)))
        )
        # First, check if a name for this combination already exists
        series = net.line.loc[condition, 'std_type']
        if series.size > 0 and series.nunique() == 1:
            indices = set(series.index)
            all_indices = set(net.line[net.line['std_type'] == series.iloc[0]].index)
            diff = indices - all_indices
            if not diff:  # if the original name has elements that do not meet the condition, we must choose a new name
                continue
        net.line.loc[condition, 'std_type'] = std_type_name

        # Step 3: Add to std_types
        if 'line' not in net.std_types:
            net.std_types['line'] = {}

        net.std_types['line'][std_type_name] = {
            'c_nf_per_km': row.c_nf_per_km, 'r_ohm_per_km': row.r_ohm_per_km,
            'x_ohm_per_km': row.x_ohm_per_km, 'max_i_ka': row.max_i_ka,
            'g_us_per_km': row.g_us_per_km, 'r0_ohm_per_km': row.r0_ohm_per_km,
            'x0_ohm_per_km': row.x0_ohm_per_km, 'c0_nf_per_km': row.c0_nf_per_km,
            'g0_us_per_km': row.g0_us_per_km
        }
    return net


def update_line_std_types(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Update the standard types in the network with the latest values from the lines
    line values might have changed to due default settings
    :param net:
    :return:
    """
    for idx, line in net.line.iterrows():
        std_type = line['std_type']
        net.std_types['line'][std_type] = {
            'c_nf_per_km': line.get('c_nf_per_km', np.nan),
            'r_ohm_per_km': line.get('r_ohm_per_km', np.nan),
            'x_ohm_per_km': line.get('x_ohm_per_km', np.nan),
            'max_i_ka': line.get('max_i_ka', np.nan),
            'g_us_per_km': line.get('g_us_per_km', np.nan),
            'r0_ohm_per_km': line.get('r0_ohm_per_km', np.nan),
            'x0_ohm_per_km': line.get('x0_ohm_per_km', np.nan),
            'c0_nf_per_km': line.get('c0_nf_per_km', np.nan),
            'g0_us_per_km': line.get('g0_us_per_km', np.nan)
        }
    return net


def get_all_grid_components(net: pp.pandapowerNet, vn_kvs: np.ndarray, branch_only: bool = True,
                            update_net=True):
    """
    Get a list of all grid components in the network
    :param net: pandapower network
    :param vn_kvs: nominal voltage level of all components, order in bus order
    :param branch_only: if True, only BranchComponents are returned
    :param update_net: if True, updates the original pandapower network with calculated parameters
    caution: new columns are added manually to the Series.
    :return:
    """
    if not update_net:
        from copy import deepcopy
        net = deepcopy(net)
    grid_components = []
    fn_hz = net.f_hz if hasattr(net, 'f_hz') else 50
    for idx, t in net.trafo.iterrows():
        transformer = Transformer(t, vn_kvs[t['hv_bus']], fn_hz, ignore_branch_admittances=False)
        grid_components.append(transformer)
        df, additional_columns = transformer.get_updated_df()
        # update the original dataframe, and add columns if necessary
        for col in additional_columns:
            if col not in net.trafo:
                net.trafo[col] = pd.Series()
        net.trafo.loc[idx] = transformer.df

    for idx, ext in net.ext_grid.iterrows():
        ext_grid = ExternalGrid(ext, vn_kvs[ext['bus']], fn_hz)
        grid_components.append(ext_grid)
        df, additional_columns = ext_grid.get_updated_df()
        # update the original dataframe, and add columns if necessary
        for col in additional_columns:
            if col not in net.ext_grid:
                net.ext_grid[col] = pd.Series()
        net.ext_grid.loc[idx] = df

    for idx, l in net.line.iterrows():
        if not l['in_service']:
            continue
        line = Line(l, vn_kvs[l['from_bus']], fn_hz)
        grid_components.append(line)
        df, additional_columns = line.get_updated_df()
        # update the original dataframe, and add columns if necessary
        for col in additional_columns:
            if col not in net.line:
                net.line[col] = pd.Series()
        net.line.loc[idx] = df

    for idx, s in net.switch.iterrows():
        switch = Switch(s, vn_kvs[s['bus']], fn_hz)
        grid_components.append(switch)

    net = update_line_std_types(net)
    net = infer_line_std_types(net)
    if branch_only:
        return grid_components

    for idx, s in net.switch.iterrows():
        switch = Switch(s, vn_kvs[s['bus']], fn_hz)
        df, additional_columns = switch.get_updated_df()
        # update the original dataframe, and add columns if necessary
        for col in additional_columns:
            if col not in net.switch:
                net.switch[col] = pd.Series()
        net.switch.loc[idx] = df
    for idx, s in net.sgen.iterrows():
        sgen = StaticGenerator(s, vn_kvs[s['bus']], fn_hz)
        grid_components.append(sgen)
        df, additional_columns = sgen.get_updated_df()
        for col in additional_columns:
            if col not in net.sgen:
                net.sgen[col] = pd.Series()
        net.sgen.loc[idx] = df
    for idx, l in net.load.iterrows():
        load = Load(l, vn_kvs[l['bus']], fn_hz)
        grid_components.append(load)
        df, additional_columns = load.get_updated_df()
        for col in additional_columns:
            if col not in net.load:
                net.load[col] = pd.Series()
        net.load.loc[idx] = df
    return grid_components
