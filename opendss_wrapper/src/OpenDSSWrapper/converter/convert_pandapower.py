import logging
import os

import numpy as np
import pandapower as pp
import pandas as pd

from OpenDSSWrapper.converter import get_all_grid_components, ExternalGrid, Switch, Line, Transformer, \
    StaticGenerator, Load

logger = logging.getLogger(__name__)


def pandapower_to_opendss(net: pp.pandapowerNet, name: str, output_path: str, monitor_loads=True,
                          monitor_generators=True, monitor_power=False):
    """
    Convert a pandapower network to OpenDSS scripts
    The scripts are saved at the given location
    :param net:
    :param name: of the network
    :param output_path: path to save the OpenDSS scripts to
    :param monitor_loads: if True adds current and voltage monitors to load objects
    :param monitor_generators: if True adds current and voltage monitors to generator objects
    :param monitor_power: if True adds power monitors to lines, and if enabled to loads and generators
    :return:
    """
    logger.info(
        f"Converting pandapower network {name} to OpenDSS, saving to {output_path}, monitoring loads: {monitor_loads}, "
        f"monitoring generators: {monitor_generators}, monitoring power: {monitor_power}")
    os.makedirs(output_path, exist_ok=True)  # Create the output path if it does not exist

    # allowed name characters
    allowed_chars = 'abcdefghijklmnopqrstuvwxyzöäüABCDEFGHIJKLMNOPQRSTUVWXYZÖÄÜ0123456789-'
    if not all(c in allowed_chars for c in name):
        raise ValueError(f"Grid name {name} contains invalid characters. Only {allowed_chars} are allowed.")

    grid_components = get_all_grid_components(net, np.array(net['bus']['vn_kv']), branch_only=False, update_net=True)
    vsources = []
    lines = []
    loads = []
    renewables = []
    transformers = []
    monitors = []
    grid_impedances = []
    # the following require special filtering
    line_monitors = []
    line_power_monitors = []

    is_asymmetric = False
    if 'is_asymmetric' in net.keys():
        is_asymmetric = net['is_asymmetric']

    for gc in grid_components:
        if isinstance(gc, ExternalGrid):
            vsources.extend(list(gc.to_opendss(name=name, is_asymmetric=is_asymmetric).split('\n')))
            vm, vpm = gc.get_opendss_monitors()
            monitors.extend(vm)
            if monitor_power:
                monitors.extend(vpm)
        elif isinstance(gc, Line):
            lines.append(gc.to_opendss())
            lm, lpm = gc.get_opendss_monitors()
            line_monitors.extend(lm)
            if monitor_power:
                line_power_monitors.extend(lpm)
        elif isinstance(gc, Switch):
            lines.append(gc.to_opendss())
            sm, spm = gc.get_opendss_monitors()
            line_monitors.extend(sm)
            if monitor_power:
                line_monitors.extend(spm)
        elif isinstance(gc, Load):
            loads.append(gc.to_opendss())
            pm, ppm = gc.get_opendss_monitors()
            if not monitor_loads:
                pm, ppm = [], []
            monitors.extend(pm)
            if monitor_power:
                monitors.extend(ppm)
        elif isinstance(gc, Transformer):
            transformers.append(gc.to_opendss())
        elif isinstance(gc, StaticGenerator):
            renewables.append(gc.to_opendss())
            sm, spm = gc.get_opendss_monitors()
            if not monitor_generators:
                sm, spm = [], []
            monitors.extend(sm)
            if monitor_power:
                monitors.extend(spm)
        else:
            raise ValueError(f'Grid component unknown.')

    filtered_lines_monitors = filter_duplicates(line_monitors)
    filtered_line_power_monitors = filter_duplicates(line_power_monitors)
    monitors = [tup[1] for tup in monitors]
    all_monitors = monitors + filtered_line_power_monitors + filtered_lines_monitors

    os.makedirs(output_path, exist_ok=True)

    line_types = get_unique_line_types(net)
    line_codes = convert_line_codes(net, line_types)
    if 'grid_impedance' in net.keys():
        vsources = convert_grid_impedance(net, vsources)
    master = create_master(net, vsources, output_path)
    bus_coords = convert_bus_coords(net)


    # write_lines_to_file(path, 'Vsource.dss', lines_vsources)
    write_to_file(output_path, 'LineCodes.dss', line_codes, master)
    write_to_file(output_path, 'Lines.dss', lines, master)
    write_to_file(output_path, 'Transformers.dss', transformers, master)
    write_to_file(output_path, 'Loads.dss', loads, master)
    write_to_file(output_path, 'Renewables.dss', renewables, master)
    write_to_file(output_path, 'Monitors.dss', all_monitors, master)
    write_to_file(output_path, 'BusCoords.dss', bus_coords, master)
    write_to_file(output_path, 'Master.dss', master)


def convert_bus_coords(net):
    """
    Convert Bus Geo Coordinates
    :param net: pandapower net
    :return: bus coordinates as lines
    """
    bus_coords = []
    # virtualLoadsAndMonitors = []
    for b, bc in net.bus_geodata.iterrows():
        bus = str(b)
        x = int((bc.x + 5) * 10)
        y = int((bc.y + 5) * 10)
        bus_coords.append(bus + ', ' + str(x) + ', ' + str(y))
        # on every bus add a mini-Load to make it measurable by monitor object
        # name_load = 'VirtualLoadAtBus' + bus
        # virtual_loads_and_monitors.append(
        # 'New Load.' + name_load + ' Bus1=' + bus + '.1 Phases=3 Conn=Wye Model=1 kW=0.001 kvar=0')
        # name_monitor = 'MonitorAtBus' + bus
    return bus_coords


def convert_line_codes(net, line_types):
    line_codes = []
    base_freq = net.f_hz
    for t in line_types:
        if t == Switch.SWITCH_STRING:
            lc = pd.Series(net.std_types['line'][t])
        else:
            lc = [item for _, item in net.line.iterrows() if item.std_type == t][0]

        lc_r = lc.r_ohm_per_km
        lc_x = lc.x_ohm_per_km
        lc_c = lc.c_nf_per_km
        lc_max_curr = lc.max_i_ka
        nphases = str(3)
        units = 'km'

        str_line_code = (f'New LineCode.{t} nphases={str(nphases)} baseFreq={str(base_freq)} units={units}'
                         f' R1={str(lc_r)} X1={str(lc_x)} C1={str(lc_c)} Normamps={str(lc_max_curr)}')
        if 'r0_ohm_per_km' in lc and 'x0_ohm_per_km' in lc and 'c0_nf_per_km' in lc:
            str_line_code += f' R0={str(lc["r0_ohm_per_km"])} X0={str(lc["x0_ohm_per_km"])} C0={str(lc["c0_nf_per_km"])}'
        line_codes.append(str_line_code)

    return line_codes


def convert_grid_impedance(net, vsources):

    impedance_lines = []
    updated_vsources = []
    phase = 0
    has_neutral_reactor = 'zn' in net['grid_impedance'].columns
    for vs in vsources:
        base_kv = vs.split('basekv=')[1].split(' ')[0]
        current_vs_bus = vs.split('bus1=')[1].split(' ')[0]

        if '.' in current_vs_bus:

            impedance_lines.clear()
            phase = current_vs_bus.split('.')[1]

            current_vs_bus = current_vs_bus.split('.')[0]
            # Phase impedance
            impedance_lines.extend(create_phase_reactor(net, current_vs_bus, f'{phase}', phase, base_kv))
            # Add neutral reactor only once
            if phase == '1' and has_neutral_reactor:
                impedance_lines.extend(create_phase_reactor(net, current_vs_bus, f'N', 0, base_kv))
        else:
            # Phase L1
            impedance_lines.extend(create_phase_reactor(net, current_vs_bus, 'l1n', 1, base_kv))
            # Phase L2
            impedance_lines.extend(create_phase_reactor(net, current_vs_bus, 'l2n', 2, base_kv))
            # Phase L3
            impedance_lines.extend(create_phase_reactor(net, current_vs_bus, 'l3n', 3, base_kv))
            # Neutral
            if has_neutral_reactor:
                impedance_lines.append(create_phase_reactor(net, current_vs_bus, f'N', 0, base_kv))

        # update v source to high power as the new grid impedance is in now series
        vs = vs.replace('bus1=' + current_vs_bus, f'bus1={current_vs_bus}_{current_vs_bus}')
        current_r1 = vs.split('R1=')[1].split(' ')[0]
        current_r0 = vs.split('R0=')[1].split(' ')[0]
        current_x1 = vs.split('X1=')[1].split(' ')[0]
        current_x0 = vs.split('X0=')[1].split(' ')[0]
        vs = vs.replace('R1=' + current_r1, 'R1=0.000001')
        vs = vs.replace('X1=' + current_x1, 'X1=0.000001')
        vs = vs.replace('R0=' + current_r0, 'R0=0.000001')
        vs = vs.replace('X0=' + current_x0, 'X0=0.000001')
        updated_vsources.append(vs)
        updated_vsources.extend(impedance_lines)

    return updated_vsources


def create_phase_reactor(net, bus, phase, terminal, base_kv, include_dc=False):
    reactor_lines = []
    phase_imp_data = net['grid_impedance'][net['grid_impedance'].index.str.contains(f'z{phase}')]
    mag_imp_data = phase_imp_data[phase_imp_data.index == f'z{phase}']
    ang_imp_data = phase_imp_data[phase_imp_data.index == f'z{phase}angle']
    fund_r = mag_imp_data[net.f_hz].values[0] * np.cos(ang_imp_data[net.f_hz].values[0]*np.pi/180)
    fund_x = mag_imp_data[net.f_hz].values[0] * np.sin(ang_imp_data[net.f_hz].values[0]*np.pi/180)

    # l/r curve data
    r_curve_name = f'r_curve_{phase}'
    l_curve_name = f'l_curve_{phase}'
    x_f = ''
    y_r = ''
    y_x = ''
    npts = 0
    for f in phase_imp_data.columns:
        if include_dc or f > 0:
            x_f += '{:.12f}'.format(f) + ','
            r = mag_imp_data[f].values[0] * np.cos(ang_imp_data[f].values[0] * np.pi / 180)
            x = mag_imp_data[f].values[0] * np.sin(ang_imp_data[f].values[0] * np.pi / 180)
            y_r += '{:.12f}'.format(r/fund_r) + ','
            y_x += '{:.12f}'.format(x/fund_x) + ','
            npts += 1
    x_f = x_f[:-1]
    y_r = y_r[:-1]
    y_x = y_x[:-1]
    reactor_lines.append(f'New XYCurve.{r_curve_name} npts={npts} Yarray=({y_r}) Xarray=({x_f})')
    reactor_lines.append(f'New XYCurve.{l_curve_name} npts={npts} Yarray=({y_x}) Xarray=({x_f})')

    reactor_lines.append(
        f'New Reactor.{phase} phases=1 conn=wye R={fund_r} X={fund_x} '
        f'kv={base_kv} bus1={bus}_{bus}.{terminal} bus2={bus}.{terminal} RCurve={r_curve_name} LCurve={l_curve_name}')
    return reactor_lines


def filter_duplicates(items: list[tuple]):
    """
    Filter duplicates from items
    :param items: list of tuples where the first item in the tuple is an identifier
    :return:
    """
    already_added = set()
    filtered_items = []

    for mon in items:
        if mon[0] not in already_added:
            already_added.add(mon[0])
            filtered_items.append(mon[1])
    return filtered_items


def get_unique_line_types(net):
    """
    Get unique line types from pandapower network
    :param net: pandapower network
    :return: List of line types
    """
    line_types = list(set(net.line['std_type']))
    if not net.switch.empty:
        line_types.append(Switch.SWITCH_STRING)
    return line_types


def create_master(net: pp.pandapowerNet, vsources: list[str], path: str = None):
    """
    Master DSS file
    :param net: Pandapower network
    :param vsources: list of voltage sources
    :param path: path to save the file to
    :return: openDSS master file as lines
    """
    master = ["clear"]

    # General
    base_freq = net.f_hz
    master.append('set DefaultBaseFreq=' + str(base_freq))

    # region External grid and Circuit
    master.extend(vsources)
    # endregion

    # region master lines and redirect
    # master.append("Redirect LoadShape.DSS")
    # master.append("Redirect Vsource.dss")
    master.append("Redirect LineCodes.dss")
    master.append("Redirect Lines.dss")
    master.append("Redirect Transformers.dss")
    master.append("Redirect Loads.dss")
    master.append("Redirect Renewables.dss")
    master.append("Redirect Monitors.dss")
    # master.append("Redirect EnergyMeter.dss")

    master.append("MakeBusList")
    # Bus Coords seems to require absolute path
    master.append("Buscoords " + os.path.join(path, "BusCoords.dss"))
    voltage_bases = net.bus['vn_kv'].unique()  # get all unique voltage bases
    voltage_bases_str = ', '.join(str(value) for value in voltage_bases)  # convert to strings
    master.append(f"Set Voltagebases=({voltage_bases_str})")
    master.append("CalcVoltagebases")

    return master


def write_to_file(path, file_name, lines, master=None):
    if len(lines) > 0:
        with open(os.path.join(path, file_name), 'w') as file:
            file.write('\n'.join(lines))
    elif master:
        master.remove(f"Redirect {file_name}")
