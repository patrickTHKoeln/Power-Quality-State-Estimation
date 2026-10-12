import logging
import os

import cimpyorm
import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.converter import from_cim

from OpenDSSWrapper.tools.config import data_path
from OpenDSSWrapper.tools.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
cgmes_path = os.path.join(data_path, 'CGMES')

# List of supported profiles and components here: https://pandapower.readthedocs.io/en/latest/converter/cgmes.html
SUPPORTED_PROFILES = ['EquipmentProfile', 'SteadyStateHypothesisProfile',
                      'StateVariablesProfile', 'TopologyProfile', 'DiagramLayoutProfile',
                      'GeographicalLocationProfile']
# Technically ['TopologyBoundaryProfile', 'EquipmentBoundaryProfile'] are also supported
# However, they define nodes again leading to errors in conversion


def cgmes_to_opendss(cgmes_files, circuit_name, output_path, load_file=None, gen_file=None):
    """
    Convert CGMES network to OpenDSS
    :param cgmes_files: Zip folder of Profile export xml files
    :param load_file: export.xml or sqlite database file containing the loads (cimpyorm format)
    :param gen_file: export.xml or sqlite database file containing the generators (cimpyorm format)
    :param circuit_name: name will be used to identify the network
    :param output_path: output path to save the resulting OpenDSS network to
    :return:
    """
    cgmes_files = check_zip_files(cgmes_files)
    net = cgmes_to_pandapower(cgmes_files)
    net = cimpyorm_to_pandapower(net, load_file, gen_file)
    from OpenDSSWrapper.converter import pandapower_to_opendss
    pandapower_to_opendss(net, circuit_name, output_path)
    return


def check_zip_files(cgmes_files):
    """
    PandaPower expects one file per profile, if the zip file contains only one file, convert to multiple files
    :param cgmes_files: zip file
    :return: zip file with multiple xml files
    """
    # unzip the file
    import zipfile
    tmp_dir = os.path.join(cgmes_path, 'tmp')
    with zipfile.ZipFile(cgmes_files, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    # check if there are multiple files
    files = os.listdir(tmp_dir)
    res = cgmes_files
    if len(files) == 1:
        logger.info("Only one CGMES file found in the zip folder, converting to one file per profile.")
        # convert the export.xml file to a SQLite database
        db_session, model = cimpyorm.parse(os.path.join(tmp_dir, files[0]))
        # check which profiles exist in the database
        profiles = db_session.query('DISTINCT profile_name FROM prop_profile').all()
        # extract the first item of the tuple, then check if it is supported by pandapower
        profiles = [profile[0] for profile in profiles if profile[0] in SUPPORTED_PROFILES]
        with open(os.path.join(cgmes_path, "out.zip"), "wb+") as f:
            export = cimpyorm.export(db_session, mode="Multi", profile_whitelist=profiles)
            f.write(export.getvalue())
        res = os.path.join(cgmes_path, "out.zip")
    # clean up tmp folder
    import shutil
    logger.info("Cleaning up temporary folder.")
    shutil.rmtree(tmp_dir)
    return res


def cgmes_to_pandapower(cgmes_files) -> pp.pandapowerNet:
    """
    Convert CGMES network to pandapower
    The resulting opendss network will be saved at the specified output_path
    :param cgmes_files: folder or zip files
    :return: pandapower network
    """
    net = from_cim.from_cim(file_list=cgmes_files, create_measurements='SV', update_assets_from_sv=True,
                            use_GL_or_DL_profile='both')
    # TODO: With above settings, loads and most generators are saved as "ward" elements, convert to "load" elements;
    #  description information is lost
    #  ThermalGeneratingUnit is saved under sgen
    #  Read the descriptions from the database; then write to appropriate pandapower elements

    line_data = {'c_nf_per_km': 0.0001,  # line capacitance (line-to-earth)      nF/km
                 'r_ohm_per_km': 0.0001,  # line resistance                       Ω/km
                 'x_ohm_per_km': 0.0001,  # line reactance                        Ω/km
                 'max_i_ka': 0.06,  # maximal thermal current               kA
                 'type': 'cs'  # cs - underground cable system
                 }

    # Creation of the standard type stdswitch
    from OpenDSSWrapper.converter.grids import BaseGrid
    pp.create_std_type(net, line_data, name=BaseGrid.SWITCH_STRING, element='line')
    logger.info(f"Created standard type {BaseGrid.SWITCH_STRING} for switches with parameters {line_data}")
    net = line_parameters_to_std_type(net)
    # check if external grid is correctly defined, else add default values
    for idx, ext_grid in net.ext_grid.iterrows():
        if np.isnan(ext_grid.s_sc_max_mva):
            net.ext_grid.loc[idx, 's_sc_max_mva'] = 0.114501334
            logger.info(
                f"External grid {ext_grid.name} has no s_sc_max_mva defined, setting to default value {net.ext_grid.loc[idx, 's_sc_max_mva']}")
        if np.isnan(ext_grid.s_sc_min_mva):
            net.ext_grid.loc[idx, 's_sc_min_mva'] = 0.114501334
            logger.info(
                f"External grid {ext_grid.name} has no s_sc_min_mva defined, setting to default value {net.ext_grid.loc[idx, 's_sc_min_mva']}")
        if np.isnan(ext_grid.rx_min):
            net.ext_grid.loc[idx, 'rx_min'] = 1.429698229
            logger.info(
                f"External grid {ext_grid.name} has no rx_min defined, setting to default value {net.ext_grid.loc[idx, 'rx_min']}")
        if np.isnan(ext_grid.rx_max):
            net.ext_grid.loc[idx, 'rx_max'] = 1.429698229
            logger.info(
                f"External grid {ext_grid.name} has no rx_max defined, setting to default value {net.ext_grid.loc[idx, 'rx_max']}")
    return net


def line_parameters_to_std_type(net: pp.pandapowerNet, eps=1e-7) -> pp.pandapowerNet:
    """
    Often, standard types are not defined for the lines in the network. This function will create a unique standard type
    for each unique combination of line parameters.
    For each unique combination of parameters, rename the std_type to something unique
    the following columns are relevant for uniqueness: r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka
    r0_ohm_per_km, x0_ohm_per_km, c0_nf_per_km, g0_us_per_km

    TODO: line types are in the cimpyorm descriptions of the lines --> extract them
    :param net: pandapower network
    :return: updated pandapower network
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


def cimpyorm_to_pandapower(net: pp.pandapowerNet, load_file=None, gen_file=None) -> pp.pandapowerNet:
    """
    Add cimpyorm injections to the existing network
    TODO: is there a better way of identifying injections? Pandapower should be able to handle EquivalentInjection according to documentation
    :param net: pandapower network to add the injections to
    :param load_file: export.xml or sqlite database file containing the loads
    :param gen_file: export.xml or sqlite database file containing the generators
    :return:
    """
    if load_file is not None:
        if load_file.endswith('.xml'):
            # parse data to sqlite db (only needs to be done once)
            db_session, model = cimpyorm.parse(load_file)
        else:
            db_session, model = cimpyorm.load(load_file)

        # types: Household, Charging station private
        injections = db_session.query(model.EquivalentInjection).all()
        for inj in injections:
            # find equivalent terminal
            term = db_session.query(model.Terminal).filter(model.Terminal.name == inj.name).first()
            # match with node
            id = term.ConnectivityNode.id

            bus_id = _match_bus_to_node(net, id)
            if bus_id is None:
                continue
            # add load
            # check if name is unique
            if 'Household' in inj.description:
                name = 'house' + id
            elif 'Charging station' in inj.description:
                name = 'cs' + id
            else:  # general load
                logger.warning(f"Unknown load type: {inj.description} at node {id}")
                name = 'load' + id
            p = inj.p if inj.p > 0 else -inj.p
            pp.create_load(net, bus=bus_id, p_mw=p, q_mvar=inj.q, name=name)

    if gen_file is None:
        return net

    if gen_file.endswith('.xml'):
        db_session, model = cimpyorm.parse(gen_file)
    else:
        db_session, model = cimpyorm.load(gen_file)

    # types: Photovoltaic, ...
    injections = db_session.query(model.EquivalentInjection).all()
    for inj in injections:
        # find equivalent terminal
        term = db_session.query(model.Terminal).filter(model.Terminal.name == inj.name).first()
        # match with node
        id = term.ConnectivityNode.id
        bus_id = _match_bus_to_node(net, id)
        if bus_id is None:
            continue
        if 'Photovoltaic' in inj.description:
            name = 'pv' + id
        else:
            logger.warning(f"Unknown generator type: {inj.description} at node {id}")
            name = 'gen' + id
        p = inj.p if inj.p > 0 else -inj.p
        # add generator
        pp.create_sgen(net, bus=bus_id, p_mw=p, q_mvar=inj.q, name=name)

    # further example queries
    pv = db_session.query(model.SolarGeneratingUnit).first()
    wind = db_session.query(model.WindGeneratingUnit).first()
    switches = db_session.query(model.Switch).all()
    all_objects = db_session.query(model.IdentifiedObject).all()
    current_limits = db_session.query(model.CurrentLimit).all()
    return net


def _match_bus_to_node(net: pp.pandapowerNet, node_id):
    """
    Match the node_id to the bus in the network
    :param net: pandapower network
    :param node_id: node id
    :return: bus id
    """
    bus = net.bus[net.bus['name'] == 'Node' + node_id]
    if bus.empty:
        logger.warning(f"Warning: bus not found for node {node_id}")
        return None
    return bus.index[0]


if __name__ == '__main__':
    gangolf_demo = os.path.join(cgmes_path,
                                'Anonymisiertes_Netz_lasten_und_einspeiser_mit_reservierungen-Gangolf_Demo1.zip')
    cgmes_to_opendss(gangolf_demo, 'gangolf_demo', os.path.join(cgmes_path, 'gangolf_demo'))

    example_cgmes = os.path.join(cgmes_path, '20240227_CGMES_example_geographical_location_profile.zip')
    gen_path = os.path.join(cgmes_path, 'Anonymisiertes_Netz_nur_einspeiser_mit_reservierungen', 'out.db')
    load_path = os.path.join(cgmes_path, 'Anonymisiertes_Netz_nur_lasten_mit_reservierungen', 'out.db')
    cgmes_to_opendss(example_cgmes, 'example', os.path.join(cgmes_path, 'example'), load_path, gen_path)
    pass
