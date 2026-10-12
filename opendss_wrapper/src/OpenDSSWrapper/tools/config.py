import configparser
import os

config = configparser.ConfigParser()
config.sections()
config_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# check if local_config.ini exists and read it
if os.path.exists(os.path.join(config_path, 'local_config.ini')):
    config.read(os.path.join(config_path, 'local_config.ini'))
else:
    config.read(os.path.join(config_path, 'config.ini'))

data_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['data_path'])))
results_sub_path = config['DEFAULT']['results_sub_path']
transient_path = os.path.join(data_path, 'transients')
# SQLite databases
spectrum_db_name = config['DEFAULT']['spectrum_db_name']
power_profile_db_name = config['DEFAULT']['power_profile_db_name']

# Logging
log_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['log_path'])))

# PostgreSQL database
postgres_config = {
    "dbname": config['POSTGRESQL']['dbname'],
    "user": config['POSTGRESQL']['user'],
    "password": config['POSTGRESQL']['password'],
    "host": config['POSTGRESQL']['host'],
    "port": config['POSTGRESQL']['port']
}

DC_FREQUENCY = float(config['CONSTANTS']['dc_frequency'])
SHORT_CIRCUIT_EXT_GRID_TOLERANCE = float(config['CONSTANTS']['short_circuit_ext_grid_tolerance'])

# Simulation
MULTI_PROCESSING = True if config['SIMULATION']['multi_processing'] == 'True' else False
MONITOR_LOADS = True if config['SIMULATION']['monitor_loads'] == 'True' else False
MONITOR_GENERATORS = True if config['SIMULATION']['monitor_generators'] == 'True' else False
MONITOR_POWER = True if config['SIMULATION']['monitor_power'] == 'True' else False
SPECTRA_INTERPOLATION = config['SIMULATION']['spectra_interpolation']

DEBUG_MODE = config['LOGGING']['sim_debug_mode']
DEBUG_PLOT = config['LOGGING']['sim_debug_plot']

# maximum harmonic values used for random spectra
# right now these are at 3x the maximum allowed deviations according to EN 50160.
# Values with comments have been adjusted manually (value in comments = 3x max deviation)
MAX_HARMONIC_VALUES = {
    1: 1,  # this one is not actually used, only serves as reference point
    2: 0.04, # 0.06
    3: 0.05, # 0.15
    4: 0.03,
    5: 0.06, # 0.18
    6: 0.015,
    7: 0.05, # 0.15
    8: 0.015,
    9: 0.015,
    10: 0.015,
    11: 0.035,
    12: 0.015,
    13: 0.03,
    14: 0.015,
    15: 0.01,
    16: 0.015,
    17: 0.02,
    18: 0.015,
    19: 0.015,
    20: 0.015,
    21: 0.0075,
    22: 0.015,
    23: 0.015,
    24: 0.015,
    25: 0.015,
    26: 0.015, # from here on, values were added manually. TODO use some form of extrapolation 
    27: 0.015,
    28: 0.015,
    29: 0.015,
    30: 0.015,
    31: 0.015,
    32: 0.015,
    33: 0.015,
    34: 0.015,
    35: 0.015,
    36: 0.015,
    37: 0.015,
    38: 0.015,
    39: 0.015,
    40: 0.015,
    41: 0.015,
    42: 0.015,
    43: 0.015,
    44: 0.015,
    45: 0.015,
    46: 0.015,
    47: 0.015,
    48: 0.015,
    49: 0.015,
}

RANGE_ANGLES_INTERVALS = {
    1: {1: (0, 0), 2: (239.2, 240.4), 3: (119.8, 120.4)},  # as reference for node 0 Phase 1 (always 0°)
    2: {1: (100, 360), 2: (180, 360), 3: (0, 360)},
    3: {1: (80, 340), 2: (0, 360), 3: (120, 260)},
    4: {1: (0, 300), 2: (0, 150), 3: (260, 360)},
    5: {1: (150, 180), 2: (300, 340), 3: (40, 70)},
    6: {1: (300, 380), 2: (340, 20), 3: (0, 230)},
    7: {1: (260, 285), 2: (130, 180), 3: (25, 50)},
    8: {1: (210, 310), 2: (0, 140), 3: (0, 180)},
    9: {1: (260, 335), 2: (280, 380), 3: (220, 340)},
    10: {1: (0, 320), 2: (0, 360), 3: (240, 440)},
    11: {1: (0, 180), 2: (0, 180), 3: (180, 320)},
    12: {1: (160, 360), 2: (350, 520), 3: (310, 520)},
    13: {1: (150, 210), 2: (30, 80), 3: (280, 330)},
    14: {1: (100, 390), 2: (100, 360), 3: (0, 360)},
    15: {1: (100, 180), 2: (200, 360), 3: (10, 210)},
    16: {1: (50, 360), 2: (260, 410), 3: (130, 370)},
    17: {1: (230, 310), 2: (260, 460), 3: (120, 200)},
    18: {1: (20, 360), 2: (290, 310), 3: (0, 360)},
    19: {1: (50, 100), 2: (350, 530), 3: (140, 250)},
    20: {1: (0, 360), 2: (140, 170), 3: (0, 360)},
    21: {1: (40, 180), 2: (0, 140), 3: (0, 150)},
    22: {1: (250, 380), 2: (270, 20), 3: (335, 410)},
    23: {1: (240, 440), 2: (350, 520), 3: (0, 360)},
    24: {1: (0, 1), 2: (0, 150), 3: (200, 360)},
    25: {1: (80, 180), 2: (350, 520), 3: (130, 300)},
    26: {1: (0, 0), 2: (0, 0), 3: (0, 0)}, # from here on, angles will not be measured reliably.
    27: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    28: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    29: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    30: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    31: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    32: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    33: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    34: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    35: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    36: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    37: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    38: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    39: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    40: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    41: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    42: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    43: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    44: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    45: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    46: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    47: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    48: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    49: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
}

RANGE_ANGLES_INTERVALS_DEFAULT_WR = {
    1: (0, 0),  # as reference for node 0 Phase 1 (always 0°)
    2: (0, 360),
    3: (0, 360),
    4: (0, 360),
    5: (0, 360),
    6: (0, 360),
    7: (0, 360),
    8: (0, 360),
    9: (0, 360),
    10: (0, 360),
    11: (0, 360),
    12: (0, 360),
    13: (0, 360),
    14: (0, 360),
    15: (0, 360),
    16: (0, 360),
    17: (0, 360),
    18: (0, 360),
    19: (0, 360),
    20: (0, 360),
    21: (0, 360),
    22: (0, 360),
    23: (0, 360),
    24: (0, 360),
    25: (0, 360),
    26: (0, 360), # from here on, angles will not be measured reliably.
    27: (0, 360),
    28: (0, 360),
    29: (0, 360),
    30: (0, 360),
    31: (0, 360),
    32: (0, 360),
    33: (0, 360),
    34: (0, 360),
    35: (0, 360),
    36: (0, 360),
    37: (0, 360),
    38: (0, 360),
    39: (0, 360),
    40: (0, 360),
    41: (0, 360),
    42: (0, 360),
    43: (0, 360),
    44: (0, 360),
    45: (0, 360),
    46: (0, 360),
    47: (0, 360),
    48: (0, 360),
    49: (0, 360),
}