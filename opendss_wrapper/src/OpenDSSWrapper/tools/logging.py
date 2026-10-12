import logging
import os
from datetime import datetime
from enum import Enum
from logging.config import dictConfig
from OpenDSSWrapper.tools.config import DEBUG_MODE, DEBUG_PLOT

import coloredlogs


class DebugMode(Enum):
    """
    Debug mode for OpenDSS Simulation on a step by step basis
    If verbose Log messages, monitor data and simulation properties to a file in step directory
    NONE: No debug messages saved step by step
    VERBOSE: All properties, monitor data and messages saved to file
    """
    NONE = 0
    VERBOSE = 1


SIM_DEBUG_MODE = DebugMode.VERBOSE if DEBUG_MODE == 'VERBOSE' else DebugMode.NONE
SIM_DEBUG_PLOT = True if DEBUG_PLOT == 'True' else False


def setup_logging(log_path=None):
    """
    Setup logging configuration
    :param log_path: path to the log directory (if None, logs are not saved to file)
    :return:
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }
    if log_path is not None:
        # Ensure the logs directory exists
        os.makedirs(log_path, exist_ok=True)

        # Create a unique log file for each simulation
        log_filename = f'simulation_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
        log_file = os.path.join(log_path, log_filename)
        LOGGING_CONFIG['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'filename': log_file,
            'level': 'INFO',
            'formatter': 'standard',
        }
        LOGGING_CONFIG['loggers']['']['handlers'].append('file')

    # Apply logging configuration
    dictConfig(LOGGING_CONFIG)

    # Setup colored logs for console
    coloredlogs.install(level='INFO', logger=logging.getLogger(''),
                        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
