import copy
import json
import logging
import os
import random
from datetime import datetime
from enum import Enum

import numpy as np
from sqlalchemy import create_engine, Column, Integer, Text, Float, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.types import TypeDecorator

from OpenDSSWrapper.harmonic_spectra import Spectrum, LoadVaryingSpectrum, DynamicSpectrum, TimeSeriesSpectrum, \
    RandomSpectrum
from OpenDSSWrapper.tools.orm_models import Base

logger = logging.getLogger(__name__)


class DynamicSpectrumType(Enum):
    """
    Type of dynamic spectrum
    """
    TIME_SERIES = 1
    LOAD_VARYING = 2
    RANDOM = 3


class SpectrumCategory(Enum):
    """
    Extension of panda Categories
    """
    LIGHTING = 1
    COMPUTER_AND_COMMUNICATION = 2
    ENTERTAINMENT = 3
    OTHER_APPLIANCES = 4
    GENERATION = 5
    ELECTRIC_VEHICLES = 6
    # extensions
    DEFAULT = 7
    INDUSTRIAL = 8
    RESIDENTIAL = 9
    COMMERCIAL = 10


class SpectrumSubCategory(Enum):
    # Lighting Subcategories
    INCANDESCENT_FILAMENT_LAMP = (1, 1)
    COMPACT_FLUORESCENT_LAMP = (1, 2)
    FLUORESCENT_LAMP_ELECTRONIC = (1, 3)
    FLUORESCENT_LAMP_NON_ELECTRONIC = (1, 4)
    INTEGRATED_SOLID_STATE_LAMP = (1, 5)
    NON_INTEGRATED_SOLID_STATE_LAMP = (1, 6)
    LIGHTING_OTHER = (1, 7)

    # Computer and Communication Subcategories
    DESKTOP = (2, 1)
    LAPTOP = (2, 2)
    MOBILE_DEVICE = (2, 3)
    MONITOR = (2, 4)
    PRINTER_SCANNER_MULTIFUNCTIONAL = (2, 5)
    NETWORK_EQUIPMENT = (2, 6)
    STORAGE = (2, 7)
    PERMANENT_CONNECTED_PHONES = (2, 8)
    COMPUTER_OTHER_CHARGING = (2, 9)
    COMPUTER_OTHER_NONCHARGING = (2, 10)

    # Entertainment Subcategories
    CRT_TELEVISION_SET = (3, 1)
    FLAT_PANEL_TELEVISION_SET = (3, 2)
    PROJECTORS = (3, 3)
    AUDIO_VIDEO_PHOTO_CHARGING = (3, 4)
    AUDIO_VIDEO_PHOTO_NONCHARGING = (3, 5)
    GAME_CONSOLE_CHARGING = (3, 6)
    GAME_CONSOLE_NONCHARGING = (3, 7)
    ENTERTAINMENT_OTHER_CHARGING = (3, 8)
    ENTERTAINMENT_OTHER_NONCHARGING = (3, 9)

    # Other Appliances Subcategories
    KITCHEN = (4, 1)
    LAUNDRY = (4, 2)
    BATHROOM = (4, 3)
    OFFICE = (4, 4)
    TOOLS_AND_GARDEN = (4, 5)
    BATTERY_CHARGERS = (4, 6)
    OTHER = (4, 7)

    # Generation Subcategories
    PHOTOVOLTAIC_SYSTEM = (5, 1)
    MICRO_CHP = (5, 2)
    SMALL_WIND_ELECTRIC_SYSTEM = (5, 3)
    GENERATION_OTHER = (5, 4)
    GRID = (5, 5)

    # Electric Vehicles Subcategories
    ELECTRIC_CAR = (6, 1)
    ELECTRIC_BIKE = (6, 2)
    ELECTRIC_MOTORCYCLE_AND_SCOOTERS = (6, 3)
    CHARGING_STATION_WITH_INTEGRATED_RECTIFIER = (6, 4)
    ELECTRIC_VEHICLES_OTHER = (6, 5)

    # extend with additional subcategories
    DEFAULT = (7, 0)

    # Industrial
    VARIABLE_FREQUENCY_DRIVE = (8, 0)
    VARIABLE_SPEED_DRIVE = (8, 1)
    INDUSTRIAL_OTHER = (8, 2)

    # Residential
    RESIDENTIAL_OTHER = (9, 0)
    RESIDENTIAL_APARTMENT = (9, 1)
    RESIDENTIAL_SINGLE_FAMILY = (9, 2)
    RESIDENTIAL_MULTI_FAMILY = (9, 3)

    # Commercial
    COMMERCIAL_OTHER = (10, 0)
    COMMERCIAL_OFFICE = (10, 1)
    COMMERCIAL_RETAIL = (10, 2)
    COMMERCIAL_HOTEL = (10, 3)
    COMMERCIAL_RESTAURANT = (10, 4)


class SupplySource(Enum):
    """
    Supply source used during measurement
    """
    GRID = 1
    PROGRAMMABLE_GENERATOR_SINUSOIDAL = 2
    PROGRAMMABLE_GENERATOR_DISTORTED = 3


class AggregationType(Enum):
    """
    Aggregation type for the measurement values
    """
    ACTUAL = 1  # actual value for each time step
    MAX = 2  # maximum value over the given time period
    AVERAGE = 3  # average value over the given time period


class MeasurementType(Enum):
    """
    Type of the measurement values
    """
    CURRENT = 1
    VOLTAGE = 2


class AccessLevel(Enum):
    """
    Who may access the spectrum?
    """
    PRIVATE = 0
    PARTNER = 1
    PUBLIC = 2


class JSONEncodedArray(TypeDecorator):
    """
    Encode arrays to JSON strings in the database
    since only PostgreSQL supports native array types
    """
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Convert NumPy types to native Python types
        if isinstance(value, (list, tuple)):
            value = [int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (
                np.floating, np.float64)) else v for v in value]

        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)


class Spectra(Base):
    """
    ORM class for the spectra table
    """
    __tablename__ = 'spectra'
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(Text)
    subcategory = Column(Text)
    manufacturer = Column(Text)
    name = Column(Text)
    measurement_type = Column(Text)
    aggregation_type = Column(Text)
    supply_source = Column(Text)
    nominal_voltage = Column(Integer)
    nominal_frequency = Column(Integer)
    rated_power = Column(Float)
    time_step_ms = Column(Integer)
    access_level = Column(Integer)
    source = Column(Text)
    description = Column(Text)
    date_added = Column(TIMESTAMP, server_default='CURRENT_TIMESTAMP')

    spectra_values = relationship('SpectraValues', back_populates='spectra')
    waveform_values = relationship('WaveformValues', back_populates='spectra')


class SpectraValues(Base):
    """
    ORM class for the spectra values table
    """
    __tablename__ = 'spectra_values'
    id = Column(Integer, primary_key=True, autoincrement=True)
    spectra_id = Column(Integer, ForeignKey('spectra.id'))
    phase = Column(Integer)
    time_step = Column(Integer)
    power = Column(Float)
    harmonics = Column(JSONEncodedArray)
    magnitudes = Column(JSONEncodedArray)
    angles = Column(JSONEncodedArray)

    spectra = relationship('Spectra', back_populates='spectra_values')


class WaveformValues(Base):
    """
    ORM class for the waveform values table
    """
    __tablename__ = 'waveform_values'
    id = Column(Integer, primary_key=True, autoincrement=True)
    spectra_id = Column(Integer, ForeignKey('spectra.id'))
    phase = Column(Integer)
    power = Column(Float)
    sample_rate_ns = Column(Integer)
    values = Column(JSONEncodedArray)

    spectra = relationship('Spectra', back_populates='waveform_values')


class SpectrumConfig:
    def __init__(self, name: str = "IdealLoad", manufacturer: str = None,
                 category: SpectrumCategory | str = SpectrumCategory.DEFAULT,
                 subcategory: SpectrumSubCategory | str = SpectrumSubCategory.DEFAULT,
                 phase: int = None, measurement_type: MeasurementType = None, nominal_voltage: int = None,
                 nominal_frequency: int = None, dynamic_type=DynamicSpectrumType.LOAD_VARYING, **kwargs):
        """
        Configuration for a spectrum
        :param name: name of the device
        :param manufacturer: manufacturer of the device, default None (any)
        :param category: device category
        :param subcategory: device subcategory
        :param phase: phase [1,2,3] or None (select any phase)
        :param measurement_type: current or voltage or None (select any measurement type)
        :param nominal_voltage: nominal voltage in Volt or None (select any nominal voltage)
        :param nominal_frequency: nominal frequency in Hz or None (select any nominal frequency)
        :param dynamic_type: type of dynamic spectrum, default LOAD_VARYING
        :param kwargs: keyword arguments especially for random spectra
        """
        if isinstance(category, str):
            if category not in SpectrumCategory.__members__:
                raise ValueError(f"Category {category} not in SpectrumCategory Enum")
            category = SpectrumCategory[category]
        if isinstance(subcategory, str):
            if subcategory not in SpectrumSubCategory.__members__:
                raise ValueError(f"Subcategory {subcategory} not in SpectrumSubCategory Enum")
            subcategory = SpectrumSubCategory[subcategory]
        self.name = name
        self.manufacturer = manufacturer
        self.category = category
        self.subcategory = subcategory
        self.phase = phase
        self.spectrum_type = measurement_type
        self.nominal_voltage = nominal_voltage
        self.nominal_frequency = nominal_frequency
        self.dynamic_type = dynamic_type
        self.kwargs = kwargs
        # add checks for phase and type
        if phase is not None and phase not in [1, 2, 3]:
            raise ValueError(f"Phase {phase} not in [1,2,3]")

    def clone(self, **attrs):
        """
        Clone the current instance and update with given attributes.
        Prototype pattern implementation.
        :param attrs: attributes to update
        :return: new instance of PowerProfileConfig
        """
        obj = copy.deepcopy(self)
        obj.__dict__.update(attrs)
        return obj


class SpectraStore:
    def __init__(self, database_url):
        self.database_url = database_url
        self.session = self.init_db(database_url)

    def init_db(self, database_url):
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def add_spectrum(self, name, manufacturer, category, subcategory, measurement_type=MeasurementType.CURRENT,
                     aggregation_type=AggregationType.ACTUAL, supply_source=SupplySource.GRID,
                     nominal_voltage=230, nominal_frequency=50, rated_power=0,
                     time_step_ms=0, access_level=AccessLevel.PRIVATE, description="",
                     source="Measurement", date_added=None):
        """
        Add a spectrum to the database.
        Either nominal_power or nominal_current and nominal_voltage must be specified.
        :param name: spectrum name as a unique identifier
        :param manufacturer: manufacturer of the device
        :param category: Device category, one of SpectrumCategory Enum
        :param subcategory: Device subcategory, one of SpectrumSubCategory Enum
        :param measurement_type: type of spectrum, current or voltage
        :param aggregation_type: aggregation type of the measurement values
        :param supply_source: supply source of the spectrum
        :param nominal_voltage: nominal voltage in Volt (default 230V)
        :param nominal_frequency: nominal frequency in Hz (default 50Hz)
        :param rated_power: nominal power in Watt, per phase (default 0 - unknown)
        :param time_step_ms: time step in milliseconds (default 0 - only one value)
        :param access_level: access level for the spectrum (default private)
        :param description: description of the spectrum
        :param source: source of the spectrum (e.g. measurement location and date)
        :param date_added: date when the spectrum was added to the database; default is the current date
        :return:
        """
        if date_added is None:
            date_added = datetime.now()
        new_spectra = Spectra(name=name, manufacturer=manufacturer,
                              category=category.name, subcategory=subcategory.name,
                              measurement_type=measurement_type.name, aggregation_type=aggregation_type.name,
                              supply_source=supply_source.name, nominal_voltage=nominal_voltage,
                              nominal_frequency=nominal_frequency, rated_power=rated_power,
                              time_step_ms=time_step_ms, access_level=access_level.value,
                              source=source, description=description, date_added=date_added
                              )
        self.session.add(new_spectra)
        self.session.commit()
        return new_spectra.id

    def add_spectrum_values(self, spectra_id, harmonics, magnitudes, angles, phase=-1, power=0, time_step=0):
        """
        Add spectrum values to the database.
        :param spectra_id: id of the spectrum in the database
        :param phase: phase for which the spectrum was recorded, -1 for all phases / unknown phase (default: -1)
        :param power: apparent power in W at the current time step.
        For 3-phase devices, chose the power of one phase.
        If unknown, choose the rated power, default 0.
        :param harmonics: list of harmonic orders
        :param magnitudes: list of magnitudes for the harmonics.
        The magnitudes are relative to the fundamental frequency. Therefore, the first harmonic will always be 100%.
        :param angles: list of angles for the harmonics
        :param time_step: time step of the recorded spectrum, 0 if only one value exists
        :return:
        """
        new_spectra_values = SpectraValues(
            spectra_id=spectra_id, phase=phase, time_step=time_step, power=power,
            harmonics=harmonics, magnitudes=magnitudes, angles=angles
        )
        self.session.add(new_spectra_values)
        self.session.commit()

    def add_waveform_values(self, spectra_id: int, phase: int, power: float, sample_rate_ns: int, values: list[float]):
        """
        Add waveform values to the database.
        :param spectra_id: id of the spectrum in the database
        :param phase: phase of the waveform data, -1 for all phases / unknown phase (default: -1)
        :param power: power in W of the recorded waveform
        :param sample_rate_ns: sample rate in nanoseconds
        :param values: array of float values for the waveform at the given sample rate
        :return:
        """
        new_waveform_values = WaveformValues(
            spectra_id=spectra_id, phase=phase, power=power, sample_rate_ns=sample_rate_ns, values=values
        )
        self.session.add(new_waveform_values)
        self.session.commit()

    def get_spectra(self, spectrum_config: SpectrumConfig) -> DynamicSpectrum:
        """
        Hierarchical search for spectra in the database
        Attention: If a category is not in the database, random spectra from the same subcategory is returned.
        :param spectrum_config: configuration for the spectrum
        :return: DynamicSpectrum object for the device or the closest match
        """
        if spectrum_config.dynamic_type == DynamicSpectrumType.RANDOM:
            import OpenDSSWrapper.tools.config as cfg
            max_harm = spectrum_config.kwargs.pop('max_harmonic_values', cfg.MAX_HARMONIC_VALUES)
            interval_angles_3ph = spectrum_config.kwargs.pop('max_angle_values', cfg.RANGE_ANGLES_INTERVALS)
            interval_angles = {}
            if spectrum_config.phase:
                phase = spectrum_config.phase
                for k in interval_angles_3ph.keys():
                    interval_angles[k] = interval_angles_3ph[k][phase]
            else:
                interval_angles = spectrum_config.kwargs.pop('max_angle_values_wr_default',
                                                             cfg.RANGE_ANGLES_INTERVALS_DEFAULT_WR)
            if 'ignore_angles' in spectrum_config.kwargs.keys():
                if spectrum_config.kwargs['ignore_angles']:
                    interval_angles = None

            random_func = spectrum_config.kwargs.pop('random_func', np.random.random)
            noise_params = spectrum_config.kwargs.pop('noise_params', {})
            return RandomSpectrum(max_harm, random_func, noise_params, range_harmonic_angles=interval_angles)
        conditions = []
        if spectrum_config.spectrum_type:
            conditions.append(Spectra.measurement_type == spectrum_config.spectrum_type.name)
        if spectrum_config.nominal_frequency:
            conditions.append(Spectra.nominal_frequency == spectrum_config.nominal_frequency)
        if spectrum_config.nominal_voltage:
            conditions.append(Spectra.nominal_voltage == spectrum_config.nominal_voltage)

        base_query = self.session.query(Spectra, SpectraValues).join(SpectraValues)

        # Hierarchical search with checks for None
        search_params = [
            ([
                Spectra.name == spectrum_config.name if spectrum_config.name else True,
                Spectra.manufacturer == spectrum_config.manufacturer if spectrum_config.manufacturer else True,
                Spectra.subcategory == spectrum_config.subcategory.name,
                Spectra.category == spectrum_config.category.name
            ]),
            ([
                Spectra.manufacturer == spectrum_config.manufacturer if spectrum_config.manufacturer else True,
                Spectra.subcategory == spectrum_config.subcategory.name,
                Spectra.category == spectrum_config.category.name
            ]),
            ([
                Spectra.subcategory == spectrum_config.subcategory.name,
                Spectra.category == spectrum_config.category.name
            ]),
            ([
                Spectra.category == spectrum_config.category.name
            ])
        ]

        results = None
        for params in search_params:
            query = base_query.filter(*[param for param in params if param is not True])
            if conditions:
                query = query.filter(*conditions)
            results = query.all()
            if results:
                # Group results by spectrum ID
                spectrum_groups = {}
                for spectra, spectra_values in results:
                    if spectra.id not in spectrum_groups:
                        spectrum_groups[spectra.id] = []
                    spectrum_groups[spectra.id].append((spectra, spectra_values))

                # Randomly select one spectrum ID
                random_spectrum_id = random.choice(list(spectrum_groups.keys()))
                results = spectrum_groups[random_spectrum_id]
                break

        return self._process_results(spectrum_config, results)

    @staticmethod
    def _process_results(spectrum_config: SpectrumConfig, results) -> DynamicSpectrum:
        """
        Create Dynamic spectrum object from database query results
        :param spectrum_config: Spectrum Config
        :param results: query results
        :return: DynamicSpectrum object
        """
        phase, filtered_results = SpectraStore._filter_phase(results, spectrum_config.phase)
        # Extract the rated_power from the first result's spectra
        _, first_spectra_values = filtered_results[0]
        rated_power = first_spectra_values.spectra.rated_power

        if spectrum_config.dynamic_type == DynamicSpectrumType.TIME_SERIES:
            time_step_ms = first_spectra_values.spectra.time_step_ms
            # sort by time_step
            filtered_results = sorted(filtered_results, key=lambda x: x[1].time_step)
            dynamic_spectrum = TimeSeriesSpectrum(rated_power=rated_power, phase=phase, time_step_ms=time_step_ms)
        else:  # Load-varying spectrum
            dynamic_spectrum = LoadVaryingSpectrum(rated_power=rated_power, phase=phase)

        for spectra, spectra_values in filtered_results:
            dynamic_spectrum.add_spectrum(
                Spectrum(spectra_values.harmonics, spectra_values.magnitudes, spectra_values.angles),
                load_power_watt=spectra_values.power, time_step=None
            )
        return dynamic_spectrum

    @staticmethod
    def _filter_phase(results, phase, phase_idx=1):
        """
        Filter the results for the desired phase. If the desired phase is not in the results, choose a random phase.
        :param results: fetched results from the database, a list of (spectra, spectra_values) tuples
        :param phase: desired phase
        :return: chosen phase, filtered results
        """
        # Extract unique phases from spectra_values
        unique_phases = set(row[phase_idx].phase for row in results)
        if phase in unique_phases:
            chosen_phase = phase
        else:
            # Select a random phase if the desired phase is not present
            chosen_phase = random.choice(list(unique_phases))

        # Filter results to only include the chosen phase
        filtered_results = [row for row in results if row[phase_idx].phase == chosen_phase]
        return chosen_phase, filtered_results


class SpectraStoreSQLite(SpectraStore):
    def __init__(self, db_file):
        if not os.path.exists(db_file):
            raise ValueError("Provided path does not exist.")
        connection_string = f"sqlite:///{db_file}"
        super().__init__(connection_string)


class SpectraStorePostgres(SpectraStore):
    def __init__(self, db_config):
        connection_string = (f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:"
                             f"{db_config['port']}/{db_config['database']}")
        super().__init__(connection_string)
