import copy
import datetime
import enum
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Enum, TIMESTAMP
from sqlalchemy.orm import sessionmaker, relationship

from OpenDSSWrapper.tools.orm_models import Base

logger = logging.getLogger(__name__)


class PowerProfileCategory(enum.Enum):
    LOAD = 'LOAD'
    GENERATOR = 'GENERATOR'
    SOURCE = 'SOURCE'
    DEFAULT = 'DEFAULT'
    SAMPLED = 'SAMPLED'


class PowerProfileSubCategory(enum.Enum):
    RESIDENTIAL = 'RESIDENTIAL'
    COMMERCIAL = 'COMMERCIAL'
    INDUSTRIAL = 'INDUSTRIAL'
    OTHER_LOAD = 'OTHER_LOAD'
    PHOTOVOLTAIC_SYSTEM = 'PHOTOVOLTAIC_SYSTEM'
    WIND_TURBINE = 'WIND_TURBINE'
    HYDRO_TURBINE = 'HYDRO_TURBINE'
    OTHER_GENERATOR = 'OTHER_GENERATOR'
    EXTERNAL_GRID = 'EXTERNAL_GRID'
    DEFAULT = 'DEFAULT'


class PowerProfileConfig:
    def __init__(self, name: str, category: PowerProfileCategory, subcategory: PowerProfileSubCategory,
                 step_size_ms: int, time_step: int | list[int] = None, random_func: Callable = None,
                 noise_params: dict = None,
                 scaling_factor: float = 1., normalize=True, samples=None):
        """
        Configuration for a power profile
        :param name: name of the power profile
        :param category: instance of PowerProfileCategory
        :param subcategory: instance of PowerProfileSubCategory
        :param step_size_ms: desired step size in milliseconds
        :param time_step: time step or time steps to retrieve (integer or list of ints) if None, all time steps are retrieved
        :param random_func: A numpy.random function to generate noise. Expected to return an ndarray.
        :param noise_params: parameters for the random function (dictionary keys are the parameter names)
        :param scaling_factor: scaling factor for the data (e.g. to scale PV system data which is given in W/m²)
        :param normalize: normalize the data to the maximum value
        :param samples: samples if the profile is sampled from a previously defined distribution
        """
        self.name = name
        self.category = category
        self.subcategory = subcategory
        self.step_size_ms = step_size_ms
        self.time_step = time_step
        self.random_func = random_func
        self.noise_params = noise_params
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.samples = samples

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


def validate_datetime(begin_time_str, end_time_str):
    format_str = "%Y-%m-%d %H:%M:%S"  # Define the datetime format without milliseconds
    try:
        begin_time = datetime.datetime.strptime(begin_time_str, format_str)
        end_time = datetime.datetime.strptime(end_time_str, format_str)

        if begin_time >= end_time:
            logger.error(f"Validation error: begin_time {begin_time} must be before end_time {end_time}")
            raise ValueError("begin_time must be before end_time")
    except ValueError as e:
        # Handle incorrect format or logical error
        logger.error(f"Validation error: {e}")
        return False
    return True


class PowerProfile(Base):
    """
    ORM class for storing power profile descriptions
    """
    __tablename__ = 'power_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    category = Column(Enum(PowerProfileCategory), nullable=False)
    subcategory = Column(Enum(PowerProfileSubCategory), nullable=False)
    begin_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    step_size_ms = Column(Integer, nullable=False)
    date_added = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    source = Column(String(255))


class PowerProfileValue(Base):
    """
    ORM class for storing power profile values
    """
    __tablename__ = 'power_profile_values'

    profile_id = Column(Integer, ForeignKey('power_profiles.id'), primary_key=True)
    time_step = Column(Integer, primary_key=True)
    PL1 = Column(Float)
    PL2 = Column(Float)
    PL3 = Column(Float)
    QL1 = Column(Float)
    QL2 = Column(Float)
    QL3 = Column(Float)
    profile = relationship("PowerProfile", back_populates="values")


PowerProfile.values = relationship("PowerProfileValue", order_by=PowerProfileValue.time_step, back_populates="profile")


class PowerProfileStore:
    """
    Base class for storing power profile data in a database using SQLAlchemy.
    """
    DEFAULT_PROFILE = pd.DataFrame({
        'profile_id': [-1, -1],
        'pl1': [1, 1],
        'pl2': [1, 1],
        'pl3': [1, 1],
        'ql1': [0, 0],
        'ql2': [0, 0],
        'ql3': [0, 0],
        'time_step': [0, 1]
    })
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_profile(self, name, category, subcategory, begin_time, end_time, step_size_ms, source):
        session = self.Session()
        if not validate_datetime(begin_time, end_time):
            logger.error(
                f"Invalid datetime format, power profile with name {name} not inserted into database.")
            return
        try:
            profile = PowerProfile(
                name=name,
                category=category,
                subcategory=subcategory,
                begin_time=begin_time,
                end_time=end_time,
                step_size_ms=step_size_ms,
                source=source
            )
            session.add(profile)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"An error occurred: {e}")
        finally:
            session.close()

    def _get_profile_id(self, name: str):
        """
        Get the profile ID for a given profile name
        :param name:
        :return:
        """
        session = self.Session()
        try:
            profile = session.query(PowerProfile.id).filter_by(name=name).first()
            return profile.id if profile else None
        finally:
            session.close()

    def insert_profile_value(self, name: str, pl1: float, pl2: float, pl3: float, ql1: float, ql2: float, ql3: float,
                             time_step: int):
        """
        Insert a single power profile value into the database
        :param name:
        :param pl1: Active power phase 1
        :param pl2:
        :param pl3:
        :param ql1: Reactive power phase 1
        :param ql2:
        :param ql3:
        :param time_step: time step as integer, order must be ensured by the user
        :return:
        """
        session = self.Session()
        try:
            profile_id = self._get_profile_id(name)
            if profile_id:
                profile_value = PowerProfileValue(
                    profile_id=profile_id,
                    time_step=time_step,
                    PL1=pl1,
                    PL2=pl2,
                    PL3=pl3,
                    QL1=ql1,
                    QL2=ql2,
                    QL3=ql3
                )
                session.add(profile_value)
                session.commit()
            else:
                logger.error(f"Profile ID {name} not found.")
        except Exception as e:
            session.rollback()
            logger.error(f"An error occurred: {e}")
        finally:
            session.close()

    def insert_profile_values(self, name: str, values: list[tuple]):
        """
        bulk insert power profile values into the database
        :param name: name of the profile
        :param values: Power profile values as list of tuples (PL1, PL2, PL3, QL1, QL2, QL3, time_step)
        :return:
        """
        session = self.Session()
        try:
            profile_id = self._get_profile_id(name)
            if profile_id:
                profile_values = [
                    PowerProfileValue(
                        profile_id=profile_id,
                        time_step=value[6],
                        PL1=value[0],
                        PL2=value[1],
                        PL3=value[2],
                        QL1=value[3],
                        QL2=value[4],
                        QL3=value[5]
                    ) for value in values
                ]
                session.bulk_save_objects(profile_values)
                session.commit()
            else:
                logger.error(f"Profile ID {name} not found.")
        except Exception as e:
            session.rollback()
            logger.error(f"An error occurred: {e}")
        finally:
            session.close()

    def retrieve_profile_values(self, power_profile_config: PowerProfileConfig) -> pd.DataFrame:
        """
        Retrieve power profile values from the database
        :param power_profile_config: Instance of PowerProfileConfig
        :return: processed DataFrame with columns profile_id, pl1, pl2, pl3, ql1, ql2, ql3, time_step
        """
        if power_profile_config.category == PowerProfileCategory.SAMPLED:
            if power_profile_config.samples is None:
                raise ValueError("No samples provided for sampled profile.")
            # symmetric profile with Q = 0 and P = samples
            df = pd.DataFrame({
                'profile_id': [-1] * len(power_profile_config.samples),
                'pl1': power_profile_config.samples,
                'pl2': power_profile_config.samples,
                'pl3': power_profile_config.samples,
                'ql1': [0] * len(power_profile_config.samples),
                'ql2': [0] * len(power_profile_config.samples),
                'ql3': [0] * len(power_profile_config.samples),
                'time_step': range(len(power_profile_config.samples))
            })
            return PowerProfileStore._process_data_frame(df, power_profile_config.step_size_ms, power_profile_config)
        if (power_profile_config.category == PowerProfileCategory.DEFAULT and
                power_profile_config.subcategory == PowerProfileSubCategory.DEFAULT):
            # Default category and subcategory: return a profile with P=1, Q=0 for all phases that will be interpolated
            step_size = 1000 * 60 * 60 * 24 * 365  # 1 year in milliseconds
            df = PowerProfileStore.DEFAULT_PROFILE
            return PowerProfileStore._process_data_frame(df, step_size, power_profile_config)
        session = self.Session()
        try:
            profile = self.find_profile(power_profile_config, session)
            if not profile:
                raise ValueError("No profile found for the given parameters.")

            # Select only the columns needed directly from the PowerProfileValue model
            query = session.query(
                PowerProfileValue.profile_id,
                PowerProfileValue.PL1,
                PowerProfileValue.PL2,
                PowerProfileValue.PL3,
                PowerProfileValue.QL1,
                PowerProfileValue.QL2,
                PowerProfileValue.QL3,
                PowerProfileValue.time_step
            ).filter_by(profile_id=profile.id)

            # Additional filtering based on time step
            if power_profile_config.time_step is not None:
                if isinstance(power_profile_config.time_step, list):
                    query = query.filter(PowerProfileValue.time_step.in_(power_profile_config.time_step))
                else:
                    query = query.filter_by(time_step=power_profile_config.time_step)

            # Execute the query and directly create DataFrame
            result = query.all()
            df = pd.DataFrame(result, columns=['profile_id', 'pl1', 'pl2', 'pl3', 'ql1', 'ql2', 'ql3', 'time_step'])

            step_size = profile.step_size_ms
            return PowerProfileStore._process_data_frame(df, step_size, power_profile_config)

        finally:
            session.close()

    def find_profile(self, power_profile_config: PowerProfileConfig, session):
        """
        Find a power profile based on the given configuration
        :param power_profile_config:
        :param session:
        :return:
        """
        profile = None
        if power_profile_config.name is not None:
            profile = session.query(PowerProfile).filter_by(name=power_profile_config.name).first()
        if profile is None:
            profiles = session.query(PowerProfile).filter_by(
                category=power_profile_config.category,
                subcategory=power_profile_config.subcategory
            ).all()
            if profiles:
                profile = profiles[np.random.randint(0, len(profiles))]
        if profile is None:
            profiles = session.query(PowerProfile).filter_by(category=power_profile_config.category).all()
            if profiles:
                profile = profiles[np.random.randint(0, len(profiles))]
        return profile

    @staticmethod
    def _process_data_frame(df: pd.DataFrame, step_size: int, power_profile_config: PowerProfileConfig) -> pd.DataFrame:
        """
        Process the data frame retrieved from the database
        - Interpolate the data if the step size does not match the desired step size
        - Add random noise if a random number generator is provided
        - Scale the data if a scaling factor is provided
        - Normalize the data if the normalize flag is set
        :param df: pandas DataFrame with columns: profile_id, pl1, pl2, pl3, ql1, ql2, ql3, time_step
        :param step_size: profile step size in milliseconds
        :param power_profile_config: instance of PowerProfileConfig
        :return: processed DataFrame with columns profile_id, pl1, pl2, pl3, ql1, ql2, ql3, time_step
        """
        if step_size != power_profile_config.step_size_ms:
            # interpolate the data to the desired step size
            df = df.set_index('time_step')
            # step size is an integer value counting up; convert to milliseconds
            df.index = pd.to_timedelta(df.index * step_size, unit='ms')
            if len(df) < 3:  # not enough data for polynomial interpolation
                logger.info(
                    f"Profile id {df['profile_id'][0]} has less than 3 data points. Using linear interpolation.")
                df = df.resample(str(power_profile_config.step_size_ms) + 'ms').interpolate(method='linear')
            else:
                df = df.resample(str(power_profile_config.step_size_ms) + 'ms').interpolate(method='polynomial',
                                                                                            order=2)
            df.reset_index(inplace=True)
            df.drop(columns=['time_step'], inplace=True)  # time step is equal to index and no longer required
        # add random noise
        if power_profile_config.random_func is not None:
            logger.debug(f"Adding random noise to profile {df['profile_id'][0]} with parameters "
                         f"{power_profile_config.noise_params}")
            # evenly across all columns
            noise = power_profile_config.random_func(size=df.iloc[:, 1:7].shape, **power_profile_config.noise_params)
            df.iloc[:, 1:7] = df.iloc[:, 1:7] * noise
        # scale the data
        df.iloc[:, 1:7] = df.iloc[:, 1:7] * power_profile_config.scaling_factor
        if power_profile_config.normalize:
            logger.debug(f"Normalizing profile {df['profile_id'][0]}")
            # normalize the data to the maximum value
            df.iloc[:, 1:7] = df.iloc[:, 1:7] / df.iloc[:, 1:7].max()
        return df


class PowerProfileStoreSQLite(PowerProfileStore):
    def __init__(self, db_file):
        if not os.path.exists(db_file):
            raise ValueError("Provided path does not exist.")
        connection_string = f"sqlite:///{db_file}"
        super().__init__(connection_string)


class PowerProfileStorePostgres(PowerProfileStore):
    def __init__(self, db_config):
        connection_string = (f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:"
                             f"{db_config['port']}/{db_config['database']}")
        super().__init__(connection_string)


def migrate_data(sqlite_conn_str, postgres_conn_str):
    """
    Migrate data from SQLite to PostgreSQL
    :param sqlite_conn_str:
    :param postgres_conn_str:
    :return:
    """
    # Create engines for both databases
    sqlite_engine = create_engine(sqlite_conn_str)
    postgres_engine = create_engine(postgres_conn_str)

    # Create session factories for both databases
    SQLiteSession = sessionmaker(bind=sqlite_engine)
    PostgresSession = sessionmaker(bind=postgres_engine)

    # Create a SQLite session
    sqlite_session = SQLiteSession()

    # Create a PostgreSQL session
    postgres_session = PostgresSession()

    try:
        # Query all PowerProfiles from SQLite
        profiles = sqlite_session.query(PowerProfile).all()

        for profile in profiles:
            # Create a new PowerProfile instance for PostgreSQL
            new_profile = PowerProfile(
                name=profile.name,
                category=profile.category,
                subcategory=profile.subcategory,
                begin_time=profile.begin_time,
                end_time=profile.end_time,
                step_size_ms=profile.step_size_ms,
                source=profile.source
            )

            # Add the new profile to the PostgreSQL session
            postgres_session.add(new_profile)
            postgres_session.commit()

            # Get the newly created profile ID
            new_profile_id = new_profile.id

            # Query related PowerProfileValues from SQLite
            values = sqlite_session.query(PowerProfileValue).filter_by(profile_id=profile.id).all()

            for value in values:
                # Create a new PowerProfileValue instance for PostgreSQL
                new_value = PowerProfileValue(
                    profile_id=new_profile_id,
                    time_step=value.time_step,
                    PL1=value.PL1,
                    PL2=value.PL2,
                    PL3=value.PL3,
                    QL1=value.QL1,
                    QL2=value.QL2,
                    QL3=value.QL3
                )

                # Add the new value to the PostgreSQL session
                postgres_session.add(new_value)

            # Commit all values for the current profile
            postgres_session.commit()

    except Exception as e:
        postgres_session.rollback()
        logger.error(f"An error occurred: {e}")
    finally:
        sqlite_session.close()
        postgres_session.close()


if __name__ == '__main__':
    from OpenDSSWrapper.tools.config import postgres_config, power_profile_db_name

    sqlite_path = os.path.join(os.path.dirname(__file__), '../../../../power-profile-generator/src', '..', 'data',
                               power_profile_db_name)
    # Initialize SQLite
    sqlite_connection_string = "sqlite:///" + str(sqlite_path)

    # Initialize PostgreSQL
    postgres_connection_string = (f"postgresql+psycopg2://{postgres_config['user']}:{postgres_config['password']}@"
                                  f"{postgres_config['host']}:{postgres_config['port']}/{postgres_config['dbname']}")
    migrate_data(sqlite_connection_string, postgres_connection_string)
