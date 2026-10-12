import csv
import logging
import os
from typing import Sequence, Callable
import random

import numpy as np
from scipy import interpolate
import OpenDSSWrapper.tools.config as cfg
logger = logging.getLogger(__name__)


def find_key_and_value_with_epsilon(target_value, dictionary, epsilon=1e-6):
    """
    Check if a key is in a dictionary with a given epsilon
    :param target_value: target value
    :param dictionary: dictionary to check
    :param epsilon: allowed deviation from target value
    :return: key, value pair if found, else None
    """
    for key, value in dictionary.items():
        if abs(key - target_value) < epsilon:
            return key, value
    return None


class Spectrum:
    def __init__(self, harmonics: Sequence[int | float], magnitudes: Sequence[float], angles: Sequence[float],
                 name: str = None):
        """
        Spectrum class to store harmonics, magnitudes and angles of a spectrum.
        Validates input shapes and values.
        :param harmonics: List of harmonics as float or int
        :param magnitudes: Magnitudes for each harmonic
        :param angles: Angles for each harmonic
        :param name: Name of the spectrum (optional)
        :return:
        """
        self.harmonics = harmonics
        self.magnitudes = magnitudes
        self.angles = angles
        self.name = name
        self.validate_spectrum()
        self.number_of_harmonics = len(self.harmonics)

    def validate_spectrum(self):
        """
        Validate spectrum
        Length of harmonics, magnitudes and angles must be equal.
        Length of harmonics, magnitudes and angles must be greater than 0.
        Harmonics must be unique.
        :return:
        """
        if len(self.harmonics) != len(self.magnitudes) or len(self.harmonics) != len(self.angles):
            raise Exception("Spectrum is not valid. Length of harmonics, magnitudes and angles must be equal.")
        if len(self.harmonics) == 0:
            raise Exception("Spectrum is not valid. Length of harmonics, magnitudes and angles must be greater than 0.")
        if len(self.harmonics) != len(set(self.harmonics)):
            raise Exception("Spectrum is not valid. Harmonics must be unique.")

    def get_opendss_string(self, name: str = None):
        """
        Get spectrum as string in OpenDSS format
        :param name: Name of the spectrum, if no name is given, the name of the spectrum given in initialization is used.
        :return: String in OpenDSS format
        """
        if name is None:
            name = self.name
        if name is None:
            raise ValueError("No name given for spectrum.")
        spectrum_string = f"New Spectrum.{name} NumHarm={self.number_of_harmonics} Harmonic=["
        for harmonic in self.harmonics:
            spectrum_string += f"{harmonic},"
        # remove trailing comma
        spectrum_string = spectrum_string[:-1] + "] "
        spectrum_string += "%mag=["
        for magnitude in self.magnitudes:
            spectrum_string += f"{magnitude:.6f},"
        # remove trailing comma
        spectrum_string = spectrum_string[:-1] + "] "
        spectrum_string += "ang=["
        for angle in self.angles:
            spectrum_string += f"{angle:.6f},"
        # remove trailing comma
        spectrum_string = spectrum_string[:-1] + "]"
        return spectrum_string

    @classmethod
    def from_csv(cls, path, file_name, header=False):
        """
        Read spectrum from csv file, first column is harmonics, second column is magnitudes, third column is angles
        :param path: path to file
        :param file_name: file name
        :param header: True if file contains a header
        :return: Spectrum object
        """
        harmonics = []
        magnitudes = []
        angles = []
        with open(os.path.join(path, file_name), 'r') as f:
            reader = csv.reader(f)
            if header:
                next(reader)
            for row in reader:
                harmonics.append(float(row[0]))
                magnitudes.append(float(row[1]))
                angles.append(float(row[2]))
        return cls(harmonics, magnitudes, angles, file_name.replace(".csv", ""))

    def to_csv(self, path, file_name, header=False):
        """
        Write spectrum to csv file, first column is harmonics, second column is magnitudes, third column is angles
        :param path: path to file
        :param file_name: file name
        :param header: True if file contains a header
        :return:
        """
        with open(os.path.join(path, file_name), 'w') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(['Harmonic', 'Magnitude', 'Angle'])
            for i in range(self.number_of_harmonics):
                writer.writerow([self.harmonics[i], self.magnitudes[i], self.angles[i]])


class DynamicSpectrum:
    def __init__(self, rated_power: float = None, phase: int = -1, epsilon: float = 1e-3):
        """
        Dynamic spectrum class
        :param rated_power: Rated power of the device in Watt
        :param phase: one of 1,2,3 for single phases or -1 if all phases. NOTE: if -1, loads are calculated single phase only
        :param epsilon: Small value that gives a range for which a spectrum is considered to be valid for a given load.
        """
        self.spectra_data = {}
        self.phase = phase
        self.rated_power = rated_power
        self.epsilon = epsilon

    def get_spectrum_as_percentage_of_desired_power(self, spectrum: Spectrum, desired_power_watt: float) -> Spectrum:
        """
        Spectra are stored in the database (and in this object) as percentage of nominal power.
        For simulation purposes, we need to convert the spectra to percentages of actual power values
        with the first harmonic as 100%.
        E.g. if the spectrum was recorded at 3% of the nominal power, the magnitude of the first harmonic is 3%.
        We convert this to 100% by multiplying all magnitudes with the factor 100/3.
        :param spectrum: Spectrum object
        :param desired_power_watt: desired power in Watt
        :return: Spectrum object
        """
        if self.rated_power is None:
            raise ValueError("Nominal power must be set.")
        if np.isclose(desired_power_watt, self.rated_power, rtol=self.epsilon):
            return spectrum
        # Calculate the factor to convert the spectrum to the desired power
        factor = self.rated_power / desired_power_watt
        harmonics = spectrum.harmonics
        magnitudes = [mag * factor for mag in spectrum.magnitudes]
        angles = spectrum.angles
        return Spectrum(harmonics, magnitudes, angles, spectrum.name)

    def add_spectrum(self, **kwargs):
        raise NotImplementedError("This method must be implemented in the subclass.")

    def get_spectrum(self, **kwargs) -> Spectrum:
        raise NotImplementedError("This method must be implemented in the subclass.")

    @staticmethod
    def limit_interpolation_results(interpolated_magnitudes, interpolated_angles) -> tuple:
        """
        Limit magnitudes to 0% < x < 100%
        Limit angles to -180° to +180° by wrapping extrapolation results
        :return:
        """
        # Limit magnitudes to 0% < x < 100%
        interpolated_magnitudes = np.clip(interpolated_magnitudes, 0, 100)

        # Limit angles to -180° to +180° by wrapping extrapolation results
        interpolated_angles = np.mod(interpolated_angles, 360)
        interpolated_angles[interpolated_angles > 180] -= 360
        interpolated_angles[interpolated_angles <= -180] += 360
        return interpolated_magnitudes, interpolated_angles


class LoadVaryingSpectrum(DynamicSpectrum):
    def __init__(self, rated_power: float = None, phase: int = -1, epsilon: float = 1e-3):
        """
        Load varying spectrum class to store spectra for different load powers.
        :param rated_power: Rated power of the device in Watt
        :param phase: one of 1,2,3 for single phases or -1 if all phases. NOTE: if -1, loads are calculated single phase only
        :param epsilon: Small value that gives a range for which a spectrum is considered to be valid for a given load.
        """
        super().__init__(rated_power, phase, epsilon)

    def add_spectrum(self, spectrum: Spectrum, **kwargs):
        """
        Add a spectrum to the load varying spectrum
        :param spectrum: spectrum object
        :param kwargs: load_power_watt - Power in Watt
        :return:
        """
        load_power_watt = kwargs.get('load_power_watt', 0.0)
        self.spectra_data[load_power_watt] = spectrum

    def get_spectrum(self, **kwargs) -> Spectrum:
        """
        Get spectrum for a given load power.
        If the spectrum is not available, it is interpolated or extrapolated.
        If only one spectrum is available, it is returned.
        :param kwargs: load_power_watt, power in Watt
        :return: spectrum object
        """
        load_power_watt = kwargs.get('load_power_watt', 0.0)
        result = find_key_and_value_with_epsilon(load_power_watt, self.spectra_data, self.epsilon)
        if result:
            _, matched_value = result
            return matched_value
        if len(self.spectra_data) == 1:  # not enough data for interpolation
            return list(self.spectra_data.values())[0]
        return self._interpolate_spectrum(load_power_watt, method=cfg.SPECTRA_INTERPOLATION)

    def _interpolate_spectrum(self, load_power_watt: float, method: str = 'linear') -> Spectrum:
        """
        Interpolate or extrapolate spectrum for a given load power
        :param load_power_watt: power in Watt
        :param method: interpolation method, linear, cubic or spline
        :return: spectrum object
        """
        if len(self.spectra_data) < 2:
            raise ValueError("Insufficient data for interpolation.")

        # Create arrays for interpolation
        loads = np.array(list(self.spectra_data.keys()))
        harmonics = next(iter(self.spectra_data.values())).harmonics
        magnitudes = np.array([spectrum.magnitudes for spectrum in self.spectra_data.values()])
        angles = np.array([spectrum.angles for spectrum in self.spectra_data.values()])

        logger.debug(f"Interpolating spectrum for load {load_power_watt} W")
        logger.debug(f"Loads: {loads}")
        logger.debug(f"Magnitudes: {magnitudes}")
        logger.debug(f"Angles: {angles}")
        # Perform interpolation or extrapolation
        interpolated_magnitudes = self._interpolate_or_extrapolate(loads, magnitudes, load_power_watt, method)
        interpolated_angles = self._interpolate_or_extrapolate(loads, angles, load_power_watt, method)

        logger.debug(f"Interpolated magnitudes: {interpolated_magnitudes}")
        logger.debug(f"Interpolated angles: {interpolated_angles}")

        interpolated_magnitudes, interpolated_angles = self.limit_interpolation_results(interpolated_magnitudes,
                                                                                        interpolated_angles)

        logger.debug(f"Clipped magnitudes: {interpolated_magnitudes}")
        logger.debug(f"Wrapped angles: {interpolated_angles}")
        return Spectrum(harmonics, interpolated_magnitudes, interpolated_angles)

    @staticmethod
    def _interpolate_or_extrapolate(x: np.ndarray, y: np.ndarray, x_new: float, method: str = 'linear',
                                    extrapolate=False):
        """
        Interpolate or extrapolate a function
        :param x: x values
        :param y: y values
        :param x_new: target x values
        :param method: linear, cubic, ... see scipy documentation
        :param extrapolate: True if extrapolation is allowed; else limit x_new to the range of x
        :return: interpolated or extrapolated y values
        """
        # Sort x and y values by x
        indices = np.argsort(x)
        xs = x[indices]
        ys = y[indices]

        if not extrapolate:  # if extrapolation is not allowed, limit x_new to the range of x
            if x_new < np.min(xs):
                x_new = np.min(xs)
            elif x_new > np.max(xs):
                x_new = np.max(xs)
            else:
                pass

        if method == 'cubic_spline':
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(xs, ys, bc_type='natural')
            return cs(x_new)
        f = interpolate.interp1d(xs, ys, kind=method, fill_value='extrapolate', axis=0)
        return f(x_new)


class TimeSeriesSpectrum(DynamicSpectrum):
    def __init__(self, rated_power: float = None, phase: int = -1, time_step_ms: int = 0, epsilon: float = 1e-3):
        """
        Time series spectrum class to store spectra for different time steps.
        :param rated_power: Rated power of the device in Watt
        :param phase: one of 1,2,3 for single phases or -1 if all phases. NOTE: if -1, loads are calculated single phase only
        :param time_step_ms: Time step size in milliseconds
        :param epsilon: Small value that gives a range for which a spectrum is considered to be valid for a given load.
        """
        super().__init__(rated_power, phase, epsilon)
        self.step_size_ms = time_step_ms
        self.time_steps = []
        self.counter_index = 0

    def add_spectrum(self, spectrum: Spectrum, **kwargs):
        """
        Add a spectrum to the time series spectrum
        :param spectrum: Spectrum object
        :param kwargs: time_step - integer time step, if None: internal counter is used
        TODO: Implement more sophisticated time series handling (daily, weekly, monthly, ...)
        :return:
        """
        time_step = kwargs.get('time_step', None)
        if time_step is None:
            time_step = len(self.spectra_data)
        self.spectra_data[time_step] = spectrum
        self.time_steps = sorted(self.spectra_data.keys())  # Keep the list of time steps sorted
        self.counter_index = 0  # Reset the counter index when a new spectrum is added

    def get_spectrum(self, **kwargs) -> Spectrum:
        """
        Get spectrum for a given time step.
        If no time_step is provided, an internal counter is used.
        :param kwargs: time_step: integer time step, target_step_size_ms: target step size in milliseconds
        :return: Spectrum object
        """
        time_step = kwargs.get('time_step', None)
        target_step_size_ms = kwargs.get('target_step_size_ms', self.step_size_ms)
        if time_step is None:
            current_time_step = self.time_steps[self.counter_index]
            self.counter_index += 1
            if self.counter_index >= len(self.time_steps):  # Loop to the beginning
                self.counter_index = 0
        else:
            current_time_step = time_step
        time_keys = np.array(list(self.spectra_data.keys()))

        if np.isclose(self.step_size_ms, target_step_size_ms, rtol=self.epsilon):
            if current_time_step > max(time_keys):
                current_time_step %= max(time_keys)
            current_spectrum = self.spectra_data[current_time_step]
            return current_spectrum

        # Calculate new time index based on target step size
        time_scale_factor = target_step_size_ms / self.step_size_ms
        new_time_step = current_time_step * time_scale_factor

        if new_time_step > max(time_keys):
            new_time_step %= max(time_keys)  # Loop around if the step is out of existing range

        current_spectrum = self.spectra_data[new_time_step]
        harmonics = current_spectrum.harmonics
        time_keys = np.array(list(self.spectra_data.keys()))
        magnitudes = np.stack([spectrum.magnitudes for spectrum in self.spectra_data.values()])
        angles = np.stack([spectrum.angles for spectrum in self.spectra_data.values()])

        # Perform interpolation for each harmonic
        interpolated_magnitudes = self._interpolate_values(time_keys, magnitudes, new_time_step, cfg.SPECTRA_INTERPOLATION)
        interpolated_angles = self._interpolate_values(time_keys, angles, new_time_step, cfg.SPECTRA_INTERPOLATION)
        interpolated_magnitudes, interpolated_angles = self.limit_interpolation_results(interpolated_magnitudes,
                                                                                        interpolated_angles)
        return Spectrum(harmonics, interpolated_magnitudes, interpolated_angles)

    @staticmethod
    def _interpolate_values(x: np.ndarray, y: np.ndarray, x_new: float, method: str = 'linear'):
        """
        Helper function to interpolate or extrapolate values
        :param x: time steps as x values
        :param y: magnitudes or angles as y values
        :param x_new: target time step as x_new value
        :param method: interpolation method
        :return: interpolated or extrapolated y values
        """
        indices = np.argsort(x)
        xs = x[indices]

        if method == 'cubic_spline':
            from scipy.interpolate import CubicSpline
            # Interpolate each component of y separately
            interpolated_y = [CubicSpline(xs, y_component[indices], bc_type='natural')(x_new) for y_component in y.T]
        else:
            f = interpolate.interp1d(xs, y[indices], kind=method, fill_value='extrapolate', axis=0)
            interpolated_y = f(x_new)
        return interpolated_y


class RandomSpectrum(DynamicSpectrum):
    def __init__(self, max_harmonic_values: dict, random_func: Callable, noise_params: dict, rated_power: float = None,
                 phase: int = -1, epsilon: float = 1e-3, range_harmonic_angles=None):
        super().__init__(rated_power, phase, epsilon)
        self.random_func = random_func
        self.noise_params = noise_params
        self.max_harmonic_values = max_harmonic_values
        self.range_harmonic_angles = range_harmonic_angles
        self.name = 'randomspectrum'
        self.max_harmonic_values.pop(1, None)  # remove first harmonic
        if self.range_harmonic_angles:
            self.range_harmonic_angles.pop(1, None)  # remove first harmonic angle as it should be 0°

    def get_spectrum(self, **kwargs) -> Spectrum:
        harmonics = list(self.max_harmonic_values.keys())
        max_values_magnitudes = np.array(list(self.max_harmonic_values.values()))
        uniform_arr = self.random_func(size=len(harmonics), **self.noise_params)
        magnitudes = list(max_values_magnitudes * uniform_arr * 100)

        if self.range_harmonic_angles:
            ranges_angles = list(self.range_harmonic_angles.values())
            angles = [random.uniform(lower, upper) for lower, upper in ranges_angles]
        else:
            angles = list(np.zeros(len(magnitudes)))

        harmonics.insert(0, 1)
        magnitudes.insert(0, 100)
        angles.insert(0, 0)
        spectrum = Spectrum(harmonics, magnitudes, angles, name=self.name)
        return spectrum

    def add_spectrum(self, **kwargs):
        raise NotImplementedError("Not applicable to RandomSpectrum")
