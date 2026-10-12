import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from OpenDSSWrapper.harmonic_spectra import LoadVaryingSpectrum, Spectrum, DynamicSpectrumType


def plot_load_varying_spectrum(lv_spectrum: LoadVaryingSpectrum, save_path=None,
                               harmonics_to_drop: list[float | int] = None, show_plot=True):
    """
    Create a 3D bar plot of the load varying spectrum with harmonics on x-axis, load power on y-axis, and magnitude on z-axis.
    :param lv_spectrum: LoadVaryingSpectrum object
    :param harmonics_to_drop: None or list of harmonics to drop from the plot
    :param save_path: path to save the plot, if None, the plot will only be shown
    :param show_plot: Show the plot
    :return:
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xpos = []
    ypos = []
    zpos = []
    dz = []
    dz_angle = []

    # Sort the dictionary by load power
    sorted_spectra = sorted(lv_spectrum.spectra_data.items(), key=lambda x: x[0])

    for idx, (load, spectrum) in enumerate(sorted_spectra):
        spectrum = _drop_harmonics(spectrum, harmonics_to_drop)
        harmonics = spectrum.harmonics
        magnitudes = spectrum.magnitudes
        angles = spectrum.angles

        ypos.extend([load] * len(harmonics))
        xpos.extend(harmonics)
        zpos.extend([0] * len(harmonics))
        dz.extend(magnitudes)
        dz_angle.extend(angles)

    # Normalize the phase angles to map to colors
    angle_norm = Normalize(min(dz_angle), max(dz_angle))
    angle_colors = plt.cm.viridis(angle_norm(dz_angle))

    ax.bar3d(xpos, ypos, zpos, 0.4, 0.4, dz, shade=True, color=angle_colors)
    ax.set_xlabel('Harmonics')
    ax.set_ylabel('Load Power [W]')
    ax.set_zlabel('Magnitude')
    ax.set_title('Load Varying Spectrum Analysis')

    mappable = plt.cm.ScalarMappable(norm=angle_norm, cmap=plt.cm.viridis)
    mappable.set_array(dz_angle)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label('Phase Angle [degrees]')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"load_varying_spectrum.png"))
    if show_plot:
        plt.show()
    else:
        plt.close()


def _drop_harmonics(spectrum: Spectrum, harmonics_to_drop: list[float | int]) -> Spectrum:
    """
    Drop harmonics from a spectrum object
    :param spectrum: Spectrum object
    :param harmonics_to_drop: Harmonics to drop
    :return: Spectrum object with dropped harmonics
    """
    if harmonics_to_drop is None:
        return spectrum
    harmonics = []
    magnitudes = []
    angles = []
    for idx, harmonic in enumerate(spectrum.harmonics):
        if harmonic not in harmonics_to_drop:
            harmonics.append(harmonic)
            magnitudes.append(spectrum.magnitudes[idx])
            angles.append(spectrum.angles[idx])
    return Spectrum(harmonics, magnitudes, angles)


def reconstruct_sine_waves_3d(lv_spectrum: LoadVaryingSpectrum, mode: str = "I", phase: int = 1, f_nominal_hz: int = 50,
                              harmonics_to_drop: list[float | int] = None, save_path=None, show_plot=True):
    """
    Reconstruct sine waves for all loads in the load varying spectrum and plot them in 3D.
    :param lv_spectrum: LoadVaryingSpectrum object
    :param mode: I for current, U for voltage
    :param phase: phase number
    :param f_nominal_hz: Nominal frequency in Hz
    :param harmonics_to_drop: Harmonics to drop from the plot
    :param save_path: Path to save the plot, if None, the plot will only be shown
    :param show_plot: Show the plot
    :return:
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 1 / f_nominal_hz, 1000) * 1000  # in milliseconds

    loads = sorted(lv_spectrum.spectra_data.keys())

    for load in loads:
        spectrum = lv_spectrum.spectra_data[load]
        spectrum = _drop_harmonics(spectrum, harmonics_to_drop)
        reconstructed_sine = np.zeros((len(spectrum.harmonics), len(t)))

        for idx in range(len(spectrum.harmonics)):
            harmonic_number = spectrum.harmonics[idx]
            magnitude = spectrum.magnitudes[idx]
            phase_angle = spectrum.angles[idx]  # assuming phase angle is in radians
            reconstructed_sine[idx, :] = magnitude * np.sin(
                2 * np.pi * harmonic_number * f_nominal_hz * t + phase_angle)

        final_sine = np.sum(reconstructed_sine, axis=0)
        ax.plot(t, [load] * len(t), final_sine, label=f'{int(load)} W')  # Load rounded to no decimal

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Load Power (W)')
    if mode == 'I':
        ax.set_title(f'Reconstructed Current Sine Wave for Phase {phase}')
        ax.set_zlabel('Current (in % of fundamental current)')
    else:
        ax.set_title(f'Reconstructed Voltage Sine Wave for Phase {phase}')
        ax.set_zlabel('Voltage (in % of fundamental voltage)')

    # Move the legend to the right side of the plot
    ax.legend(title="Load Power", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.subplots_adjust(right=0.75)  # Adjust the main plot to make space for the legend

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"reconstructed_sine_{mode}{phase}.png"))
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spectrum(spectrum: Spectrum, save_path=None, harmonics_to_drop: list[float | int] = None, show_plot=True,
                  name=""):
    """
    Plot a spectrum object as 2d bar plot
    :param spectrum: Spectrum object
    :param harmonics_to_drop: Harmonics to drop from the plot
    :param save_path: Path to save the plot, if None, the plot will only be shown
    :param show_plot: Show the plot
    :return:
    """
    spectrum = _drop_harmonics(spectrum, harmonics_to_drop)
    harmonics = spectrum.harmonics
    magnitudes = spectrum.magnitudes
    angles = spectrum.angles
    fig, ax = plt.subplots()

    # Normalize phase angles for color mapping
    norm = Normalize(vmin=min(angles), vmax=max(angles))
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(angles))

    # Create a bar plot
    bars = ax.bar(harmonics, magnitudes, color=colors)

    ax.set_xlabel('Harmonics')
    ax.set_ylabel('Magnitude')
    if spectrum.name:
        ax.set_title(f'Spectrum of {spectrum.name}')
    else:
        ax.set_title(f'Spectrum {name}')

    # Create a colorbar with a label
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Phase Angle (Degrees)')

    # Save or show the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'spectrum_plot_{name}.png'))
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    from OpenDSSWrapper.harmonic_spectra import SpectraStoreSQLite, SpectrumConfig
    from OpenDSSWrapper.tools.config import data_path, spectrum_db_name

    db_file = os.path.join(data_path, spectrum_db_name)
    spectra_store = SpectraStoreSQLite(db_file)
    spectrum_config = SpectrumConfig("MicrogridlabInverter", None, "GENERATION", "PHOTOVOLTAIC_SYSTEM", phase=1,
                                     dynamic_type=DynamicSpectrumType.LOAD_VARYING)
    dyn_spectrum = spectra_store.get_spectra(spectrum_config)
    plot_load_varying_spectrum(dyn_spectrum, save_path=None, harmonics_to_drop=[1])
    reconstruct_sine_waves_3d(dyn_spectrum, save_path=None, mode='U', harmonics_to_drop=[1])
    spectrum = dyn_spectrum.get_spectrum(load_power_watt=0)
    plot_spectrum(spectrum, save_path=None, harmonics_to_drop=[1])
