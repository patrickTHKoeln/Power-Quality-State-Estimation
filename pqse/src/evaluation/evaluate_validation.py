import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from evaluation.plot import plot_cdf, plot_histogram, plot_pdf, box_plots
import os


def plot_mae_time_series(title, maes, labels, start_idx=0, stop_idx=-1, save_path=None, unit='V'):
    mae_filtered = [m[start_idx: stop_idx] for m in maes]
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    t = np.arange(len(mae_filtered[0]))
    for m, l in zip(mae_filtered, labels):
        # Erste Kurve plotten
        formatted_l = l if l == 'Prediction' else f'{l} Distance'
        plt.plot(t, m, label=formatted_l)

    # Logarithmische Skalierung der y-Achse
    # plt.yscale('log')

    # Achsentitel und Diagrammtitel hinzufügen
    plt.xlabel('n [ ]')
    plt.ylabel(f'MAE [{unit}]')
    plt.title(title)
    # Legende hinzufügen
    plt.legend()
    # Raster hinzufügen
    plt.grid(True, which="both", ls="--")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{title}.png'))
        plt.close()
    else:
        # Plot anzeigen
        plt.show()


def plot_val_vs_pred(title, maes, labels, save_path=None, unit='V'):

    x = maes[labels.index('Prediction')]
    # Plot erstellen
    for m, l in zip(maes, labels):
        if l != 'Prediction':
            # Erste Kurve plotten
            plt.plot(x, m, label=l)

    # Logarithmische Skalierung der y-Achse
    # plt.yscale('log')
    # plt.xscale('log')

    # Achsentitel und Diagrammtitel hinzufügen
    plt.xlabel(f'MAE Prediction [{unit}]')
    plt.ylabel(f'MAE Validations [{unit}]')
    plt.title(title)
    # Legende hinzufügen
    plt.legend()
    # Raster hinzufügen
    plt.grid(True, which="both", ls="--")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{title}.png'))
        plt.close()
    else:
        # Plot anzeigen
        plt.show()


def plot_compared_err_per_node(title, values_est, values, label_1, label_2, scale_label='Nodal Voltage [V]',
                               save_path=None, f_name=None):
    # Plot erstellen

    fig, ax = plt.subplots(2, 1, figsize=(12, 4))
    bar_width = 0.45
    nodes = np.arange(len(values_est))

    # Erste Datenreihe
    ax[0].set_title(title)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].bar(nodes, values_est, bar_width, color='b')
    ax[0].set_ylim(0.9, 1.05)
    # Zweite Datenreihe, verschoben um die Breite eines Balkens
    ax[1].bar(nodes, values, bar_width, label=label_2, color='r')

    # plt.yscale('log')

    # Achsentitel und Diagrammtitel
    ax[1].set_xlabel('Node')
    ax[0].set_ylabel(label_1)
    ax[1].set_ylabel(label_2)
    ax[0].set_xticks(nodes)
    ax[1].set_xticks(nodes)
    ax[0].set_xticklabels([f'{i}' for i in range(len(nodes))])
    ax[1].set_xticklabels([f'{i}' for i in range(len(nodes))])

    plt.tight_layout()
    if save_path is not None and f_name is not None:
        plt.savefig(os.path.join(save_path, f'Nodal_Val_Err_{f_name}.png'), bbox_inches='tight')
        plt.close()
    else:
        # Plot anzeigen
        plt.show()


def plot_multiple_norm_distributions(title, values_per_nodes, best_fits, preds, label_x, unit, save_path=None):

    # Beispiel-Daten: Paramater der Normalverteilungen (Mittelwerte und Standardabweichungen)
    means = np.mean(values_per_nodes, axis=0)
    std_devs = np.std(values_per_nodes, axis=0)

    # Erstellen des Plots
    fig, ax = plt.subplots(figsize=(8, 6))

    min_x = np.min(means - 2 * std_devs)
    max_x = np.max(means + 2 * std_devs)
    # Bereitstellen eines X-Wertes Bereichs
    x = np.linspace(min_x, max_x, 1000)

    # Horizontal verschieben für jede Verteilung
    for i, (mean, std, val, pred) in enumerate(zip(means, std_devs, best_fits, preds)):
        # Normalverteilungs-, pdf-Berechnung
        p = norm.pdf(x, mean, std)

        # Normierung für den Balken
        p = p / np.max(p)

        # Zeichnen des horizontalen Balkens
        ax.plot(x, i + p)
        ax.fill_between(x, i, i + p, alpha=0.7)

        # Zeichnen der vertikalen Linie für den Messwert
        if i == 0:
            ax.plot([val, val], [i, i + 1], color='red', linewidth=1.0, label='Best Fit')
            ax.plot([pred, pred], [i, i + 1], color='blue', linewidth=1.0, label='Prediction')
        else:
            ax.plot([val, val], [i, i+1], color='red', linewidth=1.0)
            ax.plot([pred, pred], [i, i+1], color='blue', linewidth=1.0)

    # Achsenanpassungen
    plt.ylim(0, len(means)-1)
    plt.xlim(min_x, max_x)
    plt.yticks([i for i in range(len(means))], [f'{i}' for i in range(len(means))])
    plt.xlabel(f'{label_x} {unit}')
    plt.ylabel('Nodes')
    plt.title(f'Normal Distributions of physically possible States ({title})')
    plt.grid(True)
    ax.legend(loc='lower left', framealpha=0.9)
    # Diagramm anzeigen
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'Nodal_Dist_{title}_{label_x}.png'), bbox_inches='tight')
        plt.close()
    else:
        # Plot anzeigen
        plt.show()


def plot_value_per_node(vals, name_val='Nodal voltage [V]'):
    # Plot erstellen
    if isinstance(vals[0], complex):
        n_cols = 2
        labels_complex = [' (Real)', ' (Imag)']
        plt_data_1 = [vals.real, vals.imag]
    else:
        n_cols = 1
        labels_complex = ['']
        plt_data_1 = [vals]

    fig, ax = plt.subplots(1, n_cols, figsize=(12, 6))
    bar_width = 0.35
    nodes = np.arange(len(vals))

    for idx_c in range(n_cols):
        # Erste Datenreihe
        ax[idx_c].bar(nodes, plt_data_1[idx_c], bar_width, color='b')

        # Achsentitel und Diagrammtitel
        ax[idx_c].set_xlabel('Node')
        ax[idx_c].set_ylabel(name_val)
        ax[idx_c].set_title(name_val + labels_complex[idx_c])
        ax[idx_c].set_xticks(nodes)
        ax[idx_c].set_xticklabels([f'{i}' for i in range(len(nodes))])  # Optional: Labels für x-Ticks

    # Layout-Anpassung für bessere Darstellung
    plt.tight_layout()

    # Plot anzeigen
    plt.show()


def list_by_key(list_of_dicts, keys):
    ret_list = []
    for k in keys:
        ret_list.append([i[k] for i in list_of_dicts])

    return ret_list


def get_worst_validations(all_states, measure='Prediction', range_states=100):
    all_states_sorted = sorted(all_states, key=lambda x: x[measure])
    return [all_states.index(v) for v in all_states_sorted][-(range_states+1):-1]


def get_best_validations(all_states, measure='Prediction', range_states=100):
    all_states_sorted = sorted(all_states, key=lambda x: x[measure])
    return [all_states.index(v) for v in all_states_sorted][0:range_states]


def count_better_results(all_data):
    counts = [i for i in all_data if i['Prediction'] > i['MSE']]
    return len(counts)


def eval_results_per_node(res, save_path=None):
    best_fits = [res[r]['state_data']['best_fit_mse'] for r in res.keys()]
    pred_states = [res[r]['state_data']['predicted_state'] for r in res.keys()]

    ref_voltage = 20000/math.sqrt(3)
    pred_states_mag = np.abs(pred_states)
    pred_states_mag /= ref_voltage
    pred_states_angle = np.angle(pred_states, deg=True)
    best_fits_mag = np.abs(best_fits)
    best_fits_mag /= ref_voltage
    best_fits_angle = np.angle(best_fits, deg=True)
    all_raw_data = np.array([np.array(l) for l in [res[r]['raw_data'] for r in res.keys()]])
    all_raw_mag = np.abs(all_raw_data)
    all_raw_mag /= ref_voltage
    all_raw_angle = [np.angle(s, deg=True) for s in all_raw_data]
    for i, l in enumerate(res.keys()):
        plot_multiple_norm_distributions(f'State No. {l}', all_raw_mag[i], best_fits_mag[i], pred_states_mag[0],
                                         'Magnitude', '[p.u.]', save_path=save_path)
        plot_multiple_norm_distributions(f'State No. {l}', all_raw_angle[i], best_fits_angle[i], pred_states_angle[0],
                                         'Phase', '[°]', save_path=save_path)


def eval_results(res, compare=None, save_path=None):

    all_comb_mag = [res[r]['mae_results_magnitude'] for r in res.keys()]
    mse_distances_pred_mag = [(res[r]['decision_data']['MAE_min'], res[r]['mae_results_magnitude']['Prediction']) for r in res.keys()]
    mse_distances_pred_mag_sorted = sorted(mse_distances_pred_mag, key=lambda x: x[1])
    all_comb_phase = [res[r]['mae_results_phase'] for r in res.keys()]
    all_comb_mag_sorted = sorted(all_comb_mag, key=lambda x: x['Prediction'])
    all_comb_phase_sorted = sorted(all_comb_phase, key=lambda x: x['Prediction'])
    all_comb_mag_masked = [compare[r]['mae_results_magnitude'] for r in compare.keys()]
    all_comb_phase_masked = [compare[r]['mae_results_phase'] for r in compare.keys()]
    all_comb_mag_sorted_masked = sorted(all_comb_mag_masked, key=lambda x: x['Prediction'])
    all_comb_phase_sorted_masked = sorted(all_comb_phase_masked, key=lambda x: x['Prediction'])
    print(f'{count_better_results(all_comb_mag)} MSE Results improved the state estimation process in magnitude.')
    print(f'{count_better_results(all_comb_phase)} MSE Results improved the state estimation process in phase.')
    indices_best = get_best_validations(all_comb_mag, 'MAE', 5)

    plot_mae_time_series("MAE Prediction vs Validation Distance (Magnitude)", list_by_key(mse_distances_pred_mag_sorted, [0, 1]),
                        ['MSE', 'Prediction'], save_path=save_path)

    # analyze MAE per node (Best)
    pu_ref = 20000/math.sqrt(3)
    unit = 'p.u.' if pu_ref != 1 else 'V'
    for i in indices_best:
        magnitudes_est = np.abs(res[i]['state_data']['predicted_state'])
        abs_errors_val = np.abs(
            np.abs(res[i]['state_data']['best_fit_mae']) - np.abs(res[i]['state_data']['predicted_state']))
        if unit == 'p.u.':
            magnitudes_est /= pu_ref
            abs_errors_val /= pu_ref

        plot_compared_err_per_node(f'Validation Error over Prediction Magnitudes (State No. {i})', magnitudes_est, abs_errors_val,
                                   f'Estimated Magnitudes\n[{unit}]', f'Absolute Validation\nError [{unit}]',
                                   'Absolute Errors in Magnitude', save_path, str(i))

    indices_worst = get_worst_validations(all_comb_mag, 'MAE', 5)
    # analyze MAE per node (Worst)
    for i in indices_worst:
        magnitudes_est = np.abs(res[i]['state_data']['predicted_state'])
        abs_errors_val = np.abs(np.abs(res[i]['state_data']['best_fit_mae']) - np.abs(res[i]['state_data']['predicted_state']))
        if unit == 'p.u.':
            magnitudes_est /= pu_ref
            abs_errors_val /= pu_ref

        plot_compared_err_per_node(f'Validation Error over Prediction Magnitudes (State No. {i})', magnitudes_est, abs_errors_val,
                                   f'Estimated Magnitudes\n[{unit}]', f'Absolute Validation\nError [{unit}]',
                            'Absolute Errors in Magnitude', save_path, str(i))

    # plot dependency of error over available combinations
    # plot CDFs for distribution fitting
    data = list_by_key(all_comb_mag, all_comb_mag[0].keys())
    names = all_comb_mag[0].keys()

    data_masked = list_by_key(all_comb_mag_masked, all_comb_mag_masked[0].keys())

    dict_compare = {}
    dict_compare_masked = {}
    for n, d, dm in zip(names, data, data_masked):
        dict_maes = {}
        if n != 'Prediction':
            dict_maes[f'fit all'] = d
            dict_maes[f'fit to measurement'] = dm
            plot_cdf(dict_maes, 'MAE [V]', f'{n} Distance', color_str='', save_path=save_path)
            dict_compare[n] = d
            dict_compare_masked[n] = dm
    plot_cdf(dict_compare, 'MAE [V]', 'different Distance Measures (fit all)', color_str='',
             save_path=save_path)
    plot_cdf(dict_compare_masked, 'MAE [V]', 'different Distance Measures (fit to measurements)', color_str='',
             save_path=save_path)
    # analyze MAE by timeseries

    select_measure = 'MSE'
    list_keys = ['Prediction', select_measure]

    plot_mae_time_series("Deviation from Ground Truth (Magnitude)", list_by_key(all_comb_mag_sorted, list_keys), list_keys,
                         save_path=save_path)
    # d_list = list_by_key(all_comb_mag_sorted_masked, [select_measure])[0]

    # plot_histogram(d_list, title='Histogram of Best Fit MAEs (Magnitude)', meas=select_measure, save_path=save_path)

    plot_mae_time_series("Deviation from Ground Truth (Phase)", list_by_key(all_comb_phase_sorted, list_keys), list_keys,
                         save_path=save_path, unit='°')
    # d_list = list_by_key(all_comb_phase_sorted_masked, [select_measure])[0]

    # plot_histogram(d_list, title='Histogram of Best Fit MAEs (Phase)', meas=select_measure, save_path=save_path)

    plot_val_vs_pred("MAE Prediction vs Validations (Phase)", list_by_key(all_comb_phase_sorted, list_keys),
                     list_keys, save_path=save_path, unit='°')

    plot_val_vs_pred("MAE Prediction vs Validations (Magnitude)", list_by_key(all_comb_mag_sorted, list_keys),
                     list_keys, save_path=save_path)

    # Plot Boxplots to identify outliers
    b_res = box_plots("MAEs for different Measures (Magnitude)", pd.DataFrame(all_comb_mag_sorted),
              [1.0, 10, 100, 1000, 1000], save_path=save_path)
    for group_name, stats in b_res.items():
        print(f"\n{group_name}")
        print("Medians:\n", stats['Medians'])
        print("1st Quartiles:\n", stats['1st Quartiles'])
        print("3rd Quartiles:\n", stats['3rd Quartiles'])
    b_res = box_plots("MAEs for different Measures (Phase)", pd.DataFrame(all_comb_phase_sorted),
              [0.01, 0.1, 1.0, 10], 'MAE [°]', save_path=save_path)
    for group_name, stats in b_res.items():
        print(f"\n{group_name}")
        print("Medians:\n", stats['Medians'])
        print("1st Quartiles:\n", stats['1st Quartiles'])
        print("3rd Quartiles:\n", stats['3rd Quartiles'])