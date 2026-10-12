import numpy as np
from scipy.spatial.distance import directed_hausdorff


# Hilfsfunktion, um komplexe Zahlen in ein 2D-Array umzuwandeln
def complex_to_2d_array(complex_array):
    return np.array([(z.real, z.imag) for z in complex_array])


def compute_hausdorff_distance(target_set_2d, compare_set_2d):

    # Berechne die gerichtete Hausdorff-Distanz in beide Richtungen
    distance_a_to_b = directed_hausdorff(compare_set_2d, target_set_2d)[0]
    distance_b_to_a = directed_hausdorff(target_set_2d, compare_set_2d)[0]

    # Berechne die symmetrische Hausdorff-Distanz
    hausdorff_distance = max(distance_a_to_b, distance_b_to_a)

    return hausdorff_distance


def compute_hausdorff_distance_set(target_set_2d, compare_sets_2d):

    # Initialisiere Variablen zur Verfolgung der minimalen Distanz
    min_distance = float('inf')
    closest_set = None

    # Berechne die Hausdorff-Distanz für jede Menge in A zu B
    for c_set in compare_sets_2d:

        # Berechne die gerichtete Hausdorff-Distanz in beide Richtungen
        distance_a_to_b = directed_hausdorff(c_set, target_set_2d)[0]
        distance_b_to_a = directed_hausdorff(target_set_2d, c_set)[0]

        # Berechne die symmetrische Hausdorff-Distanz
        hausdorff_distance = max(distance_a_to_b, distance_b_to_a)

        # Aktualisiere das nächste Set, falls erforderlich
        if hausdorff_distance < min_distance:
            min_distance = hausdorff_distance
            closest_set = c_set

    return compare_sets_2d.index(closest_set)


def compute_euclidean_distance(target_set, compare_set, deltas=None):
    if deltas is None:
        deltas = target_set - compare_set
    euclidean_distance = np.sqrt(np.sum(np.square(deltas)))
    return euclidean_distance


def compute_mae_distance(target_set, compare_set, deltas=None):
    if deltas is None:
        deltas = target_set - compare_set
    mae_distance = np.mean(np.abs(deltas))
    return mae_distance


def compute_mse_distance(target_set, compare_set, deltas=None):
    if deltas is None:
        deltas = target_set - compare_set
    mse_distance = np.mean(np.square(deltas))
    return mse_distance
