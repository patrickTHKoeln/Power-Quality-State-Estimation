from sklearn.cluster import DBSCAN
import numpy as np


"""
This class can be used to create clusters of grid component states. Best fit clusters can be obtained
"""
class DbscanCluster:

    def __init__(self, rel_eps, min_samples):
        self.db_scans_dict = {}
        self.rel_eps = rel_eps
        self.min_samples = min_samples


    def cluster_data(self, parameters):
        reduced_values = {}
        for p_key in parameters.keys():
            param = np.array(parameters[p_key])
            if param.size > 1:
                abs_eps = self.rel_eps * (np.max(np.abs(param))-np.min(np.abs(param)))
                param_array = np.column_stack((param.real, param.imag))

                # Clustering by DBSCAN
                self.db_scans_dict[p_key] = DBSCAN(eps=abs_eps, min_samples=1).fit(param_array)

                # Ein repräsentativer Wert pro Cluster wählen
                unique_clusters = np.unique(self.db_scans_dict[p_key].labels_)
                representative_values = [param[self.db_scans_dict[p_key].labels_ == label][0] for label in unique_clusters]
                reduced_values[p_key] = representative_values
            else:
                reduced_values[p_key] = parameters[p_key]
                self.db_scans_dict[p_key] = None
        return reduced_values


    def get_cluster_by_combination(self, all_keys, param_keys, best_fit_params, all_params):
        actual_cluster = {}
        for k in param_keys:
            db_scan = self.db_scans_dict[k]
            if db_scan is not None:
                p_idx = np.where(all_keys == k)[0][0]
                all_params_by_key = np.array(all_params[k])
                best_param_idx = np.where(all_params_by_key == best_fit_params[p_idx])[0][0]
                best_param_label =  db_scan.labels_[best_param_idx]
                all_params_cluster = all_params_by_key[db_scan.labels_ == best_param_label]
                actual_cluster[k] = all_params_cluster.tolist()
            else:
                actual_cluster[k] = all_params[k]
        return actual_cluster