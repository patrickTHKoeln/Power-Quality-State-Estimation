import os

from OpenDSSWrapper.converter.convert_pandapower import pandapower_to_opendss
from OpenDSSWrapper.converter.grids import BaseGrid
from OpenDSSWrapper.converter.grids import CigreLVDistPV
from OpenDSSWrapper.converter.grids import MicrogridTHKoeln
from OpenDSSWrapper.converter.grids import IEEE33TestFeeder
from OpenDSSWrapper.converter.admittance import get_y_matrices_per_frequency
from OpenDSSWrapper.tools.config import data_path


def ieee33_bus_system():
    net = IEEE33TestFeeder().pp_net
    cigre_lvs_path = os.path.join(data_path, 'IEEE33')
    pandapower_to_opendss(net, 'IEEE33', cigre_lvs_path)


def cigre_lv_dist():
    net = CigreLVDistPV().pp_net
    cigre_lvs_path = os.path.join(data_path, 'CIGRELVDist')
    pandapower_to_opendss(net, 'CIGRELVDist', cigre_lvs_path)


def microgrid_th_koeln():
    net = MicrogridTHKoeln().pp_net
    microgrid_thkoeln_path = os.path.join(data_path, 'THKoelnNetwork')
    pandapower_to_opendss(net, 'THKoelnNetwork', microgrid_thkoeln_path)


def save_y_mats(grid: BaseGrid):
    import pickle
    import numpy as np
    base_kvs = list(grid.get_base_kvs().values())
    y_mats = get_y_matrices_per_frequency(grid.pp_net, 20, base_kvs, list(np.arange(50, 1050, 50)), shunt_position=-1)
    with open('y_mats_ground.pic', 'wb') as pic:
        pickle.dump(y_mats, pic)


if __name__ == '__main__':
    # cigre_lv_dist()
    microgrid_th_koeln()
    # ieee33_bus_system()
    # grid = CigreLVDistPV()
    # compare_matrices(grid)
    # save_y_mats(grid)
