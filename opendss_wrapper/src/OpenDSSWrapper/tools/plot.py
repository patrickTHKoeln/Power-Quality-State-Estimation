import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from scipy.spatial import ConvexHull


def plot_pandapower(net, highlight_buses=None, plot_bus_labels=False, offset_label=(0, 0), meter_reach: dict = None,
                    save_path=None, show_plot=True, hatch=False, bus_geodata=None):
    """
    Plot the pandapower network

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network.
    highlight_buses : list of int, optional
        List of bus indices to highlight.
    plot_bus_labels : bool, optional
        If True, bus labels are plotted.
    offset_label : tuple, optional
        Tuple offset for positioning bus labels.
    meter_reach : dict, optional
        Dictionary where keys are meter bus indices, values are lists of tuples (node, affinity).
    save_path : str, optional
        If not None, path to save the resulting plot.
    show_plot : bool, optional
        If True, displays the plot.
    hatch : bool, optional
        If True, draws hatches on the circles around measured nodes.
    bus_geodata : pandas.DataFrame or dict
        Geographical coordinates for buses. Keys or index are bus indices, columns or values are [x, y].
    """
    # Create default geodata for the net
    net = pp.plotting.create_generic_coordinates(net, overwrite=True, respect_switches=True)

    # Set bus geodata from user
    import pandas as pd
    if isinstance(bus_geodata, pd.DataFrame):
        net["bus_geodata"] = bus_geodata
    elif isinstance(bus_geodata, dict):
        bus_geodata_df = pd.DataFrame.from_dict(bus_geodata, orient='index')
        bus_geodata_df.columns = ['x', 'y']
        bus_geodata_df.index = bus_geodata_df.index.astype(int)
        net["bus_geodata"] = bus_geodata_df
    else:
        raise TypeError("bus_geodata must be a pandas DataFrame or a dictionary.")

    # Generate line geodata if needed
    net = generate_line_geodata(net)

    # Plot the network
    ax = pp.plotting.simple_plot(
        net, respect_switches=True, switch_color='green', plot_line_switches=True,
        line_width=1, plot_loads=False, plot_sgens=True, bus_size=0.8,
        bus_color='lightsteelblue', trafo_color='red', show_plot=False
    )

    # Plot bus labels if requested
    if plot_bus_labels:
        for bus_idx, bus in net.bus.iterrows():
            bus_coords = net.bus_geodata.loc[bus_idx]
            text_x, text_y = bus_coords.x + offset_label[0], bus_coords.y + offset_label[1]
            ax.plot([bus_coords.x, text_x], [bus_coords.y, text_y], color="gray", linestyle="--", zorder=3)
            ax.text(text_x, text_y, str(bus_idx), color='black', ha='center', va='center', zorder=5)

    # Prepare a dictionary that maps meter keys (bus indices) to colors
    meter_colors = {}

    # If meter_reach is provided, plot circles around measured nodes
    if meter_reach:
        color_map = plt.cm.get_cmap('viridis', len(meter_reach))
        plt.rcParams["hatch.linewidth"] = 4
        hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '//', '\\\\', '||', '--', '++', 'xx', 'oo',
                   'OO', '..', '**']

        # Linearly spaced radii from min_r to max_r
        min_r = 0.4
        max_r = 0.6
        radius_values = np.linspace(min_r, max_r, len(meter_reach))

        # Assign a unique color and radius to each meter bus
        for i, (meter_bus, nodes) in enumerate(meter_reach.items()):
            assigned_color = color_map(i)
            meter_colors[meter_bus] = assigned_color  # <--- Store this meter's color

            # Plot circles around each node that this meter measures
            for j, (node_idx, affinity) in enumerate(nodes):
                if node_idx not in net.bus_geodata.index:
                    continue
                node_coords = net.bus_geodata.loc[node_idx, ['x', 'y']]
                alpha = affinity # should be in range 0..1
                if hatch:
                    circle = Circle(
                        (node_coords.x, node_coords.y), radius_values[i],
                        edgecolor=assigned_color, alpha=alpha, zorder=1 / (i + 1),
                        hatch=hatches[i % len(hatches)], facecolor='none'
                    )
                else:
                    circle = Circle(
                        (node_coords.x, node_coords.y), radius_values[i],
                        color=assigned_color, alpha=alpha, zorder=1 / (i + 1)
                    )
                ax.add_patch(circle)

    # Highlight buses, colored according to meter_colors if they appear as meter keys
    if highlight_buses:
        for bus_idx in highlight_buses:
            if bus_idx in net.bus_geodata.index:
                bus_coords = net.bus_geodata.loc[bus_idx]

                # Check if this highlight bus is also a key in meter_reach; otherwise use green
                if bus_idx in meter_colors:
                    highlight_color = meter_colors[bus_idx]
                else:
                    highlight_color = 'green'

                ax.plot(
                    bus_coords.x, bus_coords.y, 's', markerfacecolor=highlight_color,
                    markeredgecolor='black', markersize=15, zorder=20
                )

    # Save and/or show plot
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def generate_line_geodata(net):
    """
    Line geodata is sometimes not generated correctly in pandapower
    Generate manually
    :param net: pandapower network
    :return: modified net
    """
    for idx, line in net.line.iterrows():
        from_bus = net.bus_geodata.loc[line.from_bus]
        to_bus = net.bus_geodata.loc[line.to_bus]
        # Straight line from 'from_bus' to 'to_bus'
        coords = [[from_bus.x, from_bus.y], [to_bus.x, to_bus.y]]
        if idx in net.line_geodata.index:
            net.line_geodata.at[idx, 'coords'] = coords
        else:
            net.line_geodata.loc[idx] = [coords]
    return net
