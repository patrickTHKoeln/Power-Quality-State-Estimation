# Grids

Grid descriptions that combine pandapower networks with load profiles, generator profiles and harmonic spectra.
From this grid description a simulation can be conducted.

Grid descriptions must implement `BaseGrid`.



## How to define a grid

1. Create a new python file called MyGrid.py in the `converter.grids` subpackage.
2. Implement a class `MyGrid` that inherits from `BaseGrid`.
3. Define how to build the grid in pandapower
4. Implement all its methods:
    - define spectra, load profiles, generator profiles for each load and generator
    - spectra are only defined by a configuration object 
    - the actual spectra are stored in a database and are retrieved later

If you want to get the pandapower network from externally, use `mygrid.pp_net`. 
The `_get_pandapower()` method is private since adjustments to the grid are performed in the base class on creation.
