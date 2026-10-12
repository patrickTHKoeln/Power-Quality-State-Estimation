# Converter

In this module conversion tools are implemented.

## CGMES to Pandapower

File: `convert_cgmes.py`
Uses pandapower conversion tools and `cimpyorm` library to convert a CGMES file to a pandapower network.

- requires an old sqlalchemy version
- [cimpyorm github](https://github.com/RWTH-IAEW/cimpyorm)
- [cimpyorm paper](https://doi.org/10.1109/ISGTEurope.2019.8905684)
- [Readthedocs documentation](https://readthedocs.org/projects/cimpyorm/downloads/pdf/latest/)
- [CGMES Documentation](https://www.entsoe.eu/data/cim/cim-for-grid-models-exchange/)

## Pandapower to OpenDSS

File: `convert_pandapower.py`
Converts a pandapower network to a OpenDSS scripts (file-based).

## Components calculations and naming conventions
Converter Components, calculate missing variables starting with pandapower description.
Missing parameters are calculated in the _calculate_parameters method modifying the original dataframe.
Additional parameters which do not exist in pandapower are also written to the same dataframe

Nomenclature:
- akin to pandapower
- n = nominal
- pu = per unit
- sc = short circuit
- snake_case is used for variable names
- unit at the end of the variable name
- if the unit has two components that are divided like henry per kilometer, they are separated by '_per_'
- p2p for phase to phase
- p2e for phase to earth

For example:
- fn_hz: Nominal frequency in Hz
- vn_v: Nominal voltage in V
- i_sc_a: Short circuit current in A (different to pandapower, where this would be ik_a)
- r_ohm_per_km: Resistance in Ohm per km

Equations and additional explanations are added in comments.
Variables that are constant throughout this program, but might be defined differently depending on use case,
should be defined in the config.ini file (see OpenDSSWrapper/tools/config.py)
To access parameters of the dataframe, use self.df['column_name'] instead of self.df.column_name which is invalid
for reserved keywords like type.
When assigning a value to a dataframe column, use self.df.loc['column_name'] = value
