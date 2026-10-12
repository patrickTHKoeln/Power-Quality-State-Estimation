# Power Profiles

Stored in a SQLite database or in a PostgreSQL database.

- searchable by category, subcategory, id
- allows for
    - searching specific time steps
    - specifying a step size through interpolation
    - adding random noise
    - scaling
    - normalizing data to a range of 0 to 1 (min-max scaling)

Each profile must at least have 2 data points so that interpolation can be performed.

## Vsource
For voltage sources, the power profile must be given in per unit values.
The voltage source profile should not be normalized.