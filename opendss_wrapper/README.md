# openDSS Wrapper

Wrapper around OpenDSS tool with the following features

- Convert pandapower and CGMES networks to OpenDSS
- parallel execution of (harmonic) simulations
- Harmonic spectra dependent on load assigned to each element
- Power profiles assigned to loads and generators
- Store simulation results as
    - HDF5
    - CSV
    - SQL
    - Pickle
- Postprocessing:
    - Unify Voltage level of all buses (e.g. all to MV)
    - calculate voltages at all buses starting with a set of given buses (pre-calculations)

Examples for calling the simulation are provided in the main file `__main__.py`.

For further information about the different modules, see the respective README files.
- [Power Profiles](src/OpenDSSWrapper/power_profiles/README.md)
- [Harmonic Spectra](src/OpenDSSWrapper/harmonic_spectra/README.md)
- [Grids](src/OpenDSSWrapper/converter/grids/README.md)
- [Converter](src/OpenDSSWrapper/converter/README.md)
- [Tools](src/OpenDSSWrapper/tools/README.md)

## Build and Install

From within your virtual environment:

- change directory to the root of the project
- `pip install .`
- or in editable mode
    - `pip install -e .`

Configure paths to OpenDSS scripts, power profiles database, spectra database etc. in 'config.ini'

## Run

1. Create or copy a harmonic spectra database and a power profiles database.
2. Define a grid in the `converter.grids` subpackage that implements `BaseGrid`
3. *(optional)* convert the grid to an OpenDSS model and confirm all elements are correctly assigned
4. choose a storage type (e.g. HDF5, CSV, SQL, Pickle) by defining a storage class
5. *(optional)* configure logging in the `tools/logging.py` file
6. run the simulation (see main file)

Conversion to OpenDSS is also performed automatically when the simulation is run based on the implementation of the grid.

## PostgreSQL

If a PostgreSQL database is used for storing results, either set up your own or connect to a remote DB.
To set up your own:

1. `sudo apt install postgresql`
2. Ensure service is running
    - Check Service status `sudo systemctl status 'postgresql*'`
    - Start Umbrella Service `sudo systemctl start postgresql.service`
    - Restart Umbrella Service `sudo systemctl restart postgresql.service`
    - if it shows as active (exited) and no cluster exists, create a cluster:
    - create cluster `sudo pg_createcluster 14 main`
      - change the version number if using a different version of postgres 
    - start cluster `sudo pg_ctlcluster 14 main start`
    - check status `sudo pg_lsclusters`
    - it might also be necessary to assign the correct rights to to postgres user:
    - `sudo chown -R postgresql:postgresql /var/run/postgresql`
3. `sudo -i -u postgres`
4. `psql`
5. `CREATE USER myuser WITH PASSWORD 'mypassword';`
6. `ALTER USER myuser CREATEDB;`
7. `\q`
8. `createdb mydb`

### Move database to external drive
Follow directions [here](https://stackoverflow.com/a/66009977)

### Set up in OpenDSS-Wrapper
Configure the database in `config.ini`.

### Access tables from command line
Log in as postgres user `sudo -i -u postgres`

To view the tables:

1. `psql -d mydb`
2. `\dt`

to show a data overview:

1. `psql -d mydb`
2. `\d data`

to show exemplary data:

1. `psql -d mydb`
2. `SELECT * FROM data LIMIT 20;`