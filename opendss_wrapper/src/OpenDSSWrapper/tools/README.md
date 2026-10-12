# Backup Postgres Database

- Modify script `postgres_backup.sh` to include path and postgres config
- Make script executable `chmod +x ./postgres_backup.sh`
- schedule regular backups:
    - `crontab -e`
    - `0 3 1 * * /path/to/postgres_backup.sh`
    - min, hour, day of month, month, day of week
    - --> 3:00 am on the first day of every month
- manually execute the script when publishing papers

## Sync a table from local database to remote

- Modify script `sync_table_remote.sh` to include postgres config of local and remote
- Make script executable `chmod +x ./sync_table_remote.sh`
- Run script `./sync_table_remote.sh`
