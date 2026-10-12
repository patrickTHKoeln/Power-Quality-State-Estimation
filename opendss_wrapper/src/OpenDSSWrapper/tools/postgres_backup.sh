#!/bin/bash

# Set the date format for the backup file name
DATE=$(date +%Y%m%d%H%M)

# Set PostgreSQL credentials
PG_USER="your_postgres_user"
PG_PASSWORD="your_postgres_password"
PG_HOST="localhost"
PG_PORT="5432"
PG_DB="your_database_name"

# Set the backup directory
BACKUP_DIR="/path/to/your/backup/directory"

# Create a backup
PGPASSWORD=$PG_PASSWORD pg_dump -U $PG_USER -h $PG_HOST -p $PG_PORT -d $PG_DB > $BACKUP_DIR/db_backup_$DATE.sql
