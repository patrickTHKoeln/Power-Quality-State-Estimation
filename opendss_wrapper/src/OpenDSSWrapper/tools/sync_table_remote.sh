#!/bin/bash

## Sync a table from a local database to a remote database

# Define db connection parameters
LOCAL_DBNAME="local_dbname"
REMOTE_DBNAME="remote_dbname"
TABLE_NAME="table_name"
LOCAL_USERNAME="local_username"
REMOTE_USERNAME="remote_username"
REMOTE_HOST="remote_host"
REMOTE_PORT="remote_port"
REMOTE_PASSWORD="remote_password"

# Step 1: Dump the table from the local database
PGPASSWORD=$REMOTE_PASSWORD pg_dump -U $LOCAL_USERNAME -d $LOCAL_DBNAME -t $TABLE_NAME -f "$TABLE_NAME.dump"

# Step 2: Import the dump into the remote database
psql -U $REMOTE_USERNAME -h $REMOTE_HOST -p $REMOTE_PORT -d $REMOTE_DBNAME -f "$TABLE_NAME.dump"

# Optional: Remove the dump file if it's no longer needed
rm "$TABLE_NAME.dump"
