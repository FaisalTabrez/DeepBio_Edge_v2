import duckdb
import polars as pl
import os
import sys
from pathlib import Path

# Add project root to path to ensure imports work when running as script
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config

def fetch_deep_sea_data(limit=None):
    """
    Fetch Deep Sea data from OBIS S3 bucket using DuckDB.
    Filters for depth > 200m and genetic/material samples.
    """
    # Load configuration
    try:
        config = load_config()
        storage_conf = config.get('storage', {})
        data_conf = config.get('data', {})
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback defaults if config fails
        storage_conf = {'base_path': './data', 'raw_data_dir': 'raw'}
        data_conf = {'obis_s3_url': 's3://obis-open-data/occurrence/*.parquet', 'min_depth': 200}

    # Setup paths
    base_path = Path(storage_conf.get('base_path', './data'))
    raw_dir = base_path / storage_conf.get('raw_data_dir', 'raw')
    output_path = raw_dir / "obis_deepsea_subset.parquet"
    
    # Ensure output directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    # DuckDB setup
    con = duckdb.connect()
    
    try:
        # Install and load httpfs for S3 access
        print("Setting up DuckDB S3 extension...")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        
        # --- FIXED CONFIGURATION FOR ANONYMOUS S3 ACCESS ---
        # Instead of 's3_anonymous=true', we set keys to empty strings.
        # This tells DuckDB to try the request without signing it.
        con.execute("SET s3_region='us-east-1';")
        con.execute("SET s3_access_key_id='';")
        con.execute("SET s3_secret_access_key='';")
        con.execute("SET s3_session_token='';")
        # ---------------------------------------------------

        # Define Query Parameters
        obis_url = data_conf.get('obis_s3_url', 's3://obis-open-data/occurrence/*.parquet')
        min_depth = data_conf.get('min_depth', 200)

        # SQL Query Construction
        # FIX: Accessing fields inside the 'interpreted' struct
        # 'id' was not found, using 'interpreted.occurrenceID'
        query = f"""
        SELECT 
            interpreted.scientificName,
            interpreted.phylum,
            interpreted.class,
            interpreted."order",
            interpreted.family,
            interpreted.genus,
            interpreted.minimumDepthInMeters,
            interpreted.decimalLatitude,
            interpreted.decimalLongitude,
            interpreted.basisOfRecord,
            interpreted.occurrenceID,
            interpreted.year,
            dataset_id
        FROM '{obis_url}'
        WHERE interpreted.minimumDepthInMeters > {min_depth}
        AND (interpreted.basisOfRecord = 'MaterialSample' OR interpreted.occurrenceID LIKE '%DNA%')
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"Executing Query (Limit: {limit})...")
        # print(query) # Uncomment to debug SQL

        # Execute and Copy to Parquet
        copy_query = f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET);"
        con.execute(copy_query)
        print(f"Data saved to {output_path}")

        # Verification with Polars
        print("\n--- Verifying Output with Polars ---")
        df = pl.read_parquet(output_path)
        print("Schema:")
        print(df.schema)
        print(f"\nRow Count: {len(df)}")
        print("\nFirst 5 Rows:")
        print(df.head(5))

    except Exception as e:
        print(f"Query failed: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    # Test run with limit
    print("Starting OBIS Data Ingestion (Test Mode)...")
    fetch_deep_sea_data(limit=1000)
    print("Job Complete.")