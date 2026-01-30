import duckdb

def inspect_obis_schema():
    con = duckdb.connect()
    # Install and load httpfs for S3 access
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    # Anonymous S3 Config
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET s3_session_token='';")

    print("Inspecting OBIS S3 Schema...")
    try:
        # Get the schema description
        con.execute("DESCRIBE SELECT * FROM 's3://obis-open-data/occurrence/*.parquet' LIMIT 1;")
        results = con.fetchall()
        
        print(f"{'Column Name':<30} | {'Type':<20}")
        print("-" * 55)
        for row in results:
            print(f"{row[0]:<30} | {row[1]:<20}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_obis_schema()
