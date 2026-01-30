import duckdb

def inspect_interpreted():
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET s3_session_token='';")

    print("Inspecting interpreted struct...")
    try:
        # Select one row and get keys of interpreted
        # DuckDB 0.10+ doesn't easily 'describe' a struct field without some tricks, 
        # but we can just print the struct content of one row to see keys.
        con.execute("SELECT interpreted FROM 's3://obis-open-data/occurrence/*.parquet' LIMIT 1;")
        result = con.fetchone()
        if result:
            print(result[0].keys())
        else:
            print("No data found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_interpreted()
