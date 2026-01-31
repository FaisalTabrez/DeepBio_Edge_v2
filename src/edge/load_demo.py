import polars as pl
import sys
from pathlib import Path
import os
import ast

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.edge.database import BioDB

def load_demo_data():
    """
    Load pre-computed vectors into LanceDB.
    """
    try:
        config = load_config()
        storage_conf = config.get('storage', {})
    except Exception:
        storage_conf = {'base_path': './data', 'raw_data_dir': 'raw'}
    
    base_path = Path(storage_conf.get('base_path', './data'))
    raw_dir = base_path / storage_conf.get('raw_data_dir', 'raw')
    input_file = raw_dir / "deepbio_demo_vectors.parquet"
    
    if not input_file.exists():
        print(f"Vector file not found at {input_file}. Please run cloud/embed_sequences.py first.")
        return

    print("Loading vectors...")
    df = pl.read_parquet(input_file)
    print(f"Loaded {len(df)} records.")
    
    # Cleaning / Transformation
    # 1. Rename accession_id -> id
    if "accession_id" in df.columns:
        df = df.rename({"accession_id": "id"})
        
    # 2. Add source column
    df = df.with_columns(pl.lit("NCBI_Nucleotide").alias("source"))
    
    # 3. Ensure vector is strictly list of floats (Polars handles this well usually, but let's be safe)
    # If the parquet saved "vector" as list<double>, LanceDB will be happy.
    
    # Initialize DB
    print("Initializing Database...")
    db = BioDB()
    
    # Ingestion
    print("Ingesting data...")
    # Convert to PyArrow or list of dicts for lancedb
    # LanceDB accepts Polars df directly
    db.add_batch(df)
    print("Ingestion complete.")
    
    # Verification
    print("\n--- Verification ---")
    stats = db.stats()
    print(f"DB Stats: {stats}")
    
    if stats.get('count', 0) > 0:
        # Test search
        test_vec = df['vector'][0].to_list() # Get first vector as python list
        print("\nTest Search (Top 2 similar to first record):")
        results = db.search(test_vec, limit=2)
        for r in results:
            # Print minimal info
            print(f"ID: {r.get('id')} | Name: {r.get('scientific_name')} | Score: {r.get('_distance', 'N/A')}")
    else:
        print("Warning: Database seems empty after ingestion.")

if __name__ == "__main__":
    load_demo_data()
