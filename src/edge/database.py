import lancedb
import pyarrow as pa
from pathlib import Path
import os
import sys

# Add project root to path
# Note: This is an edge component, but we might run it from root.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config

class BioDB:
    def __init__(self, db_name: str = None, base_path: str = None):
        """
        Initialize the BioDB adapter for LanceDB.
        """
        try:
            config = load_config()
            storage_conf = config.get('storage', {})
        except Exception:
            storage_conf = {'base_path': './data', 'db_name': 'deepbio_index'}
        
        self.base_path = Path(base_path) if base_path else Path(storage_conf.get('base_path', './data'))
        self.db_name = db_name if db_name else storage_conf.get('db_name', 'deepbio_index')
        self.db_path = self.base_path / self.db_name
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "sequences"
        
        # Define Schema strictly? LanceDB can infer, but explicit is better for demo stability
        # Schema: id, scientific_name, sequence, taxonomy, vector, source
        # But we'll let it infer from the first batch for simplicity in this demo script,
        # or we can define a PyArrow schema if needed. 
        # For this task, we will allow inference or PyArrow definition in add_batch.

    def add_batch(self, data, mode="append"):
        """
        Add a batch of data to the database.
        
        Args:
            data: A list of dicts or a Pandas/Polars DataFrame.
            mode: 'append' or 'overwrite'.
        """
        if self.table_name in self.db.table_names():
            tbl = self.db.open_table(self.table_name)
            tbl.add(data, mode=mode)
        else:
            # Create table
            self.db.create_table(self.table_name, data=data)
            
    def search(self, vector: list, limit: int = 5):
        """
        Search for similar sequences.
        """
        if self.table_name not in self.db.table_names():
            print("Table not found.")
            return []
            
        tbl = self.db.open_table(self.table_name)
        
        # LanceDB search
        results = tbl.search(vector)\
            .limit(limit)\
            .to_list()
            
        return results

    def stats(self):
        """
        Return database statistics.
        """
        if self.table_name not in self.db.table_names():
            return {"status": "empty"}
            
        tbl = self.db.open_table(self.table_name)
        return {
            "table": self.table_name,
            "count": len(tbl)
        }

if __name__ == "__main__":
    # Simple test
    print("Testing BioDB...")
    db = BioDB()
    print(f"Connected to {db.db_path}")
    print(db.stats())
