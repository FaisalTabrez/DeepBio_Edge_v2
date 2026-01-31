import polars as pl
from Bio import Entrez
import time
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config

def enrich_data():
    """
    Enrich OBIS data with DNA sequences from NCBI Nucleotide database.
    Fetches COI or 18S rRNA sequences for top species.
    """
    try:
        config = load_config()
        storage_conf = config.get('storage', {})
    except Exception as e:
        print(f"Error loading config: {e}")
        storage_conf = {'base_path': './data', 'raw_data_dir': 'raw'}

    base_path = Path(storage_conf.get('base_path', './data'))
    raw_dir = base_path / storage_conf.get('raw_data_dir', 'raw')
    input_file = raw_dir / "obis_deepsea_subset.parquet"
    output_file = raw_dir / "demo_dna_sequences.parquet"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    # Load OBIS data
    print("Loading OBIS data...")
    df = pl.read_parquet(input_file)
    
    # Extract unique species
    print("Extracting unique species...")
    # 'interpreted.scientificName' might be nested in struct, but Polars reads parquet structs.
    # We need to handle if columns are structs or not.
    # In fetch_obis.py we selected interpreted columns AS top level names or kept them in struct?
    # The COPY query selected "interpreted.scientificName", so DuckDB usually flattens this 
    # to "scientificName" OR keeps "interpreted" struct depending on exact syntax.
    # Let's inspect schema briefly or assume "scientificName" if flattened, 
    # or "interpreted" struct if not.
    # Actually, "SELECT interpreted.scientificName ..." usually results in a column named "scientificName".
    
    schema_cols = df.columns
    target_col = "scientificName"
    if "scientificName" not in schema_cols:
        # Check if inside a struct
        print(f"Columns found: {schema_cols}")
        if "interpreted" in schema_cols:
            # Flatten or access struct
            # For simplicity, let's try to just get the unique values if we can find the col name
            pass
            
    # Polars robust selection
    try:
         species_list = df.select(pl.col("scientificName")).unique().head(50).to_series().to_list()
    except Exception:
         # Fallback if column name is different
         print(f"Could not find 'scientificName'. Available columns: {df.columns}")
         return

    print(f"Found {len(species_list)} unique species. Fetching DNA for top 50...")

    # Setup Entrez
    Entrez.email = "demo@deepbio.com"
    
    results = []

    for i, species in enumerate(species_list):
        if not species:
            continue
            
        print(f"[{i+1}/{len(species_list)}] Searching for: {species}")
        
        try:
            # Search for COI or 18S
            search_term = f"{species}[Organism] AND (COI OR 18S)"
            handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=5)
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                print(f"  No sequences found for {species}")
                time.sleep(0.5) 
                continue

            # Fetch sequences
            handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            count = 0
            for seq_record in records:
                # Basic parsing from XML
                accession = seq_record.get('TSeq_accver')
                sequence = seq_record.get('TSeq_sequence')
                defline = seq_record.get('TSeq_defline')
                
                if accession and sequence:
                    results.append({
                        "accession_id": accession,
                        "scientific_name": species,
                        "sequence": sequence,
                        "taxonomy": defline # Using defline as proxy for full taxonomy info in this demo
                    })
                    print(f"  Fetched {accession}")
                    count += 1
            
            time.sleep(0.5) # Rate limiting
            
        except Exception as e:
            print(f"  Error fetching {species}: {e}")
            time.sleep(1) # Wait longer on error

    # Save to Parquet
    if results:
        print(f"\nSaving {len(results)} sequences to {output_file}...")
        enrich_df = pl.DataFrame(results)
        enrich_df.write_parquet(output_file)
        print("Done.")
        
        # Verify
        print("\n--- Output Validation ---")
        print(enrich_df.head())
    else:
        print("No sequences fetched.")

if __name__ == "__main__":
    enrich_data()
