import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings():
    """
    Generate embeddings for DNA sequences using DNABERT-2.
    """
    try:
        config = load_config()
        storage_conf = config.get('storage', {})
        model_conf = config.get('model', {})
    except Exception as e:
        print(f"Error loading config: {e}")
        storage_conf = {'base_path': './data', 'raw_data_dir': 'raw'}
        model_conf = {'name': 'zhihan1996/DNABERT-2-117M'}

    base_path = Path(storage_conf.get('base_path', './data'))
    raw_dir = base_path / storage_conf.get('raw_data_dir', 'raw')
    input_file = raw_dir / "demo_dna_sequences.parquet"
    output_file = raw_dir / "deepbio_demo_vectors.parquet"
    
    model_name = model_conf.get('name', "zhihan1996/DNABERT-2-117M")

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    print("Loading sequences...")
    df = pl.read_parquet(input_file)
    print(f"Loaded {len(df)} sequences.")

    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vectors = []
    print("Computing embeddings...")
    
    # Process in batches to avoid OOM
    batch_size = 16
    sequences = df["sequence"].to_list()
    
    total = len(sequences)
    
    for i in range(0, total, batch_size):
        batch = sequences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
        
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # DNABERT-2 usually uses [CLS] token or mean pooling. 
            # The model output structure depends on implementation.
            # Assuming standard HuggingFace output: last_hidden_state is outputs[0]
            
            # Let's use mean pooling for robust sequence representation
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
            
            # Convert to list of floats
            embeddings = embeddings.cpu().numpy().tolist()
            vectors.extend(embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Append empty vectors or zero vectors to keep alignment? 
            # Better to fail or skip. For now, failing.
            return

    # Add vectors to dataframe
    # Make sure vectors list length matches
    if len(vectors) != len(df):
        print(f"Error: Vector count ({len(vectors)}) != Row count ({len(df)})")
        return

    print("Saving vectors...")
    final_df = df.with_columns(pl.Series("vector", vectors))
    final_df.write_parquet(output_file)
    
    print(f"Done. Saved to {output_file}")
    
    # Verify
    print("\n--- Output Verification ---")
    print(final_df.select(["scientific_name", "vector"]).head(1))

if __name__ == "__main__":
    generate_embeddings()
