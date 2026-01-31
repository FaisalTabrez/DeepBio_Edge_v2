import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

class DNAEmbedder:
    def __init__(self, config=None):
        """
        Initializes the Nucleotide Transformer model with Config Patching.
        """
        if config is None:
            config = {}
        self.model_name = config.get("model", {}).get("name", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
        self.device = "cpu" 
        
        print(f"Loading AI Model: {self.model_name} on {self.device.upper()}...")
        
        try:
            # 1. Load Config
            self.model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model_config.output_hidden_states = True 
            
            # --- CONFIG PATCH START ---
            # The checkpoint has weights of size 4096 (SwiGLU), but config says 2048.
            # We manually update the config to match the checkpoint.
            if hasattr(self.model_config, "intermediate_size"):
                # If it's 2048, force it to 4096 to prevent "size mismatch" error
                if self.model_config.intermediate_size == 2048:
                    print(f"Patching model config: Changing intermediate_size from 2048 to 4096")
                    self.model_config.intermediate_size = 4096
            # --- CONFIG PATCH END ---

            # 2. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # 3. Load Model using the PATCHED config
            # Use AutoModelForMaskedLM as it aligns best with the checkpoint structure
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, 
                config=self.model_config, # Passing the modified config
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
        
    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Average the embeddings across the sequence length.
        """
        # Ensure 3D [Batch, Seq, Hidden]
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)
            
        # Ensure 2D Mask [Batch, Seq]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
            
        # Reshape hidden state to match mask batch size if needed
        batch_size, seq_len = attention_mask.shape
        if last_hidden_state.shape[0] != batch_size:
             last_hidden_state = last_hidden_state.view(batch_size, seq_len, -1)

        # Expand mask
        target_size = last_hidden_state.size()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(target_size).float()
        
        # Calculate Mean
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """
        Convert a list of DNA strings to a numpy array of vectors.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        if not sequences:
            return np.array([])

        inputs = self.tokenizer(
            sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract embeddings from the last hidden layer
            # outputs.hidden_states is a tuple, [-1] is the last layer
            token_embeddings = outputs.hidden_states[-1]
            
            embeddings = self._mean_pooling(token_embeddings, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy()

    def embed(self, sequences: list[str]) -> list[list[float]]:
        """
        Wrapper to return list of lists as expected by app.py
        """
        arr = self.embed_batch(sequences)
        return arr.tolist()

# --- Integration Test ---
if __name__ == "__main__":
    print("Running Embedder Test...")
    test_config = {"model": {"name": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"}}
    embedder = DNAEmbedder(test_config)
    
    print("Testing Single Sequence...")
    vector = embedder.embed_batch(["ATCGATCG"])
    print(f"Success! Vector Shape: {vector.shape}")
    
    print("Testing Batch...")
    vectors = embedder.embed_batch(["ATCG", "GGCC"])
    print(f"Batch Shape: {vectors.shape}")
