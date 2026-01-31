import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

class DNAEmbedder:
    def __init__(self, config=None):
        """
        Initializes the Nucleotide Transformer model.
        """
        if config is None:
            config = {}
        self.model_name = config.get("model", {}).get("name", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
        self.device = "cpu" # Enforce CPU for the laptop demo
        
        print(f"Loading AI Model: {self.model_name} on {self.device.upper()}...")
        
        # Load Tokenizer & Model
        # trusting remote code is needed for some bio-models, but NT-v2 is standard enough.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Load config to verify settings
            self.model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model_config.output_hidden_states = True # Crucial for getting embeddings
            
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, 
                config=self.model_config,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
        
    def _mean_pooling(self, model_output, attention_mask):
        """
        Average the embeddings across the sequence length, ignoring padding.
        """
        # For Nucleotide Transformer/BERT, we usually take the last hidden state.
        # model_output.hidden_states[-1] contains the embeddings.
        token_embeddings = model_output.hidden_states[-1]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
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

        # Tokenize
        inputs = self.tokenizer(
            sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move to CPU (Just in case)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # Normalize (L2) - Important for Cosine Similarity
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
    # Simple test to ensure it runs without crashing
    test_config = {"model": {"name": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"}}
    embedder = DNAEmbedder(test_config)
    
    test_seqs = ["ATCGATCG", "GGCCGGCC"]
    vectors = embedder.embed_batch(test_seqs)
    
    print(f"Success! Generated vectors of shape: {vectors.shape}")
