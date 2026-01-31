import sys
from pathlib import Path
import os
import time


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from io import StringIO
from Bio import SeqIO

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.edge.database import BioDB
from src.edge.embedder import DNAEmbedder
from src.edge.taxonomy import TaxonomyPredictor
from src.utils.config import load_config


# Page Config
st.set_page_config(
    page_title="DeepBio-Edge Detective",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED RESOURCES ---
@st.cache_resource
def get_db():
    return BioDB()

@st.cache_resource
def get_embedder():
    config = load_config()
    return DNAEmbedder(config)

@st.cache_resource
def get_predictor():
    return TaxonomyPredictor(consensus_threshold=0.6)

# --- DUMMY SEQUENCE FOR DEMO ---
DUMMY_SEQ = "AACCGGTTTTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("DeepBio-Edge 🌊")
        
        db = get_db()
        stats = db.stats()
        
        if stats.get('count', 0) > 0:
            st.success(f"🟢 Database Connected\n\nIndexed: {stats['count']} sequences")
        else:
            st.error("🔴 Database Empty or Disconnected")
            st.warning("Run `src/edge/load_demo.py` first.")
        
        st.divider()
        st.header("Controls")
        
        species_threshold = st.slider("Species Threshold", 0.0, 2.0, 0.97, 0.01, help="Distance < Threshold = Matches Species")
        novelty_threshold = st.slider("Novelty Threshold", 0.0, 2.0, 0.85, 0.01, help="Distance > Threshold = Novel")
        k_neighbors = st.slider("Consensus Neighbors (k)", 3, 50, 10)
        
        st.divider()
        st.caption("DeepBio Edge v2.0")

    # --- MAIN CONTENT ---
    st.title("🧬 Deep Sea Detective")
    
    tabs = st.tabs(["🔍 Analysis", "🗺️ 3D Discovery Map", "📊 Database Stats"])
    
    # --- TAB 1: ANALYSIS ---
    with tabs[0]:
        st.header("🧬 Sequence Analysis")
        
        # Get cached resources
        embedder = get_embedder()
        predictor = get_predictor()
        
        # --- INPUT SECTION ---
        input_method = st.radio("Input Method:", ["Manual Entry", "FASTA File Upload"], horizontal=True)
        
        sequences_to_analyze = []  # List of tuples: (id, sequence_string)
        
        if input_method == "Manual Entry":
            # Dummy sequence for demo (Deep Sea Isopod partial COI)
            dummy_seq = "AACTTTATATTTTATTTTTGGTGCTTGAGCCGGCATAGTAGGCACTTCTTTAAGAATTCTAATTCGAGCTGAATTAGGACACCCGGGAGCTTTAATTGGAGATGATCAAATTTATAATACTATT"
            user_seq = st.text_area("Enter DNA Sequence:", value=dummy_seq, height=150)
            if user_seq:
                sequences_to_analyze = [("Manual_Input", user_seq.strip())]
                
        else:  # FASTA Upload
            uploaded_file = st.file_uploader("Upload .fasta file", type=["fasta", "fa"])
            if uploaded_file:
                # Parse FASTA from memory
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                for record in SeqIO.parse(stringio, "fasta"):
                    sequences_to_analyze.append((record.id, str(record.seq)))
                st.info(f"Loaded {len(sequences_to_analyze)} sequences from file.")

        # --- ANALYSIS BUTTON ---
        if st.button("Analyze Sequence(s)", type="primary"):
            if not sequences_to_analyze:
                st.error("Please enter a sequence or upload a file.")
            else:
                with st.spinner(f"Analyzing {len(sequences_to_analyze)} sequences..."):
                    
                    # 1. Prepare Data
                    ids = [x[0] for x in sequences_to_analyze]
                    seqs = [x[1] for x in sequences_to_analyze]
                    
                    # 2. Embed Batch
                    vectors = embedder.embed_batch(seqs)
                    
                    # 3. Process Results
                    results_data = []
                    
                    for i, vector in enumerate(vectors):
                        # Search DB
                        neighbors = db.search(vector.tolist(), limit=k_neighbors)
                        
                        if not neighbors:
                            st.error("Database is empty or search failed.")
                            break
                            
                        best_match = neighbors[0]
                        dist = best_match.get('_distance', 999)
                        best_tax = best_match.get('scientific_name', 'Unknown')
                        
                        # Convert distance to similarity (approx)
                        similarity = 1.0 - (dist / 2.0)
                        
                        # Classification
                        status = "Unknown"
                        if similarity >= species_threshold:
                            status = "✅ Known Species"
                        elif similarity >= novelty_threshold:
                            status = "⚠️ Ambiguous"
                        else:
                            status = "🔥 Potential Novel Taxa"
                            
                        results_data.append({
                            "Sequence ID": ids[i],
                            "Status": status,
                            "Distance": round(dist, 4),
                            "Similarity": round(similarity, 4),
                            "Best Match": best_tax,
                            "Neighbors": neighbors  # Keep for detailed view
                        })

                    # 4. DISPLAY RESULTS
                    
                    # If Single Sequence (Manual Mode) - Show Detailed Report
                    if len(sequences_to_analyze) == 1:
                        res = results_data[0]
                        
                        st.divider()
                        st.subheader("Result Analysis")
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Top Match", res['Best Match'])
                        col_m2.metric("Distance", f"{res['Distance']:.4f}")
                        col_m3.metric("Similarity Score", f"{res['Similarity']:.4f}")
                        
                        # Result Banner
                        if "Known" in res['Status']:
                            st.success(f"{res['Status']}: {res['Best Match']}")
                            st.balloons()
                        elif "Novel" in res['Status']:
                            st.warning(f"{res['Status']} (Similarity < {novelty_threshold})")
                            
                            # Consensus Logic
                            st.write("### 🧬 Consensus Classification")
                            try:
                                lineage = predictor.predict_lineage(res['Neighbors'])
                                
                                # Visual Badges
                                badge_cols = st.columns(len(lineage) if lineage else 1)
                                if lineage:
                                    for j, (rank, info) in enumerate(lineage.items()):
                                        status_val = info['status']
                                        text = f"{rank}: **{info['value']}**"
                                        if status_val == "Confirmed":
                                            badge_cols[j].success(text)
                                        else:
                                            badge_cols[j].warning(text)
                                else:
                                    st.info("No lineage info available for consensus.")
                                    
                            except Exception as e:
                                st.error(f"Classification error: {e}")
                            
                            # Evidence (Pie Chart)
                            st.write("### 🗳️ Neighbor Voting Evidence")
                            families = []
                            for r in res['Neighbors']:
                                fam = r.get('family') or r.get('Family') or "Unknown"
                                if fam == "Unknown" and 'taxonomy' in r:
                                    parts = r['taxonomy'].split(';')
                                    if len(parts) > 4: fam = parts[4].strip()
                                families.append(fam)
                                
                            df_fam = pd.DataFrame(families, columns=['Family']).value_counts().reset_index()
                            fig = px.pie(df_fam, values='count', names='Family', title=f"Top {k_neighbors} Neighbors Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"{res['Status']} (Similarity={res['Similarity']:.4f})")
                        
                        with st.expander("View Raw Neighbor Data"):
                            st.dataframe(pd.DataFrame(res['Neighbors']))
                        
                    # If Batch Mode (File Upload) - Show Summary Table
                    else:
                        st.subheader("Batch Analysis Results")
                        df_res = pd.DataFrame(results_data).drop(columns=["Neighbors"])
                        
                        # Highlighting
                        def highlight_novel(val):
                            if 'Novel' in str(val):
                                return 'background-color: #ffcccc'
                            elif 'Known' in str(val):
                                return 'background-color: #ccffcc'
                            else:
                                return 'background-color: #ffffcc'
                        
                        styled_df = df_res.style.applymap(highlight_novel, subset=['Status'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download Button
                        csv = df_res.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results (CSV)", csv, "analysis_results.csv", "text/csv")
                         
    # --- TAB 2: 3D MAP ---
    with tabs[1]:
        st.header("3D Vector Space Discovery")
        if st.button("Generate 3D Plot"):
            with st.spinner("Projecting Manifold..."):
                # 1. Background Data (Random sample)
                # LanceDB doesn't do random easily, take head 100
                # In real app, we'd cache this
                bg_results = db.search([0.0]*768, limit=100)
                
                if not bg_results:
                    st.error("No data.")
                else:
                    vectors = [r['vector'] for r in bg_results]
                    labels = [r['scientific_name'] for r in bg_results]
                    types = ["Background"] * len(vectors)
                    
                    # 2. Add Query (if exists) -> We need the last query? 
                    # For demo, just simulate a "Query" point (or use Input from Tab 1 if we could access state easily)
                    # Let's generate a fake query point slightly offset from center
                    if len(vectors) > 0:
                        center = np.mean(vectors, axis=0) 
                        query_vec = center + np.random.normal(0, 0.05, len(center))
                        vectors.append(query_vec)
                        labels.append("Query Sequence")
                        types.append("Target")
                    
                    # 3. PCA
                    pca = PCA(n_components=3)
                    components = pca.fit_transform(vectors)
                    
                    df_pca = pd.DataFrame(components, columns=['x', 'y', 'z'])
                    df_pca['Species'] = labels
                    df_pca['Type'] = types
                    
                    # Create numeric size column (Background=5, Target=20)
                    df_pca['size'] = df_pca['Type'].map({'Background': 5, 'Target': 20})
                    df_pca['size'] = pd.to_numeric(df_pca['size'])
                    
                    # 4. Plot
                    fig = px.scatter_3d(
                        df_pca, x='x', y='y', z='z',
                        color='Type', 
                        hover_name='Species',
                        size='size',
                        size_max=15, 
                        color_discrete_map={"Background": "grey", "Target": "red"},
                        opacity=0.7,
                        title="Biodiversity Vector Space (3D PCA)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: STATS ---
    with tabs[2]:
        st.header("System Statistics")
        st.json(db.stats())

if __name__ == "__main__":
    main()
