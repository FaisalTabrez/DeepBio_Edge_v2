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
        col1, col2 = st.columns([3, 1])
        with col1:
             seq_input = st.text_area("Input DNA Sequence", value=DUMMY_SEQ, height=100)
             
        with col2:
             st.write("##") # Spacer
             analyze_btn = st.button("Analyze Sequence", type="primary", use_container_width=True)

        if analyze_btn:
             if not seq_input:
                 st.error("Please enter a sequence.")
             else:
                 with st.spinner("Processing on Edge Device (CPU)..."):
                     # 1. Embed
                     embedder = get_embedder()
                     try:
                         query_vecs = embedder.embed([seq_input])
                         if not query_vecs:
                             st.error("Embedding failed.")
                             return
                         query_vec = query_vecs[0]
                     except Exception as e:
                         st.error(f"Embedding error: {e}")
                         return
                     
                     # 2. Search
                     results = db.search(query_vec, limit=k_neighbors)
                     
                     if not results:
                         st.warning("No matches found in database.")
                         return
                         
                     top_match = results[0]
                     # LanceDB returns L2 distance by default (0 is identical)
                     # Careful with interpretation: 
                     # If vectors are normalized, Cosine Similarity = 1 - (L2^2 / 2).
                     # Let's assume L2 for thresholds. Small distance = close.
                     distance = top_match.get('_distance', 1.0)
                     
                     # --- DECISION LOGIC ---
                     # We use 'species_threshold' as the cutoff for "Unknown".
                     # Actually, distance < threshold means MATCH.
                     # So if dist = 0.1 and thresh = 0.5 -> MATCH.
                     
                     # Wait, user prompt said: "Exact Match: If distance > Species Threshold -> Show Green"
                     # usually distance is 0 for match. Similarity is 1.
                     # "If distance < Species Threshold" (Small dist) -> Match.
                     # I will assume standard distance logic: Small = Match.
                     
                     # Novelty Logic:
                     # If distance > threshold -> Novel.
                     
                     # Let's align with user request logic but inverted for Distance:
                     # If distance < (2.0 - Species Threshold)? No, let's stick to Distance.
                     # If distance < 0.2 (arbitrary low) -> Match.
                     
                     # User Defaults: Species Threshold 0.97. This looks like Similarity.
                     # If user thinks in Similarity (0 to 1), then Distance = 1 - Similarity (roughly).
                     # So Threshold 0.97 Similarity ~= 0.03 Distance.
                     # Let's map user slider (0.97) to distance threshold (0.03).
                     
                     # Actually, simpler: Let's assume the slider IS the similarity threshold.
                     # So we convert distance to similarity? 
                     # Or we just interpret the slider for Demo purposes.
                     # Let's use specific Logic:
                     # If distance < 0.1 : Match
                     # If distance > 0.1 : Novel
                     
                     # Let's use the Slider value directly as Distance Threshold for clarity in dashboard?
                     # No, User set default 0.97. That definitely implies Similarity.
                     # I will convert for the demo: similarity = 1 / (1 + distance) is a common proxy or just (1-distance).
                     # Let's display Distance and interpret the slider as "Similarity".
                     
                     similarity = 1.0 - (distance / 2.0) # Approx for L2 on normalized vectors
                     
                     st.divider()
                     st.subheader("Result Analysis")
                     
                     col_m1, col_m2, col_m3 = st.columns(3)
                     col_m1.metric("Top Match", top_match['scientific_name'])
                     col_m2.metric("Distance", f"{distance:.4f}")
                     col_m3.metric("Similarity Score", f"{similarity:.4f}")

                     if similarity >= species_threshold:
                         st.success(f"✅ **Confirmed Species Match**: {top_match['scientific_name']}")
                         st.balloons()
                     else:
                         st.warning(f"⚠️ **Potential Novel Taxa Detected** (Similarity < {species_threshold})")
                         
                         st.write("### 🧬 Consensus Classification")
                         # 3. Consensus
                         try:
                             predictor = get_predictor()
                             lineage = predictor.predict_lineage(results)
                             
                             # Visual Badges
                             badge_cols = st.columns(len(lineage) if lineage else 1)
                             if lineage:
                                 for i, (rank, info) in enumerate(lineage.items()):
                                     status = info['status']
                                     text = f"{rank}: **{info['value']}**"
                                     if status == "Confirmed":
                                         badge_cols[i].success(text)
                                     else:
                                         badge_cols[i].warning(text)
                             else:
                                 st.info("No lineage info available for consensus.")
                                 
                         except Exception as e:
                             st.error(f"Classification error: {e}")

                         # 4. Evidence (Pie Chart)
                         st.write("### 🗳️ Neighbor Voting Evidence")
                         families = []
                         for r in results:
                             # Try extracting Family
                             fam = r.get('family') or r.get('Family') or "Unknown"
                             # Parsing fallback
                             if fam == "Unknown" and 'taxonomy' in r:
                                 parts = r['taxonomy'].split(';')
                                 if len(parts) > 4: fam = parts[4].strip()
                             families.append(fam)
                             
                         df_fam = pd.DataFrame(families, columns=['Family']).value_counts().reset_index()
                         
                         fig = px.pie(df_fam, values='count', names='Family', title=f"Top {k_neighbors} Neighbors Distribution")
                         st.plotly_chart(fig, use_container_width=True)
                     
                     with st.expander("View Raw Neighbor Data"):
                         st.dataframe(pd.DataFrame(results))
                         
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
                    
                    # 4. Plot
                    fig = px.scatter_3d(
                        df_pca, x='x', y='y', z='z',
                        color='Type', 
                        hover_name='Species',
                        size='Type',
                        size_max=15, 
                        color_discrete_map={"Background": "grey", "Target": "red"},
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: STATS ---
    with tabs[2]:
        st.header("System Statistics")
        st.json(db.stats())

if __name__ == "__main__":
    main()
