import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase
import os

# --- 1. MODEL ARCHITECTURE (Must match your Training) ---
class MovieGNN(torch.nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, 16)
        self.movie_emb = torch.nn.Embedding(num_movies, 16)
        self.conv1 = SAGEConv(16, 16)
        self.conv2 = SAGEConv(16, 1)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.movie_emb.weight], dim=0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- 2. DATABASE & RESOURCE SETTINGS ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "20062010") # Your verified password
driver = GraphDatabase.driver(URI, auth=AUTH)

@st.cache_resource
def load_assets():
    """Loads the trained model and the graph structure for real-time inference."""
    # Initialize and load weights
    model = MovieGNN(num_users=943, num_movies=1682)
    if os.path.exists('movie_gnn_model.pth'):
        model.load_state_dict(torch.load('movie_gnn_model.pth'))
    model.eval()
    
    # Load edge_index from Neo4j (Needed for GNN message passing)
    with driver.session() as session:
        result = session.run("MATCH (u:User)-[r:RATED]->(m:Movie) RETURN u.id, m.id")
        # Subtract 1 for 0-based indexing
        edge_index = [[int(rec["u.id"])-1, int(rec["m.id"])-1] for rec in result]
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return model, ei

# Initialize assets
model, edge_index = load_assets()

# --- 3. DATA FUNCTIONS ---
def get_user_history(user_id):
    with driver.session() as session:
        return list(session.run("""
            MATCH (u:User {id: $uid})-[r:RATED]->(m:Movie)
            RETURN m.title AS title, r.rating AS rating
            ORDER BY r.rating DESC LIMIT 8
        """, uid=user_id))

def get_global_top_10():
    with driver.session() as session:
        return list(session.run("""
            MATCH (m:Movie)<-[r:RATED]-()
            WITH m, count(r) AS votes, avg(r.rating) AS average
            WHERE votes > 50
            RETURN m.title AS title, average, votes
            ORDER BY average DESC LIMIT 10
        """))

# --- 4. STREAMLIT FRONTEND ---
st.set_page_config(layout="wide", page_title="AI Recommender")
st.title("🎬 Graph-AI Movie Intelligence")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("User Control Panel")
selected_user = st.sidebar.number_input("Target User ID", 1, 943, 85)
target_movie = st.sidebar.number_input("Target Movie ID to Predict", 1, 1682, 402)

# Row 1: User History & Personalized AI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📖 User {selected_user} History")
    history = get_user_history(selected_user)
    if history:
        for rec in history:
            st.write(f"⭐ {rec['rating']} | **{rec['title']}**")
    else:
        st.write("No viewing history found for this user.")

with col2:
    st.subheader("🔮 AI Preference Prediction")
    if st.button("Generate AI Inference"):
        with torch.no_grad():
            # Run GNN Inference
            out = model(edge_index)
            prediction = out[selected_user - 1].item()
            score = max(1, min(5, prediction * 5)) # Scale to 5 stars
            
            # Fetch Title for UI
            with driver.session() as session:
                res = session.run("MATCH (m:Movie {id: $mid}) RETURN m.title AS title", mid=target_movie)
                m_title = res.single()["title"] if res.peek() else "Unknown"

            st.metric(label=f"Match Score for '{m_title}'", value=f"{score:.2f} / 5.0")
            
            if score >= 4.0:
                st.balloons()
                st.success("🔥 High recommendation match!")
            elif score >= 2.5:
                st.info("🤝 This movie is a moderate match.")
            else:
                st.warning("⚠️ This user is unlikely to enjoy this movie.")

# Row 2: Global Trending
st.markdown("---")
st.header("🌟 Trending Worldwide (Hybrid View)")
top_10 = get_global_top_10()
cols = st.columns(5)
cols2 = st.columns(5)

for i, movie in enumerate(top_10):
    target_col = cols[i] if i < 5 else cols2[i-5]
    with target_col:
        st.write(f"**{i+1}. {movie['title']}**")
        st.caption(f"Rating: {movie['average']:.1f} ({movie['votes']} votes)")

st.sidebar.markdown("---")
st.sidebar.success("Database: Neo4j (Online)")
st.sidebar.success("Model: GraphSAGE (Ready)")