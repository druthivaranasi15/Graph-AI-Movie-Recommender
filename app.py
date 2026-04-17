import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase
import os

# --- 1. MODEL ARCHITECTURE ---
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
AUTH = ("neo4j", "20062010") 
driver = GraphDatabase.driver(URI, auth=AUTH)

@st.cache_resource
def load_assets():
    model = MovieGNN(num_users=943, num_movies=1682)
    if os.path.exists('movie_gnn_model.pth'):
        model.load_state_dict(torch.load('movie_gnn_model.pth'))
    model.eval()
    
    with driver.session() as session:
        result = session.run("MATCH (u:User)-[r:RATED]->(m:Movie) RETURN u.id, m.id")
        edge_index = [[int(rec["u.id"])-1, int(rec["m.id"])-1] for rec in result]
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return model, ei

model, edge_index = load_assets()

# --- 3. DATA FUNCTIONS ---
def get_user_history(user_id):
    with driver.session() as session:
        return list(session.run("""
            MATCH (u:User {id: $uid})-[r:RATED]->(m:Movie)
            RETURN m.id AS id, m.title AS title, r.rating AS rating
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

# NEW: Personalized Top 10 Discovery
def get_personalized_top_10(user_id, model, edge_index):
    # 1. Get list of movies user has ALREADY seen
    history = get_user_history(user_id)
    seen_ids = [int(h['id']) for h in history]
    
    with torch.no_grad():
        out = model(edge_index)
        # 2. Score all movies (indices 943 to 2624 in the embedding matrix)
        # Note: In your model, movie embeddings start after user embeddings
        all_movie_scores = out[943:].flatten().tolist()
        
    # 3. Create a list of (movie_id, score) excluding seen movies
    recommendations = []
    for i, score in enumerate(all_movie_scores):
        movie_id = i + 1
        if movie_id not in seen_ids:
            recommendations.append((movie_id, score))
    
    # 4. Sort by score and take top 10
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_10_ids = recommendations[:10]
    
    # 5. Get titles from Neo4j
    final_list = []
    with driver.session() as session:
        for mid, score in top_10_ids:
            res = session.run("MATCH (m:Movie {id: $mid}) RETURN m.title AS title", mid=mid)
            title = res.single()["title"]
            final_list.append({"title": title, "score": max(1, min(5, score * 5))})
    return final_list

# --- 4. STREAMLIT FRONTEND ---
st.set_page_config(layout="wide", page_title="AI Recommender")
st.title("🎬 Graph-AI Movie Intelligence")
st.markdown("---")

# Sidebar
st.sidebar.header("User Control Panel")
selected_user = st.sidebar.number_input("Target User ID", 1, 943, 85)

# Row 1: History & Personalized Search
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📖 User {selected_user} History")
    history = get_user_history(selected_user)
    if history:
        for rec in history:
            st.write(f"⭐ {rec['rating']} | **{rec['title']}**")
    else:
        st.write("No viewing history found.")

with col2:
    st.subheader("🔍 Individual Prediction")
    target_movie = st.number_input("Enter Movie ID to Check", 1, 1682, 402)
    if st.button("Predict Match Score"):
        with torch.no_grad():
            out = model(edge_index)
            prediction = out[selected_user - 1].item()
            score = max(1, min(5, prediction * 5))
            with driver.session() as session:
                res = session.run("MATCH (m:Movie {id: $mid}) RETURN m.title AS title", mid=target_movie)
                m_title = res.single()["title"] if res.peek() else "Unknown"
            st.metric(label=f"Match for '{m_title}'", value=f"{score:.2f} / 5.0")

# Row 2: Personalized Top 10 (THE NEW FEATURE)
st.markdown("---")
st.header(f"✨ AI Top 10 Picks for User {selected_user}")
st.caption("Movies the user hasn't seen yet, ranked by your GraphSAGE model.")

if st.button("✨ Discover My Personalized Picks"):
    with st.spinner("Analyzing graph relationships..."):
        personal_top = get_personalized_top_10(selected_user, model, edge_index)
        p_cols = st.columns(5)
        p_cols2 = st.columns(5)
        for i, item in enumerate(personal_top):
            target_col = p_cols[i] if i < 5 else p_cols2[i-5]
            with target_col:
                st.success(f"**{item['title']}**")
                st.write(f"Match: {item['score']:.2f}")

# Row 3: Global Trending
st.markdown("---")
st.header("🌟 Trending Worldwide (Global Hits)")
top_10 = get_global_top_10()
g_cols = st.columns(5)
g_cols2 = st.columns(5)
for i, movie in enumerate(top_10):
    target_col = g_cols[i] if i < 5 else g_cols2[i-5]
    with target_col:
        st.info(f"**{movie['title']}**")
        st.caption(f"⭐ {movie['average']:.1f} ({movie['votes']} votes)")
