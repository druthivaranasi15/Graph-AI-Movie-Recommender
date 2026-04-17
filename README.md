# 🎬 Graph-AI Movie Intelligence Dashboard

A high-performance Hybrid Recommendation System that combines **Graph Neural Networks (GNN)** with **Neo4j** to predict user movie preferences.

---

## 🚀 Key Features
- **Intelligent Inference:** Uses a **GraphSAGE** (SAGEConv) model to perform neighborhood aggregation across the graph.
- **Graph-Native Storage:** Leverages **Neo4j** to manage 100,000+ relationships for low-latency data retrieval.
- **Hybrid Strategy:** Provides personalized AI predictions alongside global weighted-average trending metrics.
- **Live Dashboard:** An interactive **Streamlit** interface for real-time testing and visualization.

## 📊 Dataset: MovieLens 100K
The project utilizes the industry-standard MovieLens 100k dataset:
- **Users:** 943
- **Movies:** 1,682
- **Interactions:** 100,000 ratings
- **Graph Density:** High-connectivity nodes modeled in Neo4j for efficient traversal.

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **AI Framework** | PyTorch & PyTorch Geometric |
| **Database** | Neo4j (Graph Database) |
| **Frontend** | Streamlit |

## 🧠 How the AI Works: GraphSAGE
Unlike standard recommendation engines that only look at a user's isolated history, this system uses **GraphSAGE (SAGEConv)**.
1. **Neighborhood Aggregation:** The model "polls" a user's neighbors (movies they liked) and those movies' neighbors (other users who liked them).
2. **Embedding Generation:** It converts these complex relationships into a 16-dimensional vector.
3. **Similarity Scoring:** When you ask for a prediction, the AI calculates how close the "User Vector" is to the "Movie Vector" in that 16D space.

## 📊 System Architecture
The system models the **MovieLens 100k** dataset as a heterogeneous graph. The GNN learns 16-dimensional embeddings for both users and movies by "passing messages" through their interaction history, allowing the model to capture deep relational patterns that standard collaborative filtering often misses.

## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/druthivaranasi15/Graph-AI-Movie-Recommender.git](https://github.com/druthivaranasi15/Graph-AI-Movie-Recommender.git)
   ```
2. **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```
3. **Database Configuration:**
      Ensure your Neo4j Desktop is running and the database is started.
4. **Launch the App:**
     ```
      python -m streamlit run app.py
     ```

## 💡 Key Learnings
- **The Cold Start Problem:** Solved by integrating a "Global Trending" hybrid view for users with no history.
- **Graph Optimization:** Learned how to write efficient Cypher queries to handle 100k relationships without slowing down the UI.
- **Model Checkpointing:** Implemented `.pth` saving/loading to bypass long training times during deployment.

## 📈Results
The model successfully predicts ratings on a scale of 1.0 - 5.0, with built-in logic to handle "High Recommendation" alerts based on the AI match score.
<img width="1911" height="743" alt="image" src="https://github.com/user-attachments/assets/b4e501a6-eab6-48dc-b5c9-c485b038be92" />
<img width="1907" height="791" alt="image" src="https://github.com/user-attachments/assets/48432228-848d-4c2a-b970-cf761295aa62" />

## 🔮 Future Roadmap
- [ ] **Knowledge Graph Integration:** Adding Actor/Director nodes to Neo4j to improve relationship depth.
- [ ] **Explainable AI:** implementing path-tracing to explain *why* a movie was recommended.
- [ ] **Real-time Fine-tuning:** enabling the model to learn from new ratings submitted through the UI.
- [ ] **Cloud Deployment:** Moving the architecture to AWS/Azure for public access.
## 🧪 Experimental Roadmap (Advanced)
- **Semantic Search:** Integrating Vector Embeddings (using `Sentence-Transformers`) to allow users to search for movies by mood (e.g., "Something dark but inspiring") rather than just ID.
- **GraphRAG Implementation:** Connecting a LLM agent to the Neo4j instance to provide natural language explanations for every recommendation.
- **Heterogeneous Graph Sampling:** Moving beyond a simple User-Movie graph to include `Director`, `Genre`, and `Actor` nodes for more nuanced relationship mapping.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
