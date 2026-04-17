# 🎬 Graph-AI Movie Intelligence Dashboard

A high-performance Hybrid Recommendation System that combines **Graph Neural Networks (GNN)** with **Neo4j** to predict user movie preferences.

---

## 🚀 Key Features
- **Intelligent Inference:** Uses a **GraphSAGE** (SAGEConv) model to perform neighborhood aggregation across the graph.
- **Graph-Native Storage:** Leverages **Neo4j** to manage 100,000+ relationships for low-latency data retrieval.
- **Hybrid Strategy:** Provides personalized AI predictions alongside global weighted-average trending metrics.
- **Live Dashboard:** An interactive **Streamlit** interface for real-time testing and visualization.

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **AI Framework** | PyTorch & PyTorch Geometric |
| **Database** | Neo4j (Graph Database) |
| **Frontend** | Streamlit |

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
## 📈Results
The model successfully predicts ratings on a scale of 1.0 - 5.0, with built-in logic to handle "High Recommendation" alerts based on the AI match score.
<img width="1911" height="743" alt="image" src="https://github.com/user-attachments/assets/b4e501a6-eab6-48dc-b5c9-c485b038be92" />
<img width="1907" height="791" alt="image" src="https://github.com/user-attachments/assets/48432228-848d-4c2a-b970-cf761295aa62" />

