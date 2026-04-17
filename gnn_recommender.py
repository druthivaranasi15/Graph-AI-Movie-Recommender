import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase

# 1. Connect to Neo4j
uri = "bolt://localhost:7687"
# MAKE SURE THIS PASSWORD IS CORRECT
driver = GraphDatabase.driver(uri, auth=("neo4j", "20062010")) 

def load_graph():
    with driver.session() as session:
        # Make sure User and Movie are capitalized exactly like this:
        result = session.run("MATCH (u:User)-[r:RATED]->(m:Movie) RETURN u.id, m.id, r.rating")
        
        edge_index = []
        edge_attr = []
        for record in result:
            # Subtract 1 for 0-based indexing
            u_id = int(record["u.id"]) - 1
            m_id = int(record["m.id"]) - 1
            edge_index.append([u_id, m_id])
            edge_attr.append([float(record["r.rating"])])
            
        if not edge_index:
            raise ValueError("The database is EMPTY. Please check your Neo4j data loading!")
            
        # FORCE LONG (Integer) TYPE FOR EDGE_INDEX
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

# 2. Define the GNN
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

# 3. Execution and Training
try:
    edge_index, edge_labels = load_graph()
    model = MovieGNN(num_users=943, num_movies=1682)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Graph Loaded! Connections: {edge_index.shape[1]}")
    print("Starting Training...")

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(edge_index)
        
        # Select only the indices involved in ratings
        loss = F.mse_loss(out[edge_index[0]], edge_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

except Exception as e:
    print(f"Error: {e}")