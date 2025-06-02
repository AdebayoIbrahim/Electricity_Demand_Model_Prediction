# ⚙️ Setup
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data as GeoData
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from google.colab import drive

 # Skip if already done

# 📥 Load Data
df = pd.read_csv('./caiso-electricity.csv')  # UPDATE THIS PATH
df = df.rename(columns=lambda x: x.strip())
df['Timestamp'] = pd.to_datetime(df['UTC Timestamp (Interval Ending)'])

# 🎯 Focus on 3 regions
regions = ['NP-15 LMP', 'SP-15 LMP', 'ZP-26 LMP']
df = df[['Timestamp'] + regions]

# 🔄 Normalize
scaler = MinMaxScaler()
df[regions] = scaler.fit_transform(df[regions])

# 🪟 Sliding Window
window_size = 24
horizon = 1

X, y = [], []
for i in range(len(df) - window_size - horizon):
    X.append(df[regions].iloc[i:i+window_size].values)
    y.append(df[regions].iloc[i+window_size+horizon-1].values)
X, y = np.array(X), np.array(y)

# 🧹 Dataset Class
class ElectricityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dataset = ElectricityDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 🔗 GNN + LSTM Model
class GNNLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, lstm_layers=1, output_dim=3):
        super(GNNLSTM, self).__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, window, 3]
        batch_size, seq_len, num_nodes = x.size()
        x = x.permute(0, 2, 1)  # [batch, nodes, window]

        # GCN per time step across nodes
        edge_index = torch.tensor([[0,1,2,0,1,2],[1,2,0,2,0,1]], dtype=torch.long).to(x.device)  # fully connected
        gnn_out = []
        for t in range(seq_len):
            xt = x[:,:,t].unsqueeze(-1)  # [batch, nodes, 1]
            out = []
            for b in range(batch_size):
                out.append(self.gnn(xt[b], edge_index))
            gnn_out.append(torch.stack(out))
        gnn_out = torch.stack(gnn_out, dim=1)  # [batch, seq, nodes, hidden]
        gnn_out = gnn_out.mean(dim=2)          # [batch, seq, hidden]

        # LSTM
        lstm_out, _ = self.lstm(gnn_out)       # [batch, seq, hidden]
        out = self.fc(lstm_out[:, -1, :])      # [batch, output_dim]
        return out

# 🚂 Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
