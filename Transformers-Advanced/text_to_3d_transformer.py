import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return hidden[-1]
    
class GeometryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TextTo3DModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.text_encoder = TextEncoder(num_embeddings, embedding_dim, hidden_dim)
        self.geometry_decoder = GeometryDecoder(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, text):
        hidden = self.text_encoder(text)
        geometry = self.geometry_decoder(hidden)
        return geometry
