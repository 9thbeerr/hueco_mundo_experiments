import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from model import GPTLMHeadModel, TransformerConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from dataloader import create_dataloaders

@dataclass
class TrainConfig:
    batch_size: int = 10
    epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 128

# # Simple text dataset
# class SimpleTextDataset(Dataset):
#     def __init__(self):
#         self.texts = [
#             "hello world",
#             "this is a test",
#             "pytorch dataset",
#             "transformers are powerful",
#             "simple text dataset"
#         ]
#         self.vocab = {word: idx for idx, word in enumerate(set(" ".join(self.texts).split()))}
#         self.vocab["<pad>"] = len(self.vocab)
#         self.max_length = TrainConfig.max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         tokens = self.texts[idx].split()
#         token_ids = [self.vocab[token] for token in tokens]
#         token_ids = token_ids[:self.max_length]
#         padding = [self.vocab["<pad>"]] * (self.max_length - len(token_ids))
#         return torch.tensor(token_ids + padding, dtype=torch.long)

train_config = TrainConfig()


# Model, loss, optimizer
transformerConfig = TransformerConfig(
    vocab_size=len(dataset.vocab),
    max_position_embeddings=512,
    d_model=768,
    n_layer=6,
    n_head=12,
    d_ff=3072
)
GPT = GPTLMHeadModel(transformerConfig)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(GPT.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

# Training loop
for epoch in range(train_config.epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        logits = GPT(batch)
        labels = torch.randint(0, transformerConfig.vocab_size, logits.shape[:-1])  # Fake labels for testing
        loss = criterion(logits.view(-1, transformerConfig.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
