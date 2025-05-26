import os
import torch
import numpy as np
from torch.utils.data import Dataset

class AudioTokenDataset(Dataset):
    def __init__(self, token_dir, context_len=128):
        self.token_dir = token_dir
        self.context_len = context_len

        self.samples = []
        for name in os.listdir(token_dir):
            if name.endswith("_semantic.npy"):
                base = name.replace("_semantic.npy", "")
                sem_path = os.path.join(token_dir, f"{base}_semantic.npy")
                ac_path = os.path.join(token_dir, f"{base}_acoustic.npy")
                if os.path.exists(sem_path) and os.path.exists(ac_path):
                    self.samples.append((sem_path, ac_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sem_path, ac_path = self.samples[idx]
        semantic = np.load(sem_path)
        acoustic = np.load(ac_path)

        # Cortamos o rellenamos para tener secuencias fijas
        if len(semantic) >= self.context_len:
            semantic = semantic[:self.context_len]
        else:
            semantic = np.pad(semantic, (0, self.context_len - len(semantic)))

        acoustic = acoustic.reshape(-1)  # Aplanar códigos EnCodec
        if len(acoustic) >= self.context_len:
            acoustic = acoustic[:self.context_len]
        else:
            acoustic = np.pad(acoustic, (0, self.context_len - len(acoustic)))

        return {
            "input_ids": torch.tensor(semantic, dtype=torch.long),
            "labels": torch.tensor(acoustic, dtype=torch.long),
        }

import torch.nn as nn

class MiniAudioLM(nn.Module):
    def __init__(self, vocab_size=1024, d_model=512, nhead=8, num_layers=6, context_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc_out(x)

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

dataset = AudioTokenDataset("tokens/", context_len=128)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MiniAudioLM(vocab_size=1024).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        x = batch["input_ids"].to("cuda")
        y = batch["labels"].to("cuda")

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 1024), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "mini_audiolm.pth")
print("✅ Modelo guardado como mini_audiolm.pth")

