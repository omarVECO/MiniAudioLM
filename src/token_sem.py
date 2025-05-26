from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from pathlib import Path

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

audio_path = Path("./test-clean/61/70968/61-70968-0000.flac")
waveform, sr = torchaudio.load(audio_path)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

inputs = feature_extractor(waveform.squeeze(0).numpy(), return_tensors="pt", sampling_rate=16000)
with torch.no_grad():
    outputs = hubert(**inputs)
    hidden_states = outputs.last_hidden_state

print("Embeddings de Hubert: ", hidden_states.shape)

X = hidden_states.squeeze(0).numpy()
n_clusters = min(1024, X.shape[0])
kmeans = MiniBatchKMeans(n_clusters=n_clusters)
kmeans.fit(X)
semantic_tokens = kmeans.labels_

print("Tokens semanticos: ", semantic_tokens)