import torch
import torchaudio
from encodec import EncodecModel
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Initialize models once (these could be global or class members)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Acoustic tokenizer (EnCodec)
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.to(device)
encodec_model.set_target_bandwidth(6.0)  # Adjust bandwidth as needed

# Semantic tokenizer (HuBERT + K-Means)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
kmeans = MiniBatchKMeans(n_clusters=1024, random_state=42)  # Initialize with max clusters

def get_acoustic_tokens(waveform: torch.Tensor, sr: int = 24000) -> np.ndarray:
    """
    Extract EnCodec acoustic tokens from waveform
    Returns: [num_codebooks, seq_len] numpy array
    """
    # Ensure proper shape [1, channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    # Resample if needed
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
    
    # Mono conversion
    if waveform.shape[1] > 1:
        waveform = waveform.mean(dim=1, keepdim=True)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        encoded_frames = encodec_model.encode(waveform)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [1, num_codebooks, seq_len]
    
    return codes.squeeze(0).cpu().numpy()  # [num_codebooks, seq_len]

def get_semantic_tokens(waveform: torch.Tensor, sr: int = 16000) -> np.ndarray:
    """
    Extract HuBERT semantic tokens from waveform
    Returns: [seq_len,] numpy array of cluster indices
    """
    # Ensure proper shape [1, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:  # multi-channel
            waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    # Extract features
    inputs = feature_extractor(
        waveform.squeeze(0).numpy(),
        return_tensors="pt",
        sampling_rate=16000
    ).to(device)
    
    with torch.no_grad():
        outputs = hubert_model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 1024]
    
    # Cluster the embeddings
    X = hidden_states.squeeze(0).cpu().numpy()  # [seq_len, 1024]
    
    # Dynamic cluster adjustment
    effective_clusters = min(1024, len(X))
    if effective_clusters < 1024:
        print(f"Warning: Reducing clusters to {effective_clusters} (input length: {len(X)})")
    
    # Use a fresh KMeans for short segments
    local_kmeans = MiniBatchKMeans(n_clusters=effective_clusters, random_state=42)
    semantic_tokens = local_kmeans.fit_predict(X)  # [seq_len,]
    
    return semantic_tokens.astype(np.int32)

# Pre-fit the KMeans model with some data (recommended)
def prefit_kmeans(audio_paths: list, samples: int = 10000):
    """Pre-train the KMeans model with sample data"""
    all_features = []
    for path in audio_paths[:min(10, len(audio_paths))]:  # Use first 10 files
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        inputs = feature_extractor(
            waveform.squeeze(0).numpy(),
            return_tensors="pt",
            sampling_rate=16000
        ).to(device)
        
        with torch.no_grad():
            outputs = hubert_model(**inputs)
            all_features.append(outputs.last_hidden_state.squeeze(0).cpu().numpy())
    
    X = np.concatenate(all_features, axis=0)
    if len(X) > samples:
        X = X[np.random.choice(len(X), samples, replace=False)]
    
    kmeans.fit(X)