import torch
import torchaudio
from encodec import EncodecModel
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

encodec_model = EncodecModel.encodec_model_24khz().to(device)

# Tokenización semántica
def get_semantic_tokens(waveform):
    with torch.no_grad():
        inputs = hubert_processor(waveform.squeeze(0).numpy(), return_tensors="pt", sampling_rate=16000)
        outputs = hubert_model(**inputs)
        embeddings = outputs.last_hidden_state.cpu().numpy()
        kmeans = MiniBatchKMeans(n_clusters=min(1024, embeddings.shape[0])).fit(embeddings.reshape(-1, embeddings.shape[-1]))
        return kmeans.labels_.astype(np.int64)
    
# Tokenización acústica
def get_acoustic_tokens(waveform):
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        encoded_frames = encodec_model.encode(waveform)
        # encoded_frames[0] es una lista de tensores (uno por canal)
        codes = encoded_frames[0][0]  # Primer canal
    return codes.cpu().numpy().astype(np.int64)

def save_tokens(tokens, path):
    np.save(path, tokens)

def load_tokens(path):
    return np.load(path, allow_pickle=True)


