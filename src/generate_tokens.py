import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from encodec import EncodecModel
from sklearn.cluster import MiniBatchKMeans
from joblib import load

# Paths
DATASET_PATH = "test-clean-reduced/"  # Carpeta con .flac
OUTPUT_PATH = "tokens/"    # Carpeta donde guardar tokens
KMEANS_PATH = "./kmeans_semantic.joblib"

# Asegurar carpeta de salida
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
encodec = EncodecModel.encodec_model_24khz().to(device).eval()
kmeans = load(KMEANS_PATH)

# Funciones
def extract_semantic_tokens(waveform):
    with torch.no_grad():
        inputs = processor(waveform.squeeze(0).cpu().numpy(), return_tensors="pt", sampling_rate=16000)
        outputs = hubert(**inputs.to(device))
        embeddings = outputs.last_hidden_state.cpu().numpy()
        tokens = kmeans.predict(embeddings.reshape(-1, embeddings.shape[-1]))
        return tokens.astype(np.int64)

def extract_acoustic_tokens(waveform):
    with torch.no_grad():
        waveform = waveform.unsqueeze(0).to(device)
        encoded_frames = encodec.encode(waveform)
        codes = encoded_frames[0][0]  # Primer canal
        return codes.cpu().numpy().astype(np.int64)

# Procesar archivos
files = sorted(f for f in os.listdir(DATASET_PATH) if f.endswith(".flac"))

for filename in tqdm(files, desc="Generando tokens"):
    try:
        path = os.path.join(DATASET_PATH, filename)
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        sem_tokens = extract_semantic_tokens(waveform)
        ac_tokens = extract_acoustic_tokens(waveform)

        base = os.path.splitext(filename)[0]
        np.save(os.path.join(OUTPUT_PATH, f"{base}_semantic.npy"), sem_tokens)
        np.save(os.path.join(OUTPUT_PATH, f"{base}_acoustic.npy"), ac_tokens)

    except Exception as e:
        print(f"[!] Error con {filename}: {e}")
