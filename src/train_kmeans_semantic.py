import os
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans
from joblib import dump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelos HuBERT
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

# Parámetros
DATASET_PATH = "test-clean-reduced/"  # Cambia esto a tu carpeta
KMEANS_OUT = "kmeans_semantic.joblib"
N_CLUSTERS = 1024
MAX_FILES = None  # Si quieres limitar cuántos archivos usar (ej. 100)

def extract_embeddings(waveform):
    with torch.no_grad():
        inputs = processor(
            waveform.squeeze(0).cpu().numpy(),
            return_tensors="pt",
            sampling_rate=16000
        )
        outputs = hubert(**inputs.to(device))
        return outputs.last_hidden_state.cpu().numpy()

def main():
    all_embeddings = []
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.endswith(".flac"))
    if MAX_FILES:
        files = files[:MAX_FILES]

    for filename in tqdm(files, desc="Extrayendo embeddings"):
        path = os.path.join(DATASET_PATH, filename)
        try:
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            emb = extract_embeddings(waveform)
            all_embeddings.append(emb.reshape(-1, emb.shape[-1]))
        except Exception as e:
            print(f"[!] Error con {filename}: {e}")

    # Concatenar todo
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[✔] Total embeddings: {all_embeddings.shape}")

    # Entrenar k-means
    print("[🚀] Entrenando MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=4096)
    kmeans.fit(all_embeddings)
    dump(kmeans, KMEANS_OUT)
    print(f"[✔] KMeans guardado en {KMEANS_OUT}")

if __name__ == "__main__":
    main()
