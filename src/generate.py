import torch
import numpy as np
from encodec import EncodecModel
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import MiniBatchKMeans
import soundfile as sf

# Configuración device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelos preentrenados
hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
encodec_model = EncodecModel.encodec_model_24khz().to(device).eval()

# Cargar KMeans semántico entrenado
from joblib import load
kmeans_semantic = load("./kmeans_semantic.joblib")

# Definir tu modelo MiniAudioLM (asegúrate que esté en tu PYTHONPATH o mismo folder)
from train import MiniAudioLM  # Cambia esto por la ruta real

# Función para obtener tokens semánticos
def get_semantic_tokens(waveform):
    with torch.no_grad():
        inputs = hubert_processor(waveform.squeeze(0).cpu().numpy(), return_tensors="pt", sampling_rate=16000)
        outputs = hubert_model(**inputs.to(device))
        embeddings = outputs.last_hidden_state.cpu().numpy()
        labels = kmeans_semantic.predict(embeddings.reshape(-1, embeddings.shape[-1]))
    return labels.astype(np.int64)

# Función para generar tokens acústicos a partir de tokens semánticos con el modelo
@torch.no_grad()
def generate_acoustic_tokens(model, semantic_tokens):
    semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.long, device=device).unsqueeze(0)  # batch 1
    logits = model(semantic_tokens)  # Asume que el modelo devuelve logits [1, seq_len, vocab_size]
    acoustic_tokens = torch.argmax(logits, dim=-1)  # [1, seq_len]
    return acoustic_tokens.squeeze(0).cpu().numpy()

# Función para decodificar tokens acústicos en audio
def decode_acoustic_tokens(acoustic_tokens):
    num_quantizers = 8  # usualmente 8 en EnCodec
    seq_len = len(acoustic_tokens) // num_quantizers
    tokens_reshaped = acoustic_tokens[:seq_len * num_quantizers].reshape(num_quantizers, seq_len)  # (8, seq_len)
    tokens_tensor = torch.tensor(tokens_reshaped, device=device, dtype=torch.long)

    # La lista de códigos por canal
    codes_list = [tokens_tensor[i].unsqueeze(0) for i in range(num_quantizers)]  # lista de 8 tensores (1, seq_len)

    encoded_frames = [codes_list]  # Solo UN elemento en la lista

    with torch.no_grad():
        audio = encodec_model.decode(encoded_frames)
    return audio.squeeze(0).cpu().numpy()



def main():
    # Carga modelo entrenado
    model = MiniAudioLM(vocab_size=1024)
    model.load_state_dict(torch.load("mini_audiolm.pth", map_location=device))
    model.to(device)
    model.eval()

    # Carga audio semilla - pon la ruta de tu archivo wav o flac aquí
    waveform, sr = sf.read("test-clean/61-70968-0059.flac")  # o .flac
    assert sr == 16000, "El audio debe ser 16kHz"

    # Convierte a tensor y a mono si tiene canales
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform.unsqueeze(0)  # batch 1 canal

    # Obtener tokens semánticos
    semantic_tokens = get_semantic_tokens(waveform)

    # Generar tokens acústicos con el modelo
    acoustic_tokens = generate_acoustic_tokens(model, semantic_tokens)

    # Decodificar tokens acústicos a audio
    generated_audio = decode_acoustic_tokens(acoustic_tokens)

    # Guardar audio generado
    sf.write("audio_generado.wav", generated_audio, 24000)
    print("✅ Audio generado guardado como audio_generado.wav")

if __name__ == "__main__":
    main()
