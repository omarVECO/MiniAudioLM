# generate.py

import torch
from transformers import pipeline
from utils import get_acoustic_tokens, decode_audio
import numpy as np

# Cargar modelo entrenado
generator = pipeline("text-generation", model="./tiny_gpt", tokenizer=None)

# Cargar tokens guardados
tokens_list = np.load("semantic_tokens.npy", allow_pickle=True)
prompt_tokens = tokens_list[0][:10]  # Usar primeros 10 tokens como prompt

# Generar nuevos tokens
generated_tokens = generator(torch.tensor(prompt_tokens), max_length=100, do_sample=True, temperature=0.7)
print("Tokens generados:", generated_tokens)

# Decodificar los tokens generados a audio
from encodec.utils import convert_audio
import torchaudio

def decode_tokens(codes):
    codes = torch.tensor(codes).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        decoded_audio = encodec_model.decode([(codes, None)])
    torchaudio.save("generated_audio.wav", decoded_audio.squeeze(0).cpu(), sample_rate=24000)
    print("✅ Audio generado guardado como 'generated_audio.wav'")