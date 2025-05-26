from encodec import EncodecModel
import torch
import torchaudio
from pathlib import Path

model = EncodecModel.encodec_model_24khz()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

audio_path = Path("./test-clean/61/70968/61-70968-0000.flac")
audio, sr = torchaudio.load(audio_path)

if audio.shape[0] == 2:
    audio = torch.mean(audio, dim=0, keepdim=True)

if sr != 24000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
    audio = resampler(audio)

audio = audio.unsqueeze(0).to(device)

with torch.no_grad():
    encoded_frames = model.encode(audio)
    # codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
    decoded_audio = model.decode(encoded_frames)

torchaudio.save("audio_reconstruido.wav", decoded_audio.squeeze(0).cpu(), sample_rate=24000)