import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from tqdm import tqdm
import os
import sys
from audio_tokenizers import get_acoustic_tokens, get_semantic_tokens
from datasets import Dataset
import torchaudio


DATA_DIR = "../test-clean-reduced"

device = ("cuda" if torch.cuda.is_available() else "cpu")

# 1. Model Configuration
config = GPT2Config(
    vocab_size=1024,  # Semantic tokens vocabulary size
    n_positions=512,
    n_embd=256,
    n_layer=4,
    n_head=4,
    pad_token_id=0,
    add_cross_attention=True
)

# 2. Custom Model Class
class AudioGPT(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Additional projection layer to match acoustic token dimensions
        self.acoustic_proj = nn.Linear(config.n_embd, 8)  # 8 codebooks
        
    def forward(self, input_ids, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=kwargs.get('attention_mask'),
            **kwargs
        )
        
        # Project hidden states to acoustic token space
        hidden_states = outputs[0]  # [batch_size, seq_len, n_embd]
        logits = self.acoustic_proj(hidden_states)  # [batch_size, seq_len, 8]
        
        # Reshape for cross entropy
        batch_size, seq_len, _ = logits.shape
        logits = logits.reshape(batch_size * seq_len, -1)  # [batch*seq, 8]
        
        loss = None
        if labels is not None:
            # Flatten labels and ignore padding (-100)
            labels = labels.reshape(-1)  # [batch*seq]
            loss = nn.functional.cross_entropy(
                logits,
                labels,
                ignore_index=-100
            )
        
        return (loss, logits) if loss is not None else logits

# 3. Initialize Model
model = AudioGPT(config).to(device)

# 4. Data Preparation (Modified)
def prepare_dataset(data_dir):
    semantic_sequences = []
    acoustic_sequences = []
    
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".flac"):
            path = os.path.join(data_dir, file)
            waveform, sr = torchaudio.load(path)
            
            sem_tokens = get_semantic_tokens(waveform, sr)
            ac_tokens = get_acoustic_tokens(waveform, sr)
            
            # Ensure sequences are not too long
            sem_tokens = sem_tokens[:config.n_positions]
            ac_tokens = ac_tokens[:, :config.n_positions]
            
            # Transpose acoustic tokens to [seq_len, num_codebooks]
            acoustic_seq = ac_tokens.T  # Now [seq_len, 8]
            
            semantic_sequences.append(sem_tokens)
            acoustic_sequences.append(acoustic_seq)
    
    return semantic_sequences, acoustic_sequences

# 5. Custom Collator (Updated)
class AudioCollator:
    def __call__(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        
        # Pad input_ids (semantic tokens)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        
        # Pad labels (acoustic tokens) and stack codebooks
        max_len = max(l.shape[0] for l in labels)
        padded_labels = torch.full(
            (len(labels), max_len, 8), -100, dtype=torch.long
        )
        for i, l in enumerate(labels):
            padded_labels[i, :l.shape[0]] = l
        
        return {
            "input_ids": input_ids,
            "labels": padded_labels,
            "attention_mask": (input_ids != 0).int()
        }

# 6. Training Setup
semantic, acoustic = prepare_dataset(DATA_DIR)

dataset = Dataset.from_dict({
    "input_ids": semantic,
    "labels": acoustic
})

training_args = TrainingArguments(
    output_dir="./audio_gpt",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_steps=100,
    report_to="none",
    prediction_loss_only=True,
    gradient_accumulation_steps=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioCollator(),
)

trainer.train()