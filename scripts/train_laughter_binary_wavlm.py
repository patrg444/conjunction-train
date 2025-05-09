import os
import yaml
import torchaudio
torchaudio.set_audio_backend("sox_io")
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import csv

# Optional: import WavLM from torchaudio.pipelines if available, else fallback to hub
try:
    from torchaudio.pipelines import WAVLM_BASE
    wavlm_bundle = WAVLM_BASE
except ImportError:
    wavlm_bundle = None

class LaughterBinaryDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=16000, max_len=5.0):
        self.entries = []
        self.sample_rate = sample_rate
        self.max_len = max_len
        with open(manifest_path, "r") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                path, start, end, label = row
                # Handle 'N/A' or '-1' as "use full file"
                try:
                    start_f = float(start)
                except Exception:
                    start_f = 0.0
                if end in ("N/A", "-1", "", None):
                    end_f = -1.0
                else:
                    try:
                        end_f = float(end)
                    except Exception:
                        end_f = -1.0
                self.entries.append({
                    "path": path,
                    "start": start_f,
                    "end": end_f,
                    "label": int(label)
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            wav, sr = torchaudio.load(entry["path"], normalize=True)
        except RuntimeError:
            data, sr = sf.read(entry["path"])
            wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        wav = wav.to(torch.float32)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # If segment, crop; else use full
        if entry["end"] > 0:
            start_sample = int(entry["start"] * self.sample_rate)
            end_sample = int(entry["end"] * self.sample_rate)
            wav = wav[:, start_sample:end_sample]
        # If end <= 0, use full file (no crop)
        # Pad or crop to max_len seconds
        max_samples = int(self.max_len * self.sample_rate)
        if wav.shape[1] < max_samples:
            pad = max_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, :max_samples]
        return wav, entry["label"]

def collate_fn(batch):
    wavs, labels = zip(*batch)
    wavs = torch.stack(wavs).squeeze(1)  # [batch, 1, time] -> [batch, time]
    labels = torch.tensor(labels, dtype=torch.long)
    return wavs, labels

class WavLMBinaryClassifier(nn.Module):
    def __init__(self, wavlm_model, embedding_dim, num_classes=2):
        super().__init__()
        self.wavlm = wavlm_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat_list = self.wavlm.extract_features(x)[0]  # list of [B, T', D]
            feats = feat_list[-1]  # last layer features [B, T', D]
            pooled = feats.mean(dim=1)  # global mean-pool over time [B, D]
        out = self.classifier(pooled)
        return out

def load_wavlm_device():
    if wavlm_bundle is not None:
        model = wavlm_bundle.get_model()
        # Try to get embedding dim robustly
        if hasattr(model, "encoder") and hasattr(model.encoder, "embed_dim"):
            embedding_dim = model.encoder.embed_dim
        elif hasattr(model, "projector"):
            embedding_dim = model.projector.out_features
        else:
            # Fallback: run a dummy forward to get output shape
            dummy = torch.randn(1, 16000 * 5)
            with torch.no_grad():
                features = model.extract_features(dummy)[0]
            embedding_dim = features[0].shape[-1]
    else:
        # Fallback: load from torch.hub
        model = torch.hub.load("microsoft/WavLM", "wavlm_base", trust_repo=True)
        if hasattr(model, "cfg") and hasattr(model.cfg, "encoder_embed_dim"):
            embedding_dim = model.cfg.encoder_embed_dim
        else:
            dummy = torch.randn(1, 16000 * 5)
            with torch.no_grad():
                features = model.extract_features(dummy)[0]
            embedding_dim = features[0].shape[-1]
    return model, embedding_dim

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    manifest_path = config["manifest_path"]
    log_dir = config["log_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    epochs = config.get("epochs", 25)
    batch_size = config.get("batch_size", 32)
    lr = float(config.get("learning_rate", 2e-5))
    val_split = config.get("val_split", 0.2)
    sample_rate = config.get("sample_rate", 16000)
    num_classes = config.get("num_classes", 2)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Prepare dataset
    dataset = LaughterBinaryDataset(manifest_path, sample_rate=sample_rate, max_len=5.0)
    # Group by file to prevent leakage
    from collections import defaultdict
    file_to_indices = defaultdict(list)
    for i, e in enumerate(dataset.entries):
        file_to_indices[e["path"]].append(i)
    files = list(file_to_indices.keys())
    random.shuffle(files)
    split_point = int(len(files) * (1 - val_split))
    train_files, val_files = files[:split_point], files[split_point:]
    train_idx = [idx for f in train_files for idx in file_to_indices[f]]
    val_idx = [idx for f in val_files for idx in file_to_indices[f]]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    # Compute class weights for sampler
    labels = [dataset.entries[i]["label"] for i in train_idx]
    class_sample_count = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_sample_count + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoaders: Weighted sampler for balanced training batches
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavlm_model, embedding_dim = load_wavlm_device()
    wavlm_model.to(device)
    wavlm_model.eval()  # Feature extraction only

    model = WavLMBinaryClassifier(wavlm_model, embedding_dim, num_classes=num_classes)
    model.to(device)

    # Optimizer/loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    # Logging
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for wavs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(wavs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * wavs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += wavs.size(0)
        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for wavs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                wavs, labels = wavs.to(device), labels.to(device)
                logits = model(wavs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * wavs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += wavs.size(0)
        val_acc = val_correct / val_samples
        val_loss = val_loss / val_samples
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"Saved new best model at epoch {epoch+1} (val_acc={val_acc:.4f})")

    writer.close()
    print("Training complete. Best val accuracy:", best_val_acc)

if __name__ == "__main__":
    main()
